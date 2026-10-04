"""Launch patched Armagetron dedicated servers and talk to them over Unix sockets.

Each engine process is one arena. It hosts ``slots`` neural-controlled cycles plus any number
of built-in AI cycles, runs in lockstep (game time only advances when we answer), and sends a
STEP message every decision interval. ``EnginePool`` multiplexes many engines: it collects
whichever STEPs are ready, lets the caller choose actions for all of them in one batch, and
sends the answers back.
"""

from __future__ import annotations

import os
import selectors
import shutil
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import protocol as P

# The patched engine is the rest of this repository (this package lives in <repo>/ai/arma_ai);
# build it into build-dedicated/stage (see ai/README.md). ARMA_ENGINE_ROOT points elsewhere if needed.
REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = Path(os.environ.get("ARMA_ENGINE_ROOT", REPO_ROOT / "build-dedicated/stage/usr/local"))
ENGINE_BIN = ENGINE_ROOT / "bin/armagetronad-dedicated"
ENGINE_DATA = ENGINE_ROOT / "share/games/armagetronad-dedicated"
ENGINE_CONFIG = ENGINE_ROOT / "etc/games/armagetronad-dedicated"


@dataclass
class ArenaConfig:
    """Settings for one arena. Anything in ``extra`` is written verbatim as console settings."""

    slots: int = 2  # neural-controlled cycles
    builtin_ais: int = 0  # built-in AI opponents
    ai_iq: int = 100
    size_factor: float = -3.0
    speed_factor: float = 0.0
    lockstep_dt: float = 0.025  # >= the dedicated server's 0.9/DEDICATED_FPS physics step
    decision_interval: float = 0.05
    debug: int = 0
    extra: dict[str, str] = field(default_factory=dict)

    def settings(self) -> dict[str, str]:
        n_ais = self.slots + self.builtin_ais
        s = {
            "TALK_TO_MASTER": "0",
            "SERVER_IP": "127.0.0.1",
            "PLAY_WITHOUT_HUMANS": "1",
            "NEURAL_SLOTS": str(self.slots),
            "NEURAL_DECISION_INTERVAL": f"{self.decision_interval:g}",
            "NEURAL_END_ROUND_WITHOUT_NEURAL": "1",
            "LOCKSTEP_DT": f"{self.lockstep_dt:g}",
            "LADDERLOG_WRITE_ALL": "0",
            "NEURAL_DEBUG": str(self.debug),
        }
        # With no humans connected the server uses the single-player (SP_) settings; set both.
        # Every cycle gets its own team: TEAMS_MIN teams of exactly one player each, filled with AIs
        # by the team balancer (which always picks the strongest built-in AI character).
        for prefix in ("", "SP_"):
            s.update({
                f"{prefix}NUM_AIS": "0",
                f"{prefix}MIN_PLAYERS": "0",
                f"{prefix}AI_IQ": str(self.ai_iq),
                f"{prefix}AUTO_AIS": "0",
                f"{prefix}AUTO_IQ": "0",
                f"{prefix}SIZE_FACTOR": f"{self.size_factor:g}",
                f"{prefix}SPEED_FACTOR": f"{self.speed_factor:g}",
                f"{prefix}GAME_TYPE": "1",
                f"{prefix}FINISH_TYPE": "3",
                f"{prefix}TEAMS_MIN": str(n_ais),
                f"{prefix}TEAMS_MAX": str(max(16, n_ais)),
                f"{prefix}TEAM_MIN_PLAYERS": "1",
                f"{prefix}TEAM_MAX_PLAYERS": "1",
                f"{prefix}TEAM_BALANCE_WITH_AIS": "1",
                f"{prefix}LIMIT_ROUNDS": "1000000",
                f"{prefix}LIMIT_TIME": "1000000",
                f"{prefix}LIMIT_SCORE": "1000000000",
            })
        s.update(self.extra)
        return s


@dataclass
class SlotStep:
    flags: int
    kills: int
    mask: int
    local: np.ndarray | None = None
    globl: np.ndarray | None = None
    scalars: np.ndarray | None = None

    @property
    def alive(self) -> bool:
        return bool(self.flags & P.FLAG_ALIVE)

    @property
    def needs_action(self) -> bool:
        return bool(self.flags & P.FLAG_NEEDS_ACTION)


@dataclass
class Step:
    engine: int
    round_id: int
    tick: int
    round_time: float
    round_over: bool
    n_alive: int
    n_total: int
    slots: list[SlotStep]


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("engine closed the connection")
        got += k
    return bytes(buf)


def _parse_step(engine: int, payload: bytes) -> Step:
    round_id, tick, round_time, round_over, n_alive, n_total, n = P.STEP_HEAD.unpack_from(payload, 0)
    off = P.STEP_HEAD.size
    slots = []
    g2 = P.GRID * P.GRID
    for _ in range(n):
        flags, kills, mask, _pad = P.SLOT_HEAD.unpack_from(payload, off)
        off += P.SLOT_HEAD.size
        s = SlotStep(flags, kills, mask)
        if flags & P.FLAG_NEEDS_ACTION:
            s.local = np.frombuffer(payload, np.uint8, g2, off).reshape(P.GRID, P.GRID)
            off += g2
            s.globl = np.frombuffer(payload, np.uint8, g2, off).reshape(P.GRID, P.GRID)
            off += g2
            s.scalars = np.frombuffer(payload, np.float32, P.N_SCALARS, off)
            off += P.N_SCALARS * 4
        slots.append(s)
    if off != len(payload):
        raise ValueError(f"STEP payload size mismatch: parsed {off}, got {len(payload)}")
    return Step(engine, round_id, tick, round_time, bool(round_over), n_alive, n_total, slots)


class EnginePool:
    def __init__(self, arenas: list[ArenaConfig], workdir: Path | None = None, base_port: int = 47000,
                 log_engines: bool = False, command=None):
        self.arenas = arenas
        self.n = len(arenas)
        self.base_port = base_port
        self.log_engines = log_engines
        self.command = command or self._engine_command
        self._own_workdir = workdir is None
        self.workdir = Path(workdir or tempfile.mkdtemp(prefix="arma-")).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        # Unix socket paths are limited to ~104 bytes on macOS, so the socket lives in $TMPDIR.
        self._sockdir = Path(tempfile.mkdtemp(prefix="arma"))
        self.sock_path = str(self._sockdir / "brain.sock")
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(self.sock_path)
        self.listener.listen(max(8, self.n))
        self.procs: list[subprocess.Popen | None] = [None] * self.n
        self.conns: list[socket.socket | None] = [None] * self.n
        self.sel = selectors.DefaultSelector()
        self.waiting: set[int] = set()  # engines blocked on our ACTIONS
        for i in range(self.n):
            self._spawn(i)
        self._accept_all()

    # ---------------------------------------------------------------- process management
    def _write_config(self, i: int) -> Path:
        root = self.workdir / f"engine{i:02d}"
        for sub in ("userconfig", "var", "userdata"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        s = self.arenas[i].settings()
        s["SERVER_PORT"] = str(self.base_port + i)
        s["NEURAL_SOCKET"] = self.sock_path
        s["NEURAL_ENGINE_ID"] = str(i)
        (root / "userconfig" / "autoexec.cfg").write_text("".join(f"{k} {v}\n" for k, v in s.items()))
        return root

    @staticmethod
    def _engine_command(i: int, root: Path, sock_path: str, arena: ArenaConfig) -> list[str]:
        return [str(ENGINE_BIN), "--daemon",
                "--datadir", str(ENGINE_DATA), "--configdir", str(ENGINE_CONFIG),
                "--userconfigdir", str(root / "userconfig"), "--vardir", str(root / "var"),
                "--userdatadir", str(root / "userdata")]

    def _spawn(self, i: int) -> None:
        root = self._write_config(i)
        out = open(root / "engine.log", "ab") if self.log_engines else subprocess.DEVNULL
        self.procs[i] = subprocess.Popen(
            self.command(i, root, self.sock_path, self.arenas[i]), cwd=root, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
        )

    def _accept_all(self, timeout: float = 60.0) -> None:
        self.listener.settimeout(timeout)
        pending = {i for i in range(self.n) if self.conns[i] is None}
        while pending:
            try:
                conn, _ = self.listener.accept()
            except socket.timeout as e:
                dead = [i for i in pending if self.procs[i] and self.procs[i].poll() is not None]
                raise RuntimeError(f"engines {sorted(pending)} never connected (exited: {dead})") from e
            conn.setblocking(True)
            mtype, payload = self._read_msg(conn)
            if mtype != P.MSG_HELLO:
                raise RuntimeError(f"expected HELLO, got message type {mtype}")
            engine_id, proto, n_slots, grid, nl, ng, ns, cell, interval = P.HELLO.unpack(payload)
            if (proto, grid, nl, ng, ns) != (P.PROTOCOL_VERSION, P.GRID, P.N_LOCAL_PLANES, P.N_GLOBAL_PLANES, P.N_SCALARS):
                raise RuntimeError(f"engine {engine_id} speaks an incompatible protocol: {(proto, grid, nl, ng, ns)}")
            if engine_id not in pending:
                raise RuntimeError(f"unexpected HELLO from engine {engine_id}")
            if n_slots != self.arenas[engine_id].slots:
                raise RuntimeError(f"engine {engine_id} reports {n_slots} slots, expected {self.arenas[engine_id].slots}")
            self.conns[engine_id] = conn
            self.sel.register(conn, selectors.EVENT_READ, engine_id)
            pending.discard(engine_id)

    def restart(self, i: int) -> None:
        """Kill and relaunch one engine (used if it crashes)."""
        if self.conns[i] is not None:
            self.sel.unregister(self.conns[i])
            self.conns[i].close()
            self.conns[i] = None
        if self.procs[i] and self.procs[i].poll() is None:
            self.procs[i].kill()
            self.procs[i].wait()
        self.waiting.discard(i)
        self._spawn(i)
        self._accept_all()

    def close(self) -> None:
        for c in self.conns:
            if c is not None:
                c.close()
        for p in self.procs:
            if p and p.poll() is None:
                p.terminate()
        for p in self.procs:
            if p:
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
        self.listener.close()
        shutil.rmtree(self._sockdir, ignore_errors=True)
        if self._own_workdir:
            shutil.rmtree(self.workdir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---------------------------------------------------------------- messaging
    @staticmethod
    def _read_msg(conn: socket.socket) -> tuple[int, bytes]:
        magic, mtype, length = P.HEADER.unpack(_recv_exact(conn, P.HEADER.size))
        if magic != P.MAGIC:
            raise ConnectionError(f"bad magic {magic:#x}")
        return mtype, _recv_exact(conn, length) if length else b""

    def poll(self, min_batch: int | None = None, max_wait: float = 0.004, timeout: float = 120.0) -> list[Step]:
        """Collect STEPs from engines that are not already waiting on us.

        Returns once ``min_batch`` engines (default: all running engines) are waiting, or
        ``max_wait`` seconds after the first STEP arrived, whichever is first.
        """
        want = self.n if min_batch is None else min_batch
        steps: list[Step] = []
        first = None
        deadline = time.monotonic() + timeout
        while True:
            now = time.monotonic()
            if len(self.waiting) >= want:
                break
            if first is not None and now - first >= max_wait:
                break
            if now > deadline:
                raise TimeoutError("no engine produced a STEP in time")
            wait = max_wait if first is not None else 0.5
            for key, _ in self.sel.select(timeout=wait):
                i = key.data
                try:
                    mtype, payload = self._read_msg(self.conns[i])
                except ConnectionError:
                    self.restart(i)
                    continue
                if mtype != P.MSG_STEP:
                    raise RuntimeError(f"engine {i}: unexpected message type {mtype}")
                steps.append(_parse_step(i, payload))
                self.waiting.add(i)
                if first is None:
                    first = time.monotonic()
            for i, p in enumerate(self.procs):
                if p is not None and p.poll() is not None and i not in self.waiting:
                    self.restart(i)
        return steps

    def act(self, engine: int, actions: list[int] | np.ndarray) -> None:
        if engine not in self.waiting:
            raise RuntimeError(f"engine {engine} is not waiting for actions")
        body = bytes([len(actions)]) + bytes(int(a) for a in actions)
        self.conns[engine].sendall(P.HEADER.pack(P.MAGIC, P.MSG_ACTIONS, len(body)) + body)
        self.waiting.discard(engine)

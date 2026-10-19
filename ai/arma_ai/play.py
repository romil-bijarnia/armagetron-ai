"""Play against the trained network in the real game.

Starts a local Armagetron server whose AI opponents are driven by a checkpoint, then waits for you
to join from your normal Armagetron client (Play Game > Multiplayer > Custom Connect > 127.0.0.1).

    arma-play                                  # latest checkpoint of runs/blacklight, one neural opponent
    arma-play --neural 3 --builtin 1 --size -1 # three neural opponents plus one built-in AI
"""

from __future__ import annotations

import argparse
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import protocol as P
from .engine import ENGINE_BIN, ENGINE_CONFIG, ENGINE_DATA, _parse_step, _recv_exact
from .model import make_net, net_version, recurrent_context

PROJECT = Path(__file__).resolve().parent.parent
MASK_TABLE = np.array([[(m >> a) & 1 for a in range(P.N_ACTIONS)] for m in range(1 << P.N_ACTIONS)], np.float32)


@dataclass(frozen=True)
class DecisionInput:
    """A private snapshot of the inputs immediately before one live decision."""

    maps: np.ndarray
    scalars: np.ndarray
    mask: np.ndarray
    state: torch.Tensor
    context: torch.Tensor


class PolicyHistory:
    """Life-scoped recurrent state shared by the socket player and policy inspector.

    STEP slots are stable for a life; the engine marks replacement cycles as spawned. Viewers
    can additionally supply WORLD player IDs, so a reused slot cannot inherit another player.
    """

    def __init__(self, net):
        self.net = net
        self.prev: dict[tuple[int, int], np.ndarray] = {}
        self.rounds: dict[int, int] = {}
        self.memory: dict[tuple[int, int], torch.Tensor] = {}
        self.previous_actions: dict[tuple[int, int], int] = {}
        self.observed: dict[tuple[int, int], tuple[int, float]] = {}
        self.identities: dict[tuple[int, int], object] = {}
        self.last_steps: dict[int, tuple[int, int, list[int]]] = {}
        self.last_identities: dict[int, tuple] = {}
        self.model_token = None
        self.memory_gap_resets = 0

    def clear(self, engine: int | None = None) -> None:
        for cache in (self.prev, self.memory, self.previous_actions, self.observed, self.identities):
            for key in list(cache):
                if engine is None or key[0] == engine:
                    del cache[key]
        for cache in (self.rounds, self.last_steps, self.last_identities):
            if engine is None:
                cache.clear()
            else:
                cache.pop(engine, None)

    def _clear_slot(self, key) -> None:
        for cache in (self.prev, self.memory, self.previous_actions, self.observed):
            cache.pop(key, None)

    def begin(self, step, *, identities: dict[int, object] | None = None, generation=None) -> list[int] | None:
        """Reset invalid histories; return cached controls when this STEP was already consumed."""
        token = (id(self.net), generation)
        if token != self.model_token:
            self.clear()
            self.model_token = token
        player_ids = tuple((identities or {}).get(k, getattr(s, "player_id", k))
                           for k, s in enumerate(step.slots))
        tick = int(step.tick)
        previous = self.last_steps.get(step.engine)
        same_players = player_ids == self.last_identities.get(step.engine)
        if previous and previous[:2] == (step.round_id, tick) and same_players:
            return list(previous[2])
        spawned = any(s.flags & P.FLAG_SPAWNED for s in step.slots)
        restarted = bool(previous and previous[0] == step.round_id and tick < previous[1] and spawned)
        if previous and previous[0] == step.round_id and tick < previous[1] and not spawned:
            raise ValueError("An older game observation cannot advance Blacklight's memory")
        if self.rounds.get(step.engine) != step.round_id or restarted:
            self.clear(step.engine)
            self.rounds[step.engine] = step.round_id
        active = {(step.engine, k) for k in range(len(step.slots))}
        for key in list(self.identities):
            if key[0] == step.engine and key not in active:
                self._clear_slot(key)
                del self.identities[key]
        for k, slot in enumerate(step.slots):
            key = (step.engine, k)
            if (self.identities.get(key, player_ids[k]) != player_ids[k]
                    or slot.flags & (P.FLAG_SPAWNED | P.FLAG_DIED) or not slot.alive):
                self._clear_slot(key)
            self.identities[key] = player_ids[k]
            if slot.needs_action and key in self.observed:
                old_tick, old_time = self.observed[key]
                elapsed = float(step.round_time) - old_time
                if tick != old_tick + 1 or elapsed <= 0 or elapsed > .075:
                    self._clear_slot(key)
                    self.memory_gap_resets += 1
        self.last_identities[step.engine] = player_ids
        return None

    def snapshot(self, step, slot: int) -> DecisionInput:
        """Copy the predecision state so inspection cannot alter the running policy."""
        key = (step.engine, slot)
        current = step.slots[slot]
        parameter = next(self.net.parameters())
        previous = self.prev.get(key, current.maps)
        state = self.memory.get(key)
        state = self.net.initial_state(1) if state is None else state
        stamp = self.observed.get(key)
        elapsed = 0.0 if stamp is None else float(step.round_time) - stamp[1]
        context = recurrent_context(
            torch.tensor([self.previous_actions.get(key, -1)], device=parameter.device),
            torch.tensor([elapsed], device=parameter.device)).to(dtype=parameter.dtype)
        return DecisionInput(np.concatenate([current.maps, previous]), np.array(current.scalars),
                             MASK_TABLE[current.mask, :self.net.n_actions].astype(bool),
                             state.detach().clone(), context.detach().clone())

    def commit(self, step, slot: int, action: int, next_state: torch.Tensor) -> None:
        key = (step.engine, slot)
        current = step.slots[slot]
        self.prev[key] = np.array(current.maps)
        self.memory[key] = next_state.detach().clone()
        # At time zero gCycle::Act drops ordinary socket controls because its timer is not
        # running. Teachers provide an explicit executed label instead of a requested move.
        executed = current.teacher_action
        self.previous_actions[key] = (int(executed) if executed is not None else
                                      int(action) if float(step.round_time) > 1e-6 else -1)
        self.observed[key] = (int(step.tick), float(step.round_time))

    def finish(self, step, actions: list[int]) -> None:
        self.last_steps[step.engine] = (step.round_id, int(step.tick), list(actions))


class Brain(PolicyHistory):
    """Loads a checkpoint and picks actions for STEP messages."""

    def __init__(self, checkpoint: Path, device: str | None = None, sample: bool = False):
        self.device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        ck = torch.load(checkpoint, map_location=self.device, weights_only=True)
        net = make_net(net_version(ck["model"])).to(self.device).eval()
        net.load_state_dict(ck["model"])
        super().__init__(net)
        self.update = ck.get("update", 0)
        self.sample = sample
        self._prewarm()

    @torch.no_grad()
    def _prewarm(self) -> None:
        """Pay the first inference cost before starting the server, without creating history."""
        maps = np.zeros((1, 2 * P.N_MAPS, P.GRID, P.GRID), np.uint8)
        feats = np.zeros((1, P.N_SCALARS + P.N_ACTIONS), np.float32)
        feats[:, P.N_SCALARS:] = 1
        # Always greedy: warming a sampling controller must not consume its random draws.
        action, _, _, _ = self.net.act_step(torch.from_numpy(maps).to(self.device),
                                           torch.from_numpy(feats).to(self.device),
                                           self.net.initial_state(1), greedy=True)
        action.cpu().tolist()  # include the same device-to-CPU action transfer as a real decision
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def actions(self, step, *, identities: dict[int, object] | None = None) -> list[int]:
        acts = [0] * len(step.slots)
        duplicate = self.begin(step, identities=identities)
        if duplicate is not None:
            return duplicate
        rows = [k for k, s in enumerate(step.slots) if s.needs_action]
        if not rows:
            self.finish(step, acts)
            return acts
        snapshots = [self.snapshot(step, k) for k in rows]
        maps = np.stack([snapshot.maps for snapshot in snapshots])
        feats = np.stack([np.concatenate([snapshot.scalars, snapshot.mask]) for snapshot in snapshots])
        state = torch.cat([snapshot.state for snapshot in snapshots])
        context = torch.cat([snapshot.context for snapshot in snapshots])
        a, _, _, next_state = self.net.act_step(torch.from_numpy(maps).to(self.device),
                                               torch.from_numpy(feats).to(self.device), state,
                                               greedy=not self.sample, context=context)
        for i, (k, act) in enumerate(zip(rows, a.cpu().tolist())):
            acts[k] = int(act)
            self.commit(step, k, int(act), next_state[i:i + 1])
        self.finish(step, acts)
        return acts


def server_settings(args, sock_path: str) -> dict[str, str]:
    n_ais = args.neural + args.builtin
    s = {
        "TALK_TO_MASTER": "0",
        "SERVER_IP": "127.0.0.1",
        "SERVER_PORT": str(args.port),
        "SERVER_NAME": "Blacklight",
        "NEURAL_NAME": "Blacklight",
        "NEURAL_SOCKET": sock_path,
        "NEURAL_SLOTS": str(args.neural),
        "NEURAL_DECISION_INTERVAL": "0.05",
        "NEURAL_END_ROUND_WITHOUT_NEURAL": "0",
        "NEURAL_CONTROL_AFTER_ROUND": "1",
        "LOCKSTEP_DT": "0",
        "PLAY_WITHOUT_HUMANS": "1" if args.watch else "0",
    }
    for prefix in ("", "SP_"):
        s.update({
            f"{prefix}NUM_AIS": "0",
            f"{prefix}MIN_PLAYERS": "0",
            f"{prefix}AUTO_AIS": "0",
            f"{prefix}AUTO_IQ": "0",
            f"{prefix}SIZE_FACTOR": f"{args.size:g}",
            f"{prefix}SPEED_FACTOR": f"{args.speed:g}",
            f"{prefix}WALLS_LENGTH": f"{args.walls:g}",
            f"{prefix}GAME_TYPE": "1",
            f"{prefix}FINISH_TYPE": "3" if args.watch else "1",
            # you plus one team per AI, each team exactly one cycle
            f"{prefix}TEAMS_MIN": str(n_ais + (0 if args.watch else 1)),
            f"{prefix}TEAMS_MAX": str(max(16, n_ais + 1)),
            f"{prefix}TEAM_MIN_PLAYERS": "1",
            f"{prefix}TEAM_MAX_PLAYERS": "1",
            f"{prefix}TEAM_BALANCE_WITH_AIS": "1",
        })
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=PROJECT / "runs/blacklight/latest.pt")
    ap.add_argument("--neural", type=int, default=1, help="opponents driven by the network")
    ap.add_argument("--builtin", type=int, default=0, help="extra built-in AI opponents")
    ap.add_argument("--size", type=float, default=-3, help="arena SIZE_FACTOR (-3 is the single-player default)")
    ap.add_argument("--speed", type=float, default=0, help="SPEED_FACTOR")
    ap.add_argument("--walls", type=float, default=600, help="trail length in metres (-1 = endless)")
    ap.add_argument("--port", type=int, default=4534)
    ap.add_argument("--sample", action="store_true", help="sample moves instead of always taking the best one")
    ap.add_argument("--watch", action="store_true", help="run rounds without waiting for a human (testing)")
    args = ap.parse_args()

    brain = Brain(args.checkpoint, sample=args.sample)
    root = PROJECT / "runtime" / "play"
    for sub in ("userconfig", "var", "userdata"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    sockdir = Path(tempfile.mkdtemp(prefix="arma"))
    sock_path = str(sockdir / "brain.sock")
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(sock_path)
    listener.listen(1)
    (root / "userconfig" / "autoexec.cfg").write_text(
        "".join(f"{k} {v}\n" for k, v in server_settings(args, sock_path).items()))
    log = open(root / "server.log", "wb")
    proc = subprocess.Popen(
        [str(ENGINE_BIN), "--daemon", "--datadir", str(ENGINE_DATA), "--configdir", str(ENGINE_CONFIG),
         "--userconfigdir", str(root / "userconfig"), "--vardir", str(root / "var"),
         "--userdatadir", str(root / "userdata")],
        cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)

    def shutdown(*_):
        if proc.poll() is None:
            proc.terminate()
        listener.close()
        shutil.rmtree(sockdir, ignore_errors=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print(f"Blacklight server running (checkpoint update {brain.update}, {args.neural} neural + "
          f"{args.builtin} built-in opponents, arena size {args.size:g}).")
    print(f"Join from Armagetron: Play Game > Multiplayer > Custom Connect > 127.0.0.1, port {args.port}.")
    print("Ctrl-C stops the server.")

    last_reported = -1
    while proc.poll() is None:
        listener.settimeout(1.0)
        try:
            conn, _ = listener.accept()  # the server connects once the first round starts
        except socket.timeout:
            continue
        conn.setblocking(True)
        try:
            while True:
                magic, mtype, length = P.HEADER.unpack(_recv_exact(conn, P.HEADER.size))
                payload = _recv_exact(conn, length) if length else b""
                if mtype == P.MSG_HELLO:
                    brain.clear(0)  # a reconnected engine may reuse round, tick and slot IDs
                    continue
                step = _parse_step(0, payload)
                acts = brain.actions(step)
                body = bytes([len(acts)]) + bytes(acts)
                conn.sendall(P.HEADER.pack(P.MAGIC, P.MSG_ACTIONS, len(body)) + body)
                if step.round_over and step.round_id != last_reported:
                    last_reported = step.round_id
                    won = any(s.flags & P.FLAG_WON for s in step.slots)
                    print(time.strftime("%H:%M:%S"), "round over:", "network won" if won else "network lost")
        except ConnectionError:
            conn.close()
    shutdown()


if __name__ == "__main__":
    main()

"""Play against the trained network in the real game.

Starts a local Armagetron server whose AI opponents are driven by a checkpoint, then waits for you
to join from your normal Armagetron client (Play Game > Multiplayer > Custom Connect > 127.0.0.1).

    arma-play                                  # latest checkpoint of runs/main, one neural opponent
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
from pathlib import Path

import numpy as np
import torch

from . import protocol as P
from .engine import ENGINE_BIN, ENGINE_CONFIG, ENGINE_DATA, _parse_step, _recv_exact
from .model import PolicyNet

PROJECT = Path(__file__).resolve().parent.parent
MASK_TABLE = np.array([[(m >> a) & 1 for a in range(P.N_ACTIONS)] for m in range(1 << P.N_ACTIONS)], np.float32)


class Brain:
    """Loads a checkpoint and picks actions for STEP messages."""

    def __init__(self, checkpoint: Path, device: str | None = None, sample: bool = False):
        self.device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        ck = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.net = PolicyNet().to(self.device).eval()
        self.net.load_state_dict(ck["model"])
        self.update = ck.get("update", 0)
        self.sample = sample

    @torch.no_grad()
    def actions(self, step) -> list[int]:
        acts = [0] * len(step.slots)
        rows = [k for k, s in enumerate(step.slots) if s.needs_action]
        if not rows:
            return acts
        maps = np.stack([np.stack([step.slots[k].local, step.slots[k].globl]) for k in rows])
        feats = np.stack([np.concatenate([step.slots[k].scalars, MASK_TABLE[step.slots[k].mask]]) for k in rows])
        a, _, _ = self.net.act(torch.from_numpy(maps).to(self.device), torch.from_numpy(feats).to(self.device),
                               greedy=not self.sample)
        for k, act in zip(rows, a.cpu().tolist()):
            acts[k] = int(act)
        return acts


def server_settings(args, sock_path: str) -> dict[str, str]:
    n_ais = args.neural + args.builtin
    s = {
        "TALK_TO_MASTER": "0",
        "SERVER_IP": "127.0.0.1",
        "SERVER_PORT": str(args.port),
        "SERVER_NAME": "Neural AI",
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
    ap.add_argument("--checkpoint", type=Path, default=PROJECT / "runs/main/latest.pt")
    ap.add_argument("--neural", type=int, default=1, help="opponents driven by the network")
    ap.add_argument("--builtin", type=int, default=0, help="extra built-in AI opponents")
    ap.add_argument("--size", type=float, default=-3, help="arena SIZE_FACTOR (-3 is the single-player default)")
    ap.add_argument("--speed", type=float, default=0, help="SPEED_FACTOR")
    ap.add_argument("--walls", type=float, default=700, help="trail length in metres (-1 = endless)")
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
    print(f"Neural AI server running (checkpoint update {brain.update}, {args.neural} neural + "
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

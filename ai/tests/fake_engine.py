"""A stand-in engine that speaks the wire protocol with random observations.

Used to test the Python side (EnginePool, Trainer bookkeeping) without the real game:
    python tests/fake_engine.py <socket> <engine_id> <slots>
"""

from __future__ import annotations

import random
import socket
import struct
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from arma_ai import protocol as P  # noqa: E402


def send(sock, mtype, payload):
    sock.sendall(P.HEADER.pack(P.MAGIC, mtype, len(payload)) + payload)


def recv_exact(sock, n):
    b = b""
    while len(b) < n:
        c = sock.recv(n - len(b))
        if not c:
            raise SystemExit(0)
        b += c
    return b


def main():
    path, engine_id, slots = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rng = random.Random(engine_id)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(path)
    send(s, P.MSG_HELLO, P.HELLO.pack(engine_id, P.PROTOCOL_VERSION, slots, P.GRID, P.N_LOCAL_PLANES,
                                      P.N_GLOBAL_PLANES, P.N_SCALARS, P.N_MAPS, 2.0, 0.05))
    round_id = 0
    while True:
        round_id += 1
        alive = [True] * slots
        tick = 0
        while True:
            tick += 1
            died = [False] * slots
            for k in range(slots):
                if alive[k] and rng.random() < 0.02:
                    alive[k], died[k] = False, True
            n_alive = sum(alive)
            over = n_alive <= 1 or tick > 300
            body = P.STEP_HEAD.pack(round_id, tick, tick * 0.05, int(over), n_alive, slots, slots)
            for k in range(slots):
                flags = 0
                if alive[k]:
                    flags |= P.FLAG_ALIVE
                if died[k]:
                    flags |= P.FLAG_DIED
                if over and alive[k]:
                    flags |= P.FLAG_WON
                needs = alive[k] and not over
                if needs:
                    flags |= P.FLAG_NEEDS_ACTION
                body += P.SLOT_HEAD.pack(flags, 0, 0b0111, 0)
                if needs:
                    body += np.random.randint(0, 256, P.GRID * P.GRID * P.N_MAPS, dtype=np.uint8).tobytes()
                    body += np.random.randn(P.N_SCALARS).astype(np.float32).tobytes()
            send(s, P.MSG_STEP, body)
            magic, mtype, length = P.HEADER.unpack(recv_exact(s, P.HEADER.size))
            assert magic == P.MAGIC and mtype == P.MSG_ACTIONS
            acts = recv_exact(s, length)
            assert acts[0] == slots and all(a < P.N_ACTIONS for a in acts[1:])
            if over:
                break


if __name__ == "__main__":
    main()

"""Smoke test: run real engines with random actions, report throughput, dump observations.

    python -m arma_ai.smoke --engines 1 --seconds 20
"""

from __future__ import annotations

import argparse
import struct
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

from . import protocol as P
from .engine import ArenaConfig, EnginePool

PROJECT = Path(__file__).resolve().parent.parent

PLANE_COLOURS = {
    # local planes
    "local": [(90, 90, 90), (60, 200, 255), (255, 70, 70), (80, 255, 80), (255, 255, 0), (255, 160, 0),
              (40, 0, 60), (0, 255, 160)],
    "global": [(90, 90, 90), (60, 200, 255), (255, 70, 70), (255, 255, 255), (255, 255, 0), (40, 0, 60),
               (80, 255, 80), (0, 255, 160)],
}


def write_png(path: Path, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].astype(np.uint8).tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def render(packed: np.ndarray, kind: str, scale: int = 6) -> np.ndarray:
    img = np.zeros((*packed.shape, 3), np.float32)
    # draw low-priority planes first so heads/paths stay visible
    order = [6, 0, 2, 3, 1, 5, 7, 4] if kind == "local" else [5, 0, 2, 6, 1, 4, 7, 3]
    for b in order:
        m = (packed >> b) & 1
        img[m.astype(bool)] = PLANE_COLOURS[kind][b]
    if kind == "local":
        img[39:41, 31:33] = (255, 255, 255)  # our cycle sits at the corner of these cells
    return np.kron(img, np.ones((scale, scale, 1))).clip(0, 255)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", type=int, default=1)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--builtin", type=int, default=0)
    ap.add_argument("--size", type=float, default=-3)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--lockstep", type=float, default=0.025)
    ap.add_argument("--dump", type=int, default=6, help="save this many observation images")
    ap.add_argument("--debug", type=int, default=0)
    args = ap.parse_args()

    out = PROJECT / "runtime" / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    arenas = [ArenaConfig(slots=args.slots, builtin_ais=args.builtin, size_factor=args.size,
                          lockstep_dt=args.lockstep, debug=args.debug) for _ in range(args.engines)]
    rng = np.random.default_rng(0)
    stats = Counter()
    dumped = 0
    t0 = time.time()
    with EnginePool(arenas, workdir=out / "engines", log_engines=True) as pool:
        t_start = time.time()
        print(f"engines connected in {t_start - t0:.1f}s")
        game_time = {}
        while time.time() - t_start < args.seconds:
            for st in pool.poll(max_wait=0.002):
                stats["steps"] += 1
                game_time[st.engine] = max(game_time.get(st.engine, 0.0), 0.0)
                if st.round_over:
                    stats["rounds"] += 1
                acts = []
                for k, s in enumerate(st.slots):
                    if s.flags & P.FLAG_DIED:
                        stats["deaths"] += 1
                    if s.flags & P.FLAG_WON:
                        stats["wins"] += 1
                    stats["kills"] += s.kills
                    if s.needs_action:
                        stats["decisions"] += 1
                        allowed = [a for a in range(P.N_ACTIONS) if (s.mask >> a) & 1]
                        # mostly straight so rounds last a while
                        a = 0 if rng.random() < 0.85 or len(allowed) == 1 else int(rng.choice(allowed))
                        acts.append(a)
                        stats[f"act_{P.ACTION_NAMES[a]}"] += 1
                        if dumped < args.dump and st.tick % 40 == 0:
                            img = np.concatenate([render(s.local, "local"), np.full((64 * 6, 12, 3), 255.0),
                                                  render(s.globl, "global")], axis=1)
                            write_png(out / f"obs_e{st.engine}_r{st.round_id}_t{st.tick}_s{k}.png", img)
                            np.save(out / f"obs_e{st.engine}_r{st.round_id}_t{st.tick}_s{k}_scalars.npy", s.scalars)
                            dumped += 1
                    else:
                        acts.append(0)
                pool.act(st.engine, acts)
        elapsed = time.time() - t_start
    print(f"{elapsed:.1f}s wall: {stats['decisions']:,} decisions ({stats['decisions'] / elapsed:,.0f}/s), "
          f"{stats['rounds']} rounds, {stats['deaths']} deaths, {stats['wins']} wins, {stats['kills']} kills")
    print({k: v for k, v in stats.items() if k.startswith("act_")})


if __name__ == "__main__":
    main()

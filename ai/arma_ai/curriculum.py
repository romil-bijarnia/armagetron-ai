"""Varied duel starts that give the policy practice interacting with an opponent."""

from __future__ import annotations

import random


def combat_start(size: float, rng: random.Random) -> tuple[str, list[tuple[float, ...]]]:
    """Legal fresh spawns, with normal speed, rubber and brakes and no prebuilt trails.

    Distances and approach timing vary; translation, reflection and rotation prevent a
    fixed arena location or heading from identifying the right move. Seats are assigned
    separately by the actor. These starts supplement ordinary rounds, not replace them.
    """
    side = 500 * 2 ** (size / 2)
    gap, approach = rng.uniform(8, 16), rng.uniform(14, 24)
    kind = rng.choice(("parallel", "crossing", "head_on"))
    if kind == "parallel":
        lead = rng.uniform(6, 14)
        poses = [(-gap / 2, -lead / 2, 0, 1), (gap / 2, lead / 2, 0, 1)]
    elif kind == "crossing":
        poses = [(-approach, 0, 1, 0), (0, -approach + rng.uniform(-6, 6), 0, 1)]
    else:
        poses = [(-gap / 2, -approach, 0, 1), (gap / 2, approach, 0, -1)]
    scale = min(1, side / 100)
    cx, cy = (side * rng.uniform(.35, .65) for _ in range(2))
    mirror, turns = rng.choice((-1, 1)), rng.randrange(4)
    placed = []
    for x, y, dx, dy in poses:
        x, dx = mirror * x, mirror * dx
        for _ in range(turns):
            x, y, dx, dy = -y, x, -dy, dx
        placed.append((cx + scale * x, cy + scale * y, dx, dy))
    return kind, placed

"""A turn as a sweep: the simulator's arc matches what the game does (CYCLE_TURN_RADIUS)."""
import ctypes
import math

from arma_ai import sim


def run(radius: float):
    L = sim.lib()
    w = L.tr_new()
    try:
        L.tr_reset(w, -2.0, 1, 0, None)
        L.tr_set_turn(w, radius, 0.15, 15.0)
        L.tr_place(w, 0, 100.0, 100.0, 0.0, 1.0)
        acts = (ctypes.c_int * 1)()
        state = (ctypes.c_float * 12)()
        for _ in range(10):  # the game ignores controls at time zero; drive on a bit first
            L.tr_step(w, acts)
        acts[0] = 2  # right
        L.tr_step(w, acts)
        acts[0] = 0
        for _ in range(6):
            L.tr_step(w, acts)
        L.tr_state(w, state)
        trail = (ctypes.c_float * 64)()
        n = L.tr_trail(w, 0, trail, 32)
        points = [(trail[2 * k], trail[2 * k + 1]) for k in range(n)]
        return (state[0], state[1]), (state[2], state[3]), points
    finally:
        L.tr_free(w)


def test_classic_turn_is_instant():
    (x, y), (dx, dy), points = run(0.0)
    assert (round(dx), round(dy)) == (1, 0)
    assert len(points) == 2, "spawn and the one turn"
    assert abs(points[1][0] - 100.0) < 1e-3, "the corner is on the line the cycle came along"


def test_sweep_ends_on_the_axis_and_lays_the_wall_along_the_arc():
    (x, y), (dx, dy), points = run(1.5)
    assert abs(dx - 1) < 1e-6 and abs(dy) < 1e-6, "the sweep ends exactly on the new axis"
    assert len(points) >= 6, f"wall points along the arc, got {len(points)}"
    xs = [p[0] for p in points[1:]]
    assert all(b >= a - 1e-4 for a, b in zip(xs, xs[1:])), "the arc bends monotonically to the right"
    assert 1.2 < xs[-1] - xs[0] < 1.8, f"a 1.5 m arc moves about 1.5 m sideways, got {xs[-1] - xs[0]:.2f}"
    assert y > points[-1][1] - 1e-3 and x > points[-1][0], "the cycle drives on along the new axis"

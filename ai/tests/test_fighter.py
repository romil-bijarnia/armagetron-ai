"""The space fighter as a module of its own: it survives, it hunts, and it is fast enough for the game."""
import ctypes
import time

from arma_ai import sim


def duel(fighter_a: bool, fighter_b: bool, max_decisions: int = 3000):
    L = sim.lib()
    w = L.tr_new()
    try:
        teams = (ctypes.c_int * 2)(0, 1)
        L.tr_reset(w, -2.0, 2, 0, teams)
        state = (ctypes.c_float * 24)()
        acts = (ctypes.c_int * 2)()
        decisions = 0
        plans = 0
        spent = 0.0
        while decisions < max_decisions and not L.tr_over(w):
            for p, fighter in ((0, fighter_a), (1, fighter_b)):
                if fighter and decisions % 2 == 0:
                    t = time.perf_counter()
                    acts[p] = L.tr_fighter_move(w, p, 2, 4, 6)
                    spent += time.perf_counter() - t
                    plans += 1
                elif fighter:
                    acts[p] = 3 if acts[p] in (3, 4, 5) else 0
                else:
                    acts[p] = 0
            L.tr_step(w, acts)
            decisions += 1
        L.tr_state(w, state)
        alive = [bool(state[12 * i + 5]) for i in range(2)]
        return decisions, alive, (spent / plans * 1000 if plans else 0.0)
    finally:
        L.tr_free(w)


def test_fighter_outlives_a_straight_driver():
    decisions, alive, ms = duel(True, False)
    assert not alive[1], "the straight driver must hit something"
    assert alive[0], "the fighter must still be alive when the straight driver dies"
    assert decisions > 40


def test_fighters_do_not_crash_early_and_plan_quickly():
    decisions, alive, ms = duel(True, True, max_decisions=2400)
    print(f"fighter plan: {ms:.2f} ms; the duel {'ended with a kill' if alive.count(True) <= 1 else 'was still on'} after {decisions} decisions")
    assert decisions > 60, "fighters do not crash at the start"
    assert ms < 40, f"a plan took {ms:.1f} ms; too slow for the game's bots"

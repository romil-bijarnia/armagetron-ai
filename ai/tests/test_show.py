"""The terminal arena preserves game information when resized and uses actual headings."""

import io
from types import SimpleNamespace

import pytest
from rich.console import Console

from arma_ai.show import AI_COLORS, Arena, Cycle, Match, build


def match():
    m = Match()
    m.rounds, m.time, m.bounds = 3, 15, (0, 0, 120, 120)
    for i, (dx, dy) in enumerate(((0, 1), (1, 0), (0, -1), (-1, 0))):
        c = Cycle(i, "player", i, AI_COLORS[i], x=25 + 20 * i, y=30 + 15 * i, dx=dx, dy=dy, speed=20)
        c.trail.extend([(10 + 20 * i, 10), (10 + 20 * i, c.y), (c.x, c.y)])
        m.cycles[i] = c
    m.wins = {0: 4, 1: 2}
    return m


@pytest.mark.parametrize("width", [40, 60, 80, 110])
@pytest.mark.parametrize("height", [12, 16, 30])
@pytest.mark.parametrize("empty", [False, True])
def test_arena_and_controls_fit_after_resize(width, height, empty):
    m = Match() if empty else match()
    stream = io.StringIO()
    Console(file=stream, width=width, height=height, color_system=None).print(
        build(m, SimpleNamespace(update=5026), 1, width, height))
    text = stream.getvalue()
    lines = text.splitlines()
    assert len(lines) <= height and all(len(line) <= width for line in lines)
    assert "u5,026" in text and "Round" in text and "Ctrl-C ends this match." in lines[-1]
    assert "¹" not in text and "²" not in text and "⣿" not in text
    if empty:
        assert "Waiting for players" in text
    elif height >= 16:
        for i in range(4):
            assert ("Blacklight" if i == 0 else f"Blacklight {i + 1}") in text
        assert "Speed" in text and "Wins" in text and "20 m/s" in text
    else:
        assert "4/4 alive" in text


def test_heads_show_direction_and_expired_trails_disappear():
    m = match()
    console = Console(width=60, height=25, color_system=None)
    options = console.options.update(width=60, height=25)
    text = "\n".join("".join(segment.text for segment in line) for line in console.render_lines(Arena(m), options))
    assert all(arrow in text for arrow in ("↑", "→", "↓", "←"))
    for c in m.cycles.values():
        c.alive, c.died_at = False, m.time - 9
    text = "".join(segment.text for line in console.render_lines(Arena(m), options) for segment in line)
    assert not text.strip()


def test_dead_player_retains_identity_and_results_without_live_speed():
    m = match()
    m.cycles[1].alive, m.cycles[1].died_at = False, m.time
    m.cycles[1].speed = 1234
    m.last_result = "Blacklight won"
    stream = io.StringIO()
    Console(file=stream, width=80, height=30, color_system=None).print(
        build(m, SimpleNamespace(update=5026), 2, 80, 30))
    text = stream.getvalue()
    assert "Blacklight 2" in text and "out" in text and "1234" not in text
    assert "Last: Blacklight won" in text and "2x" in text


def test_primary_cycle_is_blacklight_and_builtin_names_stay_distinct():
    assert Cycle(1, "old engine name", 0, AI_COLORS[0]).label == "Blacklight"
    assert Cycle(2, "armagetronad", 255, "#ffffff").label == "Armagetronad (bot)"

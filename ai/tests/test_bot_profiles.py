"""Candidate stock-controller profiles are explicit, reproducible and kept out of held-out tests."""

import hashlib
import json
import random
from collections import Counter
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from arma_ai.bot_profiles import (BotProfile, FIGHTER_TEACHERS, HELD_OUT_PROFILES, PROFILES,
    PROPERTY_NAMES, START_STATES, TRAINING_PROFILES, TeacherMatch, get_profile, make_roster, manifest,
    schedule)
from arma_ai.engine import AI_CHARACTERS_FILENAME, ArenaConfig, _profile_names

REPO = Path(__file__).resolve().parents[2]


def test_catalog_preserves_exact_stock_profiles_and_correct_native_states():
    stock = {}
    for line in (REPO / "config/aiplayers.cfg.in").read_text().splitlines():
        if line and not line.startswith("#"):
            name, *values = line.split()
            stock[name] = tuple(map(int, values[:13]))
    assert get_profile("stock_wall").values == stock["ArmagetronAd"]
    assert get_profile("stock_quake").values == stock["Quake"]
    enum_source = (REPO / "src/tron/gAIBase.h").read_text()
    assert enum_source.index("AI_PATH,") < enum_source.index("AI_CLOSECOMBAT,")
    assert dict(START_STATES) == {"survive": 0, "trace": 1, "path": 2, "close_combat": 3}
    assert 8 <= len(PROFILES) <= 12
    assert len({profile.values for profile in PROFILES.values()}) == len(PROFILES)
    for name in FIGHTER_TEACHERS:
        values = get_profile(name).values
        assert values[10] == 1 and 0 <= values[11] <= 3 and 1 <= values[12] <= 5
        assert min(values[i] for i in (0, 1, 2, 6, 7, 8, 9)) >= 9
    assert {profile.values[10] for profile in PROFILES.values()} == {0, 1, 2, 3}


def test_paced_teachers_enter_through_trace_instead_of_the_targetless_combat_or_path_states():
    # Native 0.05-second/no-emergency screening reproduced zero-turn deaths for cold
    # combat/path starts. All 36 revised teacher lives turned after changing this entry.
    for name in (*FIGHTER_TEACHERS, "path_hunter", "trace_defender"):
        profile = get_profile(name)
        assert profile.values[10] == START_STATES["trace"]
        assert profile.values[4] >= 8
    assert get_profile("fighter_trapper").values[11:] == (3, 5)


def test_candidate_catalog_is_immutable_but_manifests_are_independent_serializable_values():
    profile = get_profile("fighter_cutter")
    with pytest.raises(FrozenInstanceError):
        profile.name = "changed"
    with pytest.raises(TypeError):
        PROFILES[profile.name] = profile
    normalized = BotProfile("Valid_1", tuple(float(value) for value in profile.values))
    assert normalized.values == profile.values and all(type(v) is int for v in normalized.values)
    info = manifest()
    assert info["status"] == "candidates_not_ranked"
    assert info["property_order"] == list(PROPERTY_NAMES)
    assert all(not item["ranked"] and item["role"] and item["style"] for item in info["profiles"])
    assert {item["name"] for item in info["profiles"] if item["roles"] == ["evaluation"]} == set(HELD_OUT_PROFILES)
    assert any("braking" in limitation for limitation in info["limitations"])
    assert any("survival" in limitation for limitation in info["limitations"])
    assert json.loads(json.dumps(info)) == info
    info["profiles"][0]["values"][0] = -99
    assert get_profile("stock_wall").values[0] == 10 and manifest()["profiles"][0]["values"][0] == 10


@pytest.mark.parametrize("name", ["", "space name", "line\nNUM_AIS", "nonascii_é", "../file", "_initial", "x" * 49, "a#b"])
def test_profile_names_cannot_inject_engine_config_or_non_ascii_tokens(name):
    with pytest.raises(ValueError, match="ASCII token"):
        BotProfile(name, (1,) * 13)


@pytest.mark.parametrize("index,value", [(0, -1), (0, 11), (0, True), (0, .5), (0, float("nan")),
    (0, float("inf")), (0, "10"), (0, 10**500), (2, 65), (3, 31), (10, 4), (11, 11), (12, 11)])
def test_profile_parameters_reject_unsupported_native_values(index, value):
    values = [1] * 13
    values[index] = value
    with pytest.raises(ValueError, match="must be an integer"):
        BotProfile("test", tuple(values))


@pytest.mark.parametrize("values", [(1,) * 12, (1,) * 14, [1] * 13])
def test_profile_requires_the_exact_thirteen_field_tuple(values):
    with pytest.raises(ValueError, match="exactly 13"):
        BotProfile("test", values)


def test_roster_repeated_families_have_unique_seats_and_exact_reproducibility_hash():
    roster = make_roster("fighter_cutter", ("fighter_cutter", "stock_wall", "fighter_cutter"))
    selected = (roster.controlled_name, *roster.builtin_names)
    assert len(set(selected)) == 4
    assert _profile_names(roster.text) == set(selected)
    rows = [line.split() for line in roster.text.splitlines() if not line.startswith("#")]
    for row, name in zip(rows, ("fighter_cutter", "fighter_cutter", "stock_wall", "fighter_cutter")):
        assert tuple(map(int, row[1:14])) == get_profile(name).values
    assert roster.manifest["sha256"] == hashlib.sha256(roster.text.encode("ascii")).hexdigest()
    assert roster == make_roster("fighter_cutter", ("fighter_cutter", "stock_wall", "fighter_cutter"))
    changed = make_roster("fighter_cutter", ("stock_wall", "fighter_cutter", "fighter_cutter"))
    assert changed.manifest["sha256"] != roster.manifest["sha256"]
    assert roster.manifest["controlled"]["alias"] == roster.controlled_name
    assert tuple(item["alias"] for item in roster.manifest["opponents"]) == roster.builtin_names
    assert json.loads(json.dumps(roster.manifest)) == roster.manifest
    # Validate the exact contents and aliases through the real engine adapter, without launching it.
    settings = ArenaConfig(slots=1, builtin_ais=3, ai_characters=roster.text,
                           controlled_profile=roster.controlled_name, builtin_profiles=roster.builtin_names).settings()
    assert settings["AI_CHARACTER_FILE"] == AI_CHARACTERS_FILENAME
    assert settings["NEURAL_CHARACTER_PROFILE"] == roster.controlled_name
    assert settings["BUILTIN_CHARACTER_PROFILES"].split() == list(roster.builtin_names)


def test_profile_lookup_and_roster_refuse_missing_or_malformed_seats():
    for unknown in (None, [], "missing"):
        with pytest.raises(ValueError, match="Unknown bot profile"):
            get_profile(unknown)
    with pytest.raises(ValueError, match="sequence"):
        make_roster("fighter_rush", "stock_wall")
    with pytest.raises(ValueError, match="Unknown bot profile"):
        make_roster("fighter_rush", ("missing",))
    with pytest.raises(ValueError, match="at most 32"):
        make_roster("fighter_rush", ("stock_wall",) * 32)


@pytest.mark.parametrize("seed", [0, 17, 219])
def test_training_schedule_balances_teachers_match_sizes_and_varied_opponents(seed):
    state = random.getstate()
    matches = schedule(32, seed)
    assert random.getstate() == state
    assert matches == schedule(32, seed) and matches[:5] == schedule(5, seed)
    assert matches != schedule(32, seed + 1)
    teachers = Counter(match.teacher for match in matches)
    assert all(teachers[name] == 6 for name in FIGHTER_TEACHERS)
    assert all(teachers[name] == 2 for name in set(TRAINING_PROFILES) - set(FIGHTER_TEACHERS))
    assert Counter(match.mode for match in matches) == {"duel": 16, "4player": 16}
    for match in matches:
        assert (1 if match.mode == "duel" else 3) == len(match.opponents)
        if match.mode == "4player":
            assert len(set(match.opponents)) == 3
            assert set(match.opponents) & {"stock_wall", "stock_quake"}
            assert set(match.opponents) & {"trace_defender", "path_hunter"}
            assert set(match.opponents) & set(FIGHTER_TEACHERS)
    selected = {name for match in matches for name in (match.teacher, *match.opponents)}
    assert selected <= set(TRAINING_PROFILES) and not selected & set(HELD_OUT_PROFILES)
    assert not {get_profile(name).values for name in selected} & {
        get_profile(name).values for name in HELD_OUT_PROFILES}
    assert all(sum(match.teacher in FIGHTER_TEACHERS for match in matches[:n]) > n / 2 for n in range(1, 33))


@pytest.mark.parametrize("rounds,seed", [(0, 17), (-1, 17), (True, 17), (1.5, 17), (2, True), (2, "17")])
def test_training_schedule_requires_explicit_integer_rounds_and_seed(rounds, seed):
    with pytest.raises(ValueError):
        schedule(rounds, seed)


def test_teacher_matches_only_describe_supported_duels_or_four_player_rosters():
    with pytest.raises(ValueError, match="one duel opponent"):
        TeacherMatch("fighter_rush", ())
    with pytest.raises(ValueError, match="one duel opponent"):
        TeacherMatch("fighter_rush", ("stock_wall", "stock_quake"))
    with pytest.raises(ValueError, match="Unknown bot profile"):
        TeacherMatch("fighter_rush", ("missing",))

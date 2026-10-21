"""Reproducible stock-controller candidates, not a ranking of fighting strength.

The 13 fields come from gAIBase.cpp/gAICharacter.cpp. In particular, the native state
enum is survive=0, trace=1, path=2, close combat=3; aiplayers.cfg.in's state comment
reverses the last two. These profiles change that existing controller's parameters.
They do not add braking to its strategy or change the neural model.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType


PROPERTY_NAMES = ("reaction", "emergency", "range", "trace", "close_combat", "path",
                  "loop", "enemy", "tunnel", "detect_trace", "start_state",
                  "start_straight", "state_change")
START_STATES = MappingProxyType({"survive": 0, "trace": 1, "path": 2, "close_combat": 3})
_SAFE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,47}\Z", flags=re.ASCII)


@dataclass(frozen=True)
class BotProfile:
    name: str
    values: tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.name, str) or not _SAFE_NAME.fullmatch(self.name):
            raise ValueError("A bot profile needs an ASCII token name of 1..48 characters")
        if not isinstance(self.values, tuple) or len(self.values) != len(PROPERTY_NAMES):
            raise ValueError("A bot profile needs a tuple of exactly 13 integer properties")
        normalized = []
        for index, value in enumerate(self.values):
            limit = 64 if index == 2 else 30 if index == 3 else 3 if index == 10 else 10
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not 0 <= value <= limit or not math.isfinite(value) or int(value) != value):
                raise ValueError(f"{PROPERTY_NAMES[index]} must be an integer from 0 to {limit}")
            normalized.append(int(value))
        object.__setattr__(self, "values", tuple(normalized))


# The stock baselines are copied exactly, including the deliberately different trace/path
# preferences and opening delays. New candidates stay within the supported native ranges.
_profiles = (
    BotProfile("stock_wall", (10, 10, 10, 30, 1, 1, 10, 10, 10, 10, 1, 7, 10)),
    BotProfile("stock_quake", (10, 10, 2, 10, 10, 10, 10, 10, 10, 10, 1, 5, 10)),
    # Cold combat/path starts fell into target-less survival and produced straight-only
    # teacher lives. Native screening found trace entry acquires a target while turning;
    # the short state timer and preferences still distinguish these combat candidates.
   # reaction, emergency, range, trace, close_combat, path, loop, enemy, tunnel, detect_trace, start_state,
   # start_straight, state_change)
    BotProfile("fighter_rush", (10, 10, 10, 4, 10, 6, 10, 10, 10, 9, 1, 0, 1)),
    BotProfile("fighter_cutter", (10, 10, 12, 6, 10, 8, 10, 10, 10, 10, 1, 1, 2)),
    BotProfile("fighter_trapper", (9, 10, 12, 8, 10, 9, 10, 9, 10, 10, 1, 3, 5)),
    BotProfile("fighter_counter", (10, 10, 9, 7, 10, 7, 10, 10, 10, 10, 1, 0, 3)),
    BotProfile("path_hunter", (10, 10, 12, 6, 9, 10, 10, 10, 10, 9, 1, 1, 4)),
    BotProfile("trace_defender", (10, 10, 10, 10, 8, 7, 10, 9, 10, 10, 1, 3, 5)),
    BotProfile("holdout_fighter", (9, 10, 12, 6, 10, 8, 10, 9, 10, 9, 3, 2, 4)),
    BotProfile("holdout_path", (10, 9, 14, 6, 9, 10, 10, 10, 9, 10, 2, 1, 5)),
    BotProfile("holdout_defender", (9, 10, 11, 9, 8, 7, 10, 9, 10, 10, 0, 3, 3)),
)
PROFILES = MappingProxyType({profile.name: profile for profile in _profiles})
FIGHTER_TEACHERS = ("fighter_rush", "fighter_cutter", "fighter_trapper", "fighter_counter")
TRAINING_PROFILES = tuple(profile.name for profile in _profiles if not profile.name.startswith("holdout_"))
HELD_OUT_PROFILES = tuple(profile.name for profile in _profiles if profile.name.startswith("holdout_"))
PROFILE_LIMITATIONS = ("The stock strategy does not add braking actions.",
                      "A close-combat start request falls back to survival while no target is selected.")


@dataclass(frozen=True)
class _Description:
    role: str
    style: str
    source: str = "unranked parameter candidate"


_DESCRIPTIONS = MappingProxyType({
    "stock_wall": _Description("stock baseline", "Wall tracing with the original long opening delay.",
                               "config/aiplayers.cfg.in:ArmagetronAd"),
    "stock_quake": _Description("stock baseline", "Short-range, broadly capable stock trace starter.",
                                "config/aiplayers.cfg.in:Quake"),
    "fighter_rush": _Description("close-combat teacher candidate", "Trace entry with no opening delay, a one-second target-acquisition timer and high combat preference."),
    "fighter_cutter": _Description("close-combat teacher candidate", "Trace entry with a two-second target timer, longer sensing and useful path preference."),
    "fighter_trapper": _Description("close-combat teacher candidate", "Trace entry with a longer defensive opening, high combat/path preferences and loop detection."),
    "fighter_counter": _Description("close-combat teacher candidate", "Trace entry with a three-second target timer and strong enemy and tunnel checks."),
    "path_hunter": _Description("path opponent and teacher candidate", "Trace entry to acquire a target before using its high path and combat preferences."),
    "trace_defender": _Description("defensive opponent and teacher candidate", "Starts tracing with high escape, tunnel and trace-detection abilities."),
    "holdout_fighter": _Description("held-out evaluation opponent", "Separate close-combat coefficients and opening; never scheduled for training."),
    "holdout_path": _Description("held-out evaluation opponent", "Separate path coefficients and sensing range; never scheduled for training."),
    "holdout_defender": _Description("held-out evaluation opponent", "Separate survival-state opening and defensive coefficients; never scheduled for training."),
})


def get_profile(name: str) -> BotProfile:
    if not isinstance(name, str) or name not in PROFILES:
        raise ValueError(f"Unknown bot profile {name!r}")
    return PROFILES[name]


def _profile_manifest(name: str) -> dict:
    profile = get_profile(name)
    description = _DESCRIPTIONS[name]
    held_out = name in HELD_OUT_PROFILES
    return {"name": name, "values": list(profile.values), "role": description.role,
            "roles": ["evaluation"] if held_out else ["teacher", "opponent"],
            "style": description.style, "source": description.source,
            "pool": "held_out" if held_out else "training", "ranked": False}


def manifest() -> dict:
    """A fresh JSON-serializable catalog; no candidate has an assumed strength score."""
    return {"schema_version": 1, "status": "candidates_not_ranked",
            "property_order": list(PROPERTY_NAMES), "start_states": dict(START_STATES),
            "limitations": list(PROFILE_LIMITATIONS),
            "training_profiles": list(TRAINING_PROFILES), "held_out_profiles": list(HELD_OUT_PROFILES),
            "profiles": [_profile_manifest(name) for name in PROFILES]}


@dataclass(frozen=True)
class BotRoster:
    text: str
    controlled_name: str
    builtin_names: tuple[str, ...]
    manifest: dict


def make_roster(controlled_profile: str, builtin_profile_names: tuple[str, ...]) -> BotRoster:
    """Serialize exact profiles under unique seat aliases, including repeated families."""
    if not isinstance(builtin_profile_names, (tuple, list)):
        raise ValueError("builtin_profile_names must be a sequence of profile names")
    names = (controlled_profile, *tuple(builtin_profile_names))
    if len(names) > 32:
        raise ValueError("A roster supports at most 32 players")
    profiles = tuple(get_profile(name) for name in names)
    aliases = tuple(f"bl_{'c' if seat == 0 else 'b'}{seat:02}_{profile.name}"
                    for seat, profile in enumerate(profiles))
    lines = ["# Blacklight stock-controller candidates; strength has not been ranked.",
             "# Native states: survive=0 trace=1 path=2 close_combat=3.",
             "# name " + " ".join(PROPERTY_NAMES)]
    lines.extend(alias + "\t" + "\t".join(map(str, profile.values)) + "\t# profile=" + profile.name
                 for alias, profile in zip(aliases, profiles))
    text = "\n".join(lines) + "\n"
    seats = [{**_profile_manifest(name), "alias": alias, "seat": seat}
             for seat, (name, alias) in enumerate(zip(names, aliases))]
    details = {"schema_version": 1, "status": "candidates_not_ranked",
               "sha256": hashlib.sha256(text.encode("ascii")).hexdigest(),
               "limitations": list(PROFILE_LIMITATIONS),
               "property_order": list(PROPERTY_NAMES), "controlled": seats[0], "opponents": seats[1:]}
    return BotRoster(text, aliases[0], aliases[1:], details)


@dataclass(frozen=True)
class TeacherMatch:
    teacher: str
    opponents: tuple[str, ...]

    def __post_init__(self):
        get_profile(self.teacher)
        if not isinstance(self.opponents, tuple) or len(self.opponents) not in (1, 3):
            raise ValueError("A teacher match needs one duel opponent or three four-player opponents")
        for name in self.opponents:
            get_profile(name)

    @property
    def mode(self) -> str:
        return "duel" if len(self.opponents) == 1 else "4player"


def schedule(rounds: int, seed: int) -> list[TeacherMatch]:
    """Seeded, balanced candidate matches with a 3:1 emphasis on close-combat teachers.

    Each complete 16-life block uses every fighter teacher three times and each other
    training profile once. Eight lives are duels and eight have three opponents. Every
    four-player match mixes a stock baseline, a path/defensive bot and a fighter. The
    separate held-out coefficients are absent from every teacher and opponent seat.
    """
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    rng = random.Random(seed)
    result = []
    mode_offset = rng.randrange(2)
    while len(result) < rounds:
        omitted = rng.sample(FIGHTER_TEACHERS, len(FIGHTER_TEACHERS))
        varied = rng.sample(("stock_wall", "stock_quake", "path_hunter", "trace_defender"), 4)
        teachers = []
        for excluded, alternative in zip(omitted, varied):
            group = [name for name in FIGHTER_TEACHERS if name != excluded]
            rng.shuffle(group)
            teachers.extend([*group, alternative])
        duel_opponents = rng.sample(TRAINING_PROFILES, len(TRAINING_PROFILES))
        stocks = rng.sample(("stock_wall", "stock_quake"), 2)
        defenders = rng.sample(("trace_defender", "path_hunter"), 2)
        fighters = rng.sample(FIGHTER_TEACHERS, len(FIGHTER_TEACHERS))
        duel_index = multi_index = 0
        for index, teacher in enumerate(teachers):
            multi = bool((index // 4 + index % 2 + mode_offset) % 2)
            if multi:
                opponents = [stocks[multi_index % 2], defenders[multi_index % 2],
                             fighters[multi_index % len(fighters)]]
                rng.shuffle(opponents)
                multi_index += 1
            else:
                opponents = [duel_opponents[duel_index]]
                duel_index += 1
            result.append(TeacherMatch(teacher, tuple(opponents)))
            if len(result) == rounds:
                break
    return result

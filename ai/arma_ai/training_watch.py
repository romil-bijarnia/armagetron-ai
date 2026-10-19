"""Low-rate observation of the actual self-play actor; never advances the simulator."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .watch_server import read_record

MOVES = ("Straight", "Left", "Right", "Brake", "Left + brake", "Right + brake")
COLORS = ("#38d8e8", "#ff983c", "#bd9aef", "#b5d876", "#ed8faf", "#eee18a", "#82aaf0", "#ededed")


def match_results(windows) -> dict:
    """Use observed outcomes, including their denominator, rather than reverse-engineering rates."""
    result = {}
    for key, values in windows.items():
        if not key.startswith("win_") or not values:
            continue
        wins = sum(x == 1 for x in values)
        draws = sum(x == .5 for x in values)
        losses = sum(x == 0 for x in values)
        n = wins + draws + losses
        result[key] = {"wins": wins, "draws": draws, "losses": losses, "games": n,
                       "score": (wins + .5 * draws) / n if n else None}
    return result


class TrainingPublisher:
    def __init__(self, run: Path, arenas: int):
        self.run, self.arenas = run, arenas
        self.arena = 0
        self.last_request = self.last_publish = 0.0
        self.progress = {}
        self.warned = False

    def publish(self, actor, *, status: str = "running", force: bool = False) -> None:
        now = time.time()
        if now - self.last_request >= 1:
            request = read_record(self.run / "watch_request.json")
            arena = request.get("arena", self.arena)
            if isinstance(arena, int) and not isinstance(arena, bool) and 0 <= arena < self.arenas:
                self.arena = arena
            self.progress = read_record(self.run / "progress.json")
            self.last_request = now
        if not force and now - self.last_publish < .2:
            return
        if self.arena not in actor.kind:
            return
        snapshot = actor.sp.watch(self.arena)
        (kind, detail), hero, agents = actor.kind[self.arena]
        labels = {"self": "Self-play", "hunter": "Current policy vs space fighter",
                  "exploiter": "Exploiter vs frozen target"}
        names = []
        for player in snapshot["players"]:
            agent = player["agent_id"]
            if agent == -1:
                name = "Space fighter"
            elif agent == 0:
                name = "Current policy"
                if agents.count(0) > 1:
                    name += f" {player['id'] + 1}"
            elif agent == 1:
                name = f"Exploiter {actor.exploiter_gen}"
            elif agent == 2:
                name = "Frozen target"
            else:
                name = f"Opponent {agent}"
            player["name"] = name
            player["action_name"] = MOVES[player["action"]] if player["action"] is not None else "Waiting"
            names.append(name)
        for trail in snapshot["trails"]:
            trail["name"] = names[trail["id"]]
            trail["color"] = COLORS[(trail["id"] - hero) % len(agents)]
        snapshot.update(
            v=2, source="training_actor", actor_pid=os.getpid(), trainer_pid=os.getppid(), timestamp=now, status=status,
            round_id=actor.round_ids[self.arena], matchup=labels[kind], matchup_kind=kind,
            arena_kind=detail, opponent_names=names, me=hero,
            start_kind=getattr(actor, "start_kinds", {}).get(self.arena, "standard"),
            loaded_models={str(k): {"update": actor.model_updates.get(k), "version": v.version}
                           for k, v in actor.nets.items()},
            trainer_update=self.progress.get("update"), trainer_timestamp=self.progress.get("time"),
            cycles=[[p["id"], p["x"], p["y"], int(p["alive"]), snapshot["trails"][p["id"]]["color"], p["id"]]
                    for p in snapshot["players"]],
        )
        path = self.run / "watch.json"
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(snapshot, allow_nan=False, separators=(",", ":")) + "\n")
            tmp.replace(path)
        except (OSError, ValueError) as exc:
            if not self.warned:
                print(f"could not publish training arena: {exc}", flush=True)
                self.warned = True
        self.last_publish = now

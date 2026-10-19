"""Direct arma-* commands for the local training controller."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
CONTROLLER = PROJECT / "trainctl"
DESCRIPTIONS = {
    "background": "Resume Blacklight training with the background resource budget.",
    "full": "Resume Blacklight training without throttling and keep the Mac awake.",
    "stop": "Save the checkpoint and stop training.",
    "status": "Show trainer status and the latest saved checkpoint.",
    "watch": "Show live progress bars. Ctrl-C closes this view while training continues.",
    "log": "Follow the training log. Ctrl-C closes the log view.",
}


def _run(command: str) -> None:
    args = sys.argv[1:]
    if "--help" in args or "-h" in args:
        if command in ("background", "full"):
            # Help must never launch a detached trainer or create a run directory.
            from .teach import parse_config
            parse_config(["--mode", command, *[arg for arg in args if arg != "--no-watch"]])
            return
        parser = argparse.ArgumentParser(prog=f"arma-{command}", description=DESCRIPTIONS[command])
        parser.add_argument("--run", default="runs/blacklight", help="run directory relative to the AI project")
        if command == "watch":
            parser.add_argument("--once", action="store_true", help="print one frame and exit")
        parser.parse_args(args)
        return
    # Reuse the established supervisor, save/stop and terminal behavior. Arguments stay
    # separate throughout, including run paths containing spaces or shell characters.
    os.execv(str(CONTROLLER), [str(CONTROLLER), command, *args])


def background() -> None:
    _run("background")


def full() -> None:
    _run("full")


def stop() -> None:
    _run("stop")


def status() -> None:
    _run("status")


def watch() -> None:
    _run("watch")


def log() -> None:
    _run("log")

"""./blacklight is the plain-words front door to trainctl; it must never start anything by accident."""

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "blacklight"


def run(*args):
    return subprocess.run([str(SCRIPT), *args], capture_output=True, text=True, timeout=30)


def test_help_lists_every_command_and_starts_nothing():
    for args in ((), ("help",), ("--help",)):
        result = run(*args)
        assert result.returncode == 0 and result.stderr == ""
        for command in ("train", "train full", "stop", "status", "watch", "dash", "log", "ladder", "show", "brain"):
            assert f"./blacklight {command}" in result.stdout
        assert result.stdout.rstrip().endswith("command list.")


def test_unknown_command_fails_with_the_command_list():
    result = run("bogus")
    assert result.returncode == 1 and result.stdout == ""
    assert "unknown command: bogus" in result.stderr and "./blacklight train" in result.stderr

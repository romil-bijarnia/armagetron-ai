"""Direct commands preserve controller arguments and make help side-effect free."""

import json
import subprocess
import sys
import pytest


@pytest.mark.parametrize("command", ["background", "full", "stop", "status", "watch", "log"])
def test_direct_commands_forward_literal_arguments_without_a_shell(tmp_path, command):
    controller = tmp_path / "controller with spaces"
    controller.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    controller.chmod(0o755)
    args = ["--run", "runs/a space; $(do-not-execute)"]
    if command in ("background", "full"):
        args += ["--combat-prob", "0.5", "--no-watch"]
    elif command == "watch":
        args += ["--once"]
    code = ("from pathlib import Path; from arma_ai import control; "
            f"control.CONTROLLER=Path({str(controller)!r}); control.{command}()")
    result = subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True,
                            timeout=10, check=True)
    assert json.loads(result.stdout) == [command, *args]


@pytest.mark.parametrize("command", ["background", "full", "stop", "status", "watch", "log"])
def test_help_never_runs_the_controller_or_creates_a_training_run(tmp_path, command):
    run = tmp_path / "must not be created"
    code = ("from pathlib import Path; from arma_ai import control; "
            "control.CONTROLLER=Path('/missing-controller-help-must-not-exec'); "
            f"control.{command}()")
    result = subprocess.run([sys.executable, "-c", code, "--run", str(run), "--help"],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "--run" in result.stdout
    assert not run.exists()
    assert "training started" not in result.stdout

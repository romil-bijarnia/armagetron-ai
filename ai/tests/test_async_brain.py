"""Local inference queue lifetime/ordering checks require neither a game nor a GPU."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_async_inference_lifecycle_and_stale_results(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("no C++ compiler")
    source = Path(__file__).parent / "cpp/async_brain_check.cpp"
    executable = tmp_path / "async_brain_check"
    subprocess.run([compiler, "-O2", "-std=c++17", "-pthread", "-Wall", "-Wextra",
                    str(source), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True, timeout=10)

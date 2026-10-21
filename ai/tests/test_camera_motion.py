"""Exercise the real local chase-camera response without an OpenGL context."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_native_camera_response_timing_and_reversals(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("no C++ compiler")
    binary = tmp_path / "camera-motion"
    subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    str(ROOT / "ai/tests/cpp/camera_motion_check.cpp"), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)

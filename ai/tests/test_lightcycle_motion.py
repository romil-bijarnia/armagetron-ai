"""Exercise the real model-pose helper without a renderer or simulation."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_native_lightcycle_pose_timing_and_wheel_phase(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("no C++ compiler")
    binary = tmp_path / "motion"
    subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    str(ROOT / "ai/tests/cpp/lightcycle_motion_check.cpp"), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)

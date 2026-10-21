"""Check contact staging, deterministic sampling and incoming momentum."""
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]

def test_native_impact_animation(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("no C++ compiler")
    binary = tmp_path / "impact-animation"
    subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    str(ROOT / "ai/tests/cpp/impact_animation_check.cpp"), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)

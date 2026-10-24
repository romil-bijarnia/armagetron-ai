"""Run the real ribbon GLSL through macOS OpenGL, including layered/mirror views."""
import pathlib
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != "darwin", reason="Uses the native macOS OpenGL renderer")
def test_lightwall_material(tmp_path):
    root = pathlib.Path(__file__).resolve().parents[2]
    source = pathlib.Path(__file__).parent / "cpp/lightwall_material_check.cpp"
    binary = tmp_path / "lightwall_material"
    subprocess.run(["clang++", "-std=c++17", "-O2", "-framework", "OpenGL",
                    str(source), "-o", str(binary)], check=True)
    result = subprocess.run([str(binary), str(root / "src/tron/gLightwallMaterial.h"),
                             str(root / "src/tron/gLightColour.h")])
    if result.returncode == 77:
        pytest.skip("No accelerated OpenGL context on this host")
    result.check_returncode()

"""Compile the real emitter geometry and exercise rapid turns and corner joins."""
import pathlib
import subprocess


def test_lightwall_attachment(tmp_path):
    source = pathlib.Path(__file__).parent / "cpp/lightwall_attachment_check.cpp"
    binary = tmp_path / "lightwall_attachment"
    subprocess.run(["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)

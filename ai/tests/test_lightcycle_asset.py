"""Validate the real exported asset and reject malformed binary payloads."""
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]

def test_authored_lightcycle_asset_and_bad_inputs(tmp_path):
    compiler = shutil.which('clang++') or shutil.which('g++')
    if compiler is None:
        pytest.skip('no C++ compiler')
    binary = tmp_path / 'asset-check'
    subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                    str(ROOT / 'ai/tests/cpp/lightcycle_asset_check.cpp'), '-o', str(binary)], check=True)
    subprocess.run([str(binary), str(ROOT / 'models/lightcycle-authored.bin')], check=True)

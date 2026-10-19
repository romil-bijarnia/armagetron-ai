#!/usr/bin/env python3
"""Capture controlled own-rider colours with the actual installed game, headless.

All camera/player overrides and server data live in the output directory. The
installed preset is frozen before the first capture, so later source edits cannot
change the baseline. Screenshots are actual native frames; the montage only
resizes them and adds labels. No game, training, or production config is edited.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = Path(__file__).with_name("capture_preview.py")
INSTALLED = Path("/Applications/Armagetron Advanced.app/Contents/MacOS/armagetronad")
COLOURS = {
    "red": (15, 0, 0), "orange": (15, 5, 0), "gold": (15, 11, 0),
    "green": (0, 15, 0), "cyan": (0, 15, 15), "blue": (0, 0, 15),
    "violet": (10, 0, 15), "white": (15, 15, 15),
}
QA_CAMERA = {
    "SPECTATOR_MODE_1": "0", "START_CAM_1": "3", "START_FOV_1": "65",
    "CAMERA_FOLLOW_START_X": "4", "CAMERA_FOLLOW_START_Y": "-6",
    "CAMERA_FOLLOW_START_Z": "2.5", "MAX_FPS": "60", "SHOW_FPS": "0", "TEXT_OUT": "0",
}
QA_SERVER = {"PLAY_WITHOUT_HUMANS": "0", "SIZE_FACTOR": "2", "SP_SIZE_FACTOR": "2"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture_one(job_path):
    job = json.loads(Path(job_path).read_text())
    specification = importlib.util.spec_from_file_location("lightcycle_colour_native_capture", CAPTURE)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    # The helper resolves its real engine imports first. Redirect only its preset
    # root to the frozen snapshot; explicit --client keeps installed bundle assets.
    module.ROOT = Path(job["baseline_root"])
    sys.argv = [str(CAPTURE)] + job["capture_arguments"]
    module.main()


def montage(output, runs):
    from PIL import Image, ImageDraw, ImageFont
    tile_width, tile_height, label_height = 640, 400, 38
    columns = min(4, len(runs))
    rows = (len(runs) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_width, rows * (tile_height + label_height)), (12, 15, 20))
    font_path = Path("/System/Library/Fonts/Menlo.ttc")
    font = ImageFont.truetype(str(font_path), 19) if font_path.is_file() else ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    for index, run in enumerate(runs):
        frame_path = Path(run["selected_frame"])
        with Image.open(frame_path) as source:
            image = source.convert("RGB")
            image.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        x, y = (index % columns) * tile_width, (index // columns) * (tile_height + label_height)
        canvas.paste(image, (x + (tile_width - image.width) // 2, y + label_height))
        draw.text((x + 12, y + 9), f"{run['colour']}  RGB {tuple(run['rgb_0_15'])}", font=font, fill=(225, 230, 237))
    path = output / "colour-montage.png"
    canvas.save(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--client", type=Path, default=INSTALLED)
    parser.add_argument("--capture-python", type=Path, default=ROOT / "ai/.venv/bin/python")
    parser.add_argument("--colours", default=",".join(COLOURS))
    parser.add_argument("--resume", action="store_true", help="reuse matching complete runs in this baseline")
    parser.add_argument("--port", type=int, default=49680)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=500)
    parser.add_argument("--server-set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--capture-job", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.capture_job:
        capture_one(args.capture_job)
        return
    if args.output is None:
        parser.error("--output is required")
    colours = args.colours.split(",")
    if not colours or any(colour not in COLOURS for colour in colours):
        parser.error("--colours must select names from " + ",".join(COLOURS))
    if not 9 < args.seconds <= 15:
        parser.error("--seconds must be above 9 and at most 15 for the 6s/9s captures")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    client = args.client.expanduser().resolve()
    preset = client.parent.parent / "etc/armagetronad/blacklight-visuals.cfg"
    baseline_root = output / "baseline-root"
    snapshot = baseline_root / "config/blacklight-visuals.cfg"
    baseline_path = output / "baseline.json"
    server_overrides = [f"{key}={value}" for key, value in QA_SERVER.items()] + args.server_set
    installed_hash, preset_hash = sha(client), sha(preset)
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        if baseline["client_sha256"] != installed_hash or baseline["installed_preset_sha256"] != preset_hash:
            raise RuntimeError("Installed baseline changed; choose a new output directory")
        if sha(snapshot) != preset_hash:
            raise RuntimeError("The frozen baseline preset was changed")
    else:
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(preset.read_bytes())
        baseline = {
            "client": str(client), "client_sha256": installed_hash,
            "installed_preset": str(preset), "installed_preset_sha256": preset_hash,
            "preset_snapshot": str(snapshot), "capture_helper_sha256": sha(CAPTURE),
            "qa_camera_settings": QA_CAMERA, "camera_scope": "isolated capture profile only",
            "gameplay_scope": "isolated dedicated arena, starts on human join with room for a straight colour pass",
        }
    baseline["qa_camera_settings"] = QA_CAMERA
    baseline["qa_server_settings"] = QA_SERVER
    baseline_path.write_text(json.dumps(baseline, indent=2) + "\n")
    runs = []
    for index, colour in enumerate(colours):
        if sha(client) != installed_hash or sha(preset) != preset_hash:
            raise RuntimeError("Installed baseline changed during the gamut capture")
        rgb = COLOURS[colour]
        directory = output / colour
        directory.mkdir(parents=True, exist_ok=True)
        settings = dict(QA_CAMERA)
        settings.update(PLAYER_1=f"QA-{colour}", COLOR_R_1=str(rgb[0]), COLOR_G_1=str(rgb[1]), COLOR_B_1=str(rgb[2]))
        arguments = ["--client", str(client), "--output", str(directory),
                     "--width", str(args.width), "--height", str(args.height),
                     "--seconds", str(args.seconds), "--capture-at", "6", "--capture-at", "9",
                     "--port", str(args.port + index)]
        for key, value in settings.items():
            arguments.extend(["--set", f"{key}={value}"])
        for setting in server_overrides:
            arguments.extend(["--server-set", setting])
        job = {"baseline_root": str(baseline_root), "capture_arguments": arguments,
               "client_settings": settings, "server_overrides": server_overrides,
               "colour": colour, "rgb_0_15": rgb, "client_sha256": installed_hash,
               "installed_preset_sha256": preset_hash}
        job_path = directory / "colour-job.json"
        reusable = False
        if args.resume and job_path.exists() and (directory / "capture.json").exists():
            previous = json.loads(job_path.read_text())
            reusable = (previous["client_settings"] == settings
                        and previous["server_overrides"] == server_overrides
                        and previous["client_sha256"] == installed_hash
                        and previous["installed_preset_sha256"] == preset_hash
                        and (directory / "frame-06.png").is_file()
                        and (directory / "frame-09.png").is_file())
        if not reusable:
            job_path.write_text(json.dumps(job, indent=2) + "\n")
            with (directory / "harness.log").open("w") as log:
                subprocess.run([str(args.capture_python), str(Path(__file__).resolve()), "--capture-job", str(job_path)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        capture = json.loads((directory / "capture.json").read_text())
        run = {"colour": colour, "rgb_0_15": rgb, "client_settings": settings,
               "server_overrides": server_overrides, "job": str(job_path), "capture": capture,
               "selected_frame": str(directory / "frame-09.png"), "reused": reusable}
        runs.append(run)
        (output / "colour-runs.json").write_text(json.dumps(runs, indent=2) + "\n")
        print(f"{colour}: actual hidden frames {capture['captures']} captured", flush=True)
    if sha(client) != installed_hash or sha(preset) != preset_hash:
        raise RuntimeError("Installed baseline changed before the comparison completed")
    image = montage(output, runs)
    report = {"baseline": baseline, "runs": runs, "montage": str(image),
              "all_game_processes_finished": True, "production_files_modified": False}
    (output / "colour-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"montage": str(image), "report": str(output / 'colour-report.json'),
                      "colours": colours, "all_game_processes_finished": True}))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
        print(f"Native colour check: {error}", file=sys.stderr)
        raise SystemExit(1)

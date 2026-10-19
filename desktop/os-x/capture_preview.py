#!/usr/bin/env python3
"""Capture the real renderer in a hidden spectator client and an isolated arena.

Run with ai/.venv/bin/python after building the native client. This never joins
the human match, opens a window, or loads/saves a training checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import select
import shutil
import struct
import subprocess
import sys
import threading
import time
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ai"))
from arma_ai.engine import ArenaConfig, ENGINE_BIN, ENGINE_CONFIG, ENGINE_DATA


def png(path, pixels, width, height):
    def chunk(tag, data):
        return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", zlib.crc32(tag + data))
    stride = width * 3
    scanlines = b"".join(b"\0" + pixels[y * stride:(y + 1) * stride] for y in range(height))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b""))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    built_client = ROOT / "build-macos/Armagetron Advanced.app/Contents/MacOS/armagetronad"
    installed_client = Path("/Applications/Armagetron Advanced.app/Contents/MacOS/armagetronad")
    ap.add_argument("--client", type=Path, default=built_client if built_client.is_file() else installed_client)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--classic", action="store_true")
    ap.add_argument("--load-only", action="store_true", help="render without GPU readback, for concurrent latency checks")
    ap.add_argument("--width", type=int, default=768, help="logical window width; Retina drawable is twice this")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--font", type=Path, help="optional installed font to verify with the local preset")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override one local client setting for a controlled comparison")
    ap.add_argument("--server-set", action="append", default=[], metavar="KEY=VALUE",
                    help="override one isolated preview-server setting")
    ap.add_argument("--capture-at", type=float, action="append", default=None,
                    help="capture at this elapsed second; repeat to choose multiple frames")
    ap.add_argument("--video", type=Path, help="also record the actual rendered frames to an MP4 (use MAX_FPS=60)")
    ap.add_argument("--video-after", type=float, default=4, help="start recording after this elapsed second")
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--neural-slots", type=int, default=0,
                    help="use the native recurrent policy for this many of the four server bots")
    ap.add_argument("--neural-model", type=Path,
                    help="explicit native policy file for a headless inference/frame-time check")
    ap.add_argument("--port", type=int, default=49534)
    args = ap.parse_args()
    if not 1 <= args.seconds <= 60:
        ap.error("--seconds must be between 1 and 60")
    if not 320 <= args.width <= 1920 or not 240 <= args.height <= 1200:
        ap.error("preview dimensions are out of range")
    if not 0 <= args.neural_slots <= 4:
        ap.error("--neural-slots must be between zero and four")
    if args.neural_model and not args.neural_model.is_file():
        ap.error("--neural-model must be an existing policy file")
    encoder = shutil.which("ffmpeg") if args.video else None
    if args.video and (args.load_only or not encoder or not 0 <= args.video_after < args.seconds):
        ap.error("video requires ffmpeg, frame readback, and --video-after within the capture duration")
    marks = args.capture_at if args.capture_at is not None else [4, 8, 12]
    if args.capture_at is not None and any(not 0 < mark < args.seconds for mark in marks):
        ap.error("capture times must be positive and below --seconds")

    def settings(values):
        for setting in values:
            key, separator, value = setting.partition("=")
            if not separator or not re.fullmatch(r"[A-Z][A-Z_0-9]*", key) or '\n' in value or '\r' in value:
                ap.error("settings must be one KEY=VALUE each")
            yield key, value
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for side in ("server", "client"):
        for sub in ("config", "var", "data"):
            (root / side / sub).mkdir(parents=True, exist_ok=True)
    server_settings = ArenaConfig(slots=args.neural_slots, builtin_ais=4 - args.neural_slots,
                                  lockstep_dt=0).settings()
    server_settings.update(SERVER_PORT=str(args.port), NEURAL_END_ROUND_WITHOUT_NEURAL="0")
    server_settings.update(dict(settings(args.server_set)))
    if args.neural_slots:
        server_settings["NEURAL_DEBUG"] = "1"
        if args.neural_model:
            # Model paths use the engine's data hierarchy; absolute paths are rejected.
            model = root / "server/data/brain/policy.bin"
            model.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.neural_model, model)
            server_settings["NEURAL_MODEL"] = "brain/policy.bin"
    (root / "server/config/autoexec.cfg").write_text("".join(f"{key} {value}\n" for key, value in server_settings.items()))
    config = """FIRST_USE 0
FULLSCREEN 0
ARMAGETRON_WINDOWSIZE -1
ARMAGETRON_WINDOWSIZE_W 768
ARMAGETRON_WINDOWSIZE_H 480
LOW_DPI_WINDOW 0
ARMAGETRON_VSYNC 2
KEEP_WINDOW_ACTIVE 1
SPECTATOR_MODE_1 1
PLAYER_1 Preview
MOUSE_GRAB 0
SOUND_QUALITY 0
MUSIC_VOLUME 0
SHOW_FPS 1
TEXT_OUT 1
"""
    if args.classic:
        config += "BLACKLIGHT_VISUALS 0\nANTIALIAS_SAMPLES 0\nTEXTURE_ANISOTROPY 1\nFLOOR_DETAIL 3\nFLOOR_MIRROR 0\n"
    else:
        config += (ROOT / "config/blacklight-visuals.cfg").read_text()
    config += f"\nARMAGETRON_WINDOWSIZE_W {args.width}\nARMAGETRON_WINDOWSIZE_H {args.height}\n"
    if args.font:
        if not args.font.is_file():
            ap.error("font file does not exist")
        config += f"FONT_TYPE 3\nFONT_FILE_CUSTOM {args.font.resolve()}\nUSE_CUSTOM_FONT 1\n"
    for key, value in settings(args.set):
        config += f"{key} {value}\n"
    (root / "client/config/autoexec.cfg").write_text(config)
    client_path = args.client.resolve()
    if client_path.parent.name == "MacOS" and client_path.parent.parent.name == "Contents":
        bundle = client_path.parent.parent
    else:
        # A bare incremental-build executable uses the installed game assets.
        bundle = installed_client.parent.parent
    resources = bundle / "Resources/armagetronad"
    configdir = bundle / "etc/armagetronad"
    processes = []
    dimensions = []
    started = time.monotonic()
    frames = 0
    recording = None
    recording_log = None
    recorded_frames = 0
    try:
        with (root / "server.log").open("w") as log:
            server = subprocess.Popen([str(ENGINE_BIN), "--daemon", "--datadir", str(ENGINE_DATA),
                                       "--configdir", str(ENGINE_CONFIG), "--userconfigdir", str(root / "server/config"),
                                       "--vardir", str(root / "server/var"), "--userdatadir", str(root / "server/data")],
                                      stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        processes.append(server)
        time.sleep(.8)
        command = [str(args.client.resolve()), "--hidden-window", "--videoout", "--datadir", str(resources),
                   "--configdir", str(configdir), "--userconfigdir", str(root / "client/config"),
                   "--vardir", str(root / "client/var"), "--userdatadir", str(root / "client/data"),
                   "--connect", f"127.0.0.1:{args.port}"]
        if args.load_only:
            command.remove("--videoout")
        client = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                  stdout=subprocess.DEVNULL if args.load_only else subprocess.PIPE,
                                  stderr=subprocess.PIPE)
        processes.append(client)
        (root / "processes.json").write_text(json.dumps({"client": client.pid, "server": server.pid}))

        def stderr():
            with (root / "client.log").open("wb") as log:
                for line in iter(client.stderr.readline, b""):
                    log.write(line)
                    log.flush()
                    match = re.search(rb"drawable[ =:]*(\d+)\s*[xX]\s*(\d+)", line)
                    if match:
                        dimensions[:] = [int(match[1]), int(match[2])]
        reader = threading.Thread(target=stderr, daemon=True)
        reader.start()
        if args.load_only:
            while time.monotonic() - started < args.seconds:
                if client.poll() is not None:
                    raise RuntimeError(f"Preview client exited {client.returncode}")
                time.sleep(.2)
            if args.neural_slots and "loaded (training update " not in (root / "server.log").read_text():
                raise RuntimeError("Native policy did not load; built-in fallback is not a valid memory check")
            print(json.dumps({"hidden": True, "load_only": True, "dimensions": dimensions,
                              "seconds": time.monotonic() - started}))
            return
        buffer = bytearray()
        captured = set()
        while time.monotonic() - started < args.seconds:
            if client.poll() is not None:
                raise RuntimeError(f"Preview client exited {client.returncode}; see {root / 'client.log'}")
            ready, _, _ = select.select([client.stdout], [], [], .2)
            if not ready:
                continue
            chunk = os.read(client.stdout.fileno(), 1024 * 1024)
            if not chunk:
                break
            buffer.extend(chunk)
            if not dimensions:
                continue
            width, height = dimensions
            length = width * height * 3
            while len(buffer) >= length:
                frame = bytes(buffer[:length])
                del buffer[:length]
                frames += 1
                elapsed = time.monotonic() - started
                if args.video and elapsed >= args.video_after:
                    if recording is None:
                        video = args.video.resolve()
                        video.parent.mkdir(parents=True, exist_ok=True)
                        recording_log = (root / "video-encoder.log").open("wb")
                        recording = subprocess.Popen([
                            encoder, "-hide_banner", "-loglevel", "error", "-y",
                            "-f", "rawvideo", "-pixel_format", "rgb24",
                            "-video_size", f"{width}x{height}", "-framerate", "60", "-i", "pipe:0",
                            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)
                        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=recording_log)
                    recording.stdin.write(frame)
                    recorded_frames += 1
                for mark in marks:
                    if elapsed >= mark and mark not in captured:
                        png(root / f"frame-{mark:02g}.png", frame, width, height)
                        captured.add(mark)
        if not captured:
            raise RuntimeError(f"No rendered frames captured; see {root / 'client.log'}")
        report = {"frames": frames, "dimensions": dimensions, "elapsed_seconds": time.monotonic() - started,
                  "classic": args.classic, "hidden": True, "captures": sorted(captured)}
        if recording is not None:
            recording.stdin.close()
            if recording.wait(timeout=15) != 0:
                raise RuntimeError(f"Video encoder failed; see {root / 'video-encoder.log'}")
            report.update(video=str(args.video.resolve()), video_frames=recorded_frames, video_fps=60)
        (root / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
    finally:
        if recording is not None:
            if not recording.stdin.closed:
                recording.stdin.close()
            if recording.poll() is None:
                try:
                    recording.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    recording.kill()
                    recording.wait()
        if recording_log is not None:
            recording_log.close()
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()

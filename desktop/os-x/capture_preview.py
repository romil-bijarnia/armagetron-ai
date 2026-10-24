#!/usr/bin/env python3
"""Capture the real renderer in a hidden client and an isolated arena.

Run with ai/.venv/bin/python after building the native client. This never joins
the human match, opens a window, or loads/saves a training checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ai"))
from arma_ai.engine import ArenaConfig, EnginePool, ENGINE_BIN, ENGINE_CONFIG, ENGINE_DATA


def png(path, pixels, width, height):
    def chunk(tag, data):
        return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", zlib.crc32(tag + data))
    stride = width * 3
    if len(pixels) != stride * height:
        raise ValueError("RGB frame size does not match its dimensions")
    # Compress one row at a time, avoiding another entire uncompressed 8K copy.
    compressor = zlib.compressobj()
    view = memoryview(pixels)
    with path.open("wb") as output:
        output.write(b"\x89PNG\r\n\x1a\n")
        output.write(chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        for y in range(height):
            data = compressor.compress(b"\0") + compressor.compress(view[y * stride:(y + 1) * stride])
            if data:
                output.write(chunk(b"IDAT", data))
        output.write(chunk(b"IDAT", compressor.flush()))
        output.write(chunk(b"IEND", b""))


def pixel_dimensions(value):
    value = {"4k": "3840x2160", "8k": "7680x4320"}.get(value.lower(), value)
    match = re.fullmatch(r"(\d+)[xX](\d+)", value)
    if not match:
        raise argparse.ArgumentTypeError("use WIDTHxHEIGHT, 4k, or 8k")
    width, height = map(int, match.groups())
    if not 320 <= width <= 7680 or not 240 <= height <= 4320:
        raise argparse.ArgumentTypeError("pixel dimensions must be between 320x240 and 7680x4320")
    return width, height


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    installed_client = Path("/Applications/Armagetron Advanced.app/Contents/MacOS/armagetronad")
    ap.add_argument("--client", type=Path, default=installed_client)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--classic", action="store_true")
    ap.add_argument("--load-only", action="store_true", help="render without GPU readback, for concurrent latency checks")
    ap.add_argument("--width", type=int, default=768, help="logical window width; actual Retina scale is reported")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--pixels", type=pixel_dimensions, metavar="WIDTHxHEIGHT",
                    help="exact native framebuffer size (also accepts 4k or 8k); fails on mismatch, never upscales")
    ap.add_argument("--readback-interval", type=float, default=None, metavar="SECONDS",
                    help="minimum delay after GPU readback; defaults to 1s for --pixels stills, otherwise every frame")
    ap.add_argument("--font", type=Path, help="optional installed font to verify with the local preset")
    ap.add_argument("--profile-config", type=Path,
                    help="copy an existing saved user config into the isolated client before applying capture overrides")
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
    ap.add_argument("--turn-demo", action="store_true",
                    help="drive one isolated bot through scripted corners and rapid reversals; no policy is loaded")
    ap.add_argument("--local-turn-demo", action="store_true",
                    help="send rapid SDL key taps through the hidden client's normal player-input path")
    ap.add_argument("--port", type=int, default=49534)
    args = ap.parse_args()
    if not 1 <= args.seconds <= 600:
        ap.error("--seconds must be between 1 and 600")
    if not 320 <= args.width <= 3840 or not 240 <= args.height <= 2160:
        ap.error("preview dimensions are out of range")
    interval = args.readback_interval
    if interval is None:
        interval = 1.0 if args.pixels and not args.video and not args.load_only else 0.0
    if not math.isfinite(interval) or not 0 <= interval < args.seconds:
        ap.error("--readback-interval must be finite, nonnegative, and below --seconds")
    if args.video and interval:
        ap.error("video requires --readback-interval 0 to record every rendered frame")
    if not 0 <= args.neural_slots <= 4:
        ap.error("--neural-slots must be between zero and four")
    if args.turn_demo and (args.neural_slots or args.neural_model):
        ap.error("--turn-demo uses scripted actions and cannot load a neural policy")
    if args.local_turn_demo and (args.turn_demo or args.neural_slots or args.neural_model):
        ap.error("--local-turn-demo requires an ordinary human player without a neural policy")
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
    slots = 1 if args.turn_demo else args.neural_slots
    server_settings = ArenaConfig(slots=slots, builtin_ais=4 - slots,
                                  lockstep_dt=0).settings()
    server_settings.update(SERVER_PORT=str(args.port), NEURAL_END_ROUND_WITHOUT_NEURAL="0")
    server_settings.update(dict(settings(args.server_set)))
    if args.local_turn_demo:
        server_settings["PLAY_WITHOUT_HUMANS"] = "0"
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
    if args.local_turn_demo:
        config += ("SPECTATOR_MODE_1 0\nCAPTURE_LOCAL_TURNS 1\nCAPTURE_TRAIL_DIAGNOSTICS 1\n"
                   "KEYBOARD SCANCODE_LEFT PLAYER_BIND CYCLE_TURN_LEFT 1\n"
                   "KEYBOARD SCANCODE_RIGHT PLAYER_BIND CYCLE_TURN_RIGHT 1\n")
    if args.pixels:
        # In this isolated hidden client one point is one pixel. The measured
        # drawable still must match, since drivers can reject oversized windows.
        config += (f"LOW_DPI_WINDOW 1\nARMAGETRON_WINDOWSIZE -1\n"
                   f"ARMAGETRON_WINDOWSIZE_W {args.pixels[0]}\nARMAGETRON_WINDOWSIZE_H {args.pixels[1]}\n")
    profile_snapshot = None
    if args.profile_config:
        if not args.profile_config.is_file():
            ap.error("--profile-config must be an existing saved user configuration")
        profile_bytes = args.profile_config.read_bytes()
        (root / "client/config/user_3_1_utf8.cfg").write_bytes(profile_bytes)
        profile_snapshot = {"source": str(args.profile_config.resolve()),
                            "sha256": hashlib.sha256(profile_bytes).hexdigest(),
                            "scope": "Copied into the isolated test profile; capture overrides follow."}
    (root / "client/config/autoexec.cfg").write_text(config)
    client_path = args.client.resolve()
    if client_path.parent.name == "MacOS" and client_path.parent.parent.name == "Contents":
        bundle = client_path.parent.parent
    else:
        # Incremental executables share assets with the one installed app.
        # Previewing never creates another discoverable application bundle.
        bundle = installed_client.parent.parent
    resources = bundle / "Resources/armagetronad"
    configdir = bundle / "etc/armagetronad"
    processes = []
    dimensions = []
    gl_limits = {}
    started = time.monotonic()
    frames = 0
    recording = None
    recording_log = None
    recorded_frames = 0
    scenario_pool = None
    scenario_stop = threading.Event()
    scenario_thread = None
    scenario_events = []
    scenario_errors = []
    turn_schedule = [(2.0, 1), (3.2, 2),
                     (4.4, 1), (4.5, 2), (4.6, 1), (4.7, 2),
                     (5.8, 2), (5.9, 1), (6.0, 2), (6.1, 1),
                     (8.0, 2), (8.1, 2), (9.0, 1)]

    def scripted_turns():
        # Single corners, alternating taps, then a same-direction pair with
        # 100 ms spacing. All times are actual server game time.
        previous_round, index = None, 0
        try:
            while not scenario_stop.is_set():
                try:
                    steps = scenario_pool.poll(timeout=2)
                except TimeoutError:
                    continue  # countdown and round transitions have no action steps
                for step in steps:
                    if step.round_id != previous_round:
                        previous_round, index = step.round_id, 0
                    action = 0
                    if (index < len(turn_schedule) and step.round_time >= turn_schedule[index][0]
                            and step.slots[0].needs_action
                            and step.slots[0].mask & (1 << turn_schedule[index][1])):
                        action = turn_schedule[index][1]
                        scenario_events.append({"round": step.round_id, "time": step.round_time, "action": action,
                                                "mask": step.slots[0].mask})
                        index += 1
                    scenario_pool.act(step.engine, [action])
        except Exception as error:
            if not scenario_stop.is_set():
                scenario_errors.append(str(error))

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.turn_demo:
            scenario_pool = EnginePool([ArenaConfig(slots=1, builtin_ais=3, lockstep_dt=0,
                                      extra=server_settings)], workdir=root / "scenario", base_port=args.port,
                                      log_engines=True)
            server = scenario_pool.procs[0]
            scenario_thread = threading.Thread(target=scripted_turns, daemon=True)
            scenario_thread.start()
        else:
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
        elif interval:
            command.extend(["--videoout-interval", str(interval)])
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
                    limits = re.search(rb"OpenGL limits: texture=(\d+) renderbuffer=(\d+) viewport=(\d+)x(\d+)(?: samples=(\d+))?", line)
                    if limits:
                        gl_limits.update(texture=int(limits[1]), renderbuffer=int(limits[2]),
                                         viewport=[int(limits[3]), int(limits[4])])
                        if limits[5]:
                            gl_limits["samples"] = int(limits[5])
        reader = threading.Thread(target=stderr, daemon=True)
        reader.start()

        def check_dimensions():
            if args.pixels and dimensions and tuple(dimensions) != args.pixels:
                raise RuntimeError(f"Requested native {args.pixels[0]}x{args.pixels[1]}, "
                                   f"but drawable is {dimensions[0]}x{dimensions[1]}; no image was upscaled")

        if args.load_only:
            while time.monotonic() - started < args.seconds:
                if client.poll() is not None:
                    raise RuntimeError(f"Preview client exited {client.returncode}")
                if scenario_errors:
                    raise RuntimeError(f"Turn scenario failed: {scenario_errors[0]}")
                check_dimensions()
                time.sleep(.2)
            if args.neural_slots and "loaded (training update " not in (root / "server.log").read_text():
                raise RuntimeError("Native policy did not load; built-in fallback is not a valid memory check")
            check_dimensions()
            if not dimensions:
                raise RuntimeError(f"No drawable dimensions reported; see {root / 'client.log'}")
            report = {"hidden": True, "load_only": True, "dimensions": dimensions,
                      "framebuffer_dimensions": dimensions,
                      "requested_pixels": args.pixels, "gl_limits": gl_limits,
                      "seconds": time.monotonic() - started}
            if profile_snapshot:
                report["profile_snapshot"] = profile_snapshot
            (root / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report))
            return
        buffer = bytearray()
        captured = set()
        capture_times = {}
        while time.monotonic() - started < args.seconds:
            if client.poll() is not None:
                raise RuntimeError(f"Preview client exited {client.returncode}; see {root / 'client.log'}")
            if scenario_errors:
                raise RuntimeError(f"Turn scenario failed: {scenario_errors[0]}")
            ready, _, _ = select.select([client.stdout], [], [], .2)
            if not ready:
                continue
            chunk = os.read(client.stdout.fileno(), 1024 * 1024)
            if not chunk:
                break
            buffer.extend(chunk)
            if not dimensions:
                continue
            check_dimensions()
            width, height = dimensions
            length = width * height * 3
            while len(buffer) >= length:
                frame = bytes(memoryview(buffer)[:length])
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
                            "-vf", "scale=out_color_matrix=bt709:out_range=tv",
                            "-pix_fmt", "yuv420p", "-colorspace", "bt709",
                            "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
                            "-movflags", "+faststart", str(video)
                        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=recording_log)
                    recording.stdin.write(frame)
                    recorded_frames += 1
                for mark in marks:
                    if elapsed >= mark and mark not in captured:
                        png(root / f"frame-{mark:02g}.png", frame, width, height)
                        captured.add(mark)
                        capture_times[mark] = elapsed
        if not captured:
            raise RuntimeError(f"No rendered frames captured; see {root / 'client.log'}")
        local_turns = None
        if args.local_turn_demo:
            log = (root / "client.log").read_text()
            keys = re.findall(r"Capture local key: index=(\d+) ", log)
            accepted = [{"time": float(t), "direction": int(d), "turns": int(n)}
                        for t, d, n in re.findall(
                            r"Capture accepted turn: time=([\d.e+-]+) direction=(-?\d+) turns=(\d+)", log)]
            if "67" not in keys:
                raise RuntimeError("Local turn demo did not complete its 68 input events; see client.log")
            if len(accepted) < 26 or not any(event["time"] >= 6.5 for event in accepted):
                raise RuntimeError("Local turn demo did not exercise both rapid-turn bursts; see client.log")
            local_turns = {"input_events": len(keys), "accepted_turns": len(accepted), "events": accepted,
                           "note": "Input events are not accepted turns; cooldown and death can reject keys."}
            (root / "local-turns.json").write_text(json.dumps(local_turns, indent=2) + "\n")
        if args.turn_demo and not any(event["time"] >= 8.1 for event in scenario_events):
            raise RuntimeError("Turn demo ended before completing rapid turns; see turn-scenario.json")
        report = {"frames": frames, "dimensions": dimensions, "elapsed_seconds": time.monotonic() - started,
                  "classic": args.classic, "hidden": True, "captures": sorted(captured),
                  "capture_times": capture_times, "requested_pixels": args.pixels,
                  "framebuffer_dimensions": dimensions, "output_dimensions": dimensions,
                  "pixel_source": "native_opengl_framebuffer", "pixel_format": "rgb8_sdr",
                  "readback_interval_seconds": interval, "gl_limits": gl_limits}
        if local_turns is not None:
            report["local_turns"] = {key: local_turns[key] for key in ("input_events", "accepted_turns", "note")}
        if recording is not None:
            recording.stdin.close()
            if recording.wait(timeout=15) != 0:
                raise RuntimeError(f"Video encoder failed; see {root / 'video-encoder.log'}")
            report.update(video=str(args.video.resolve()), video_frames=recorded_frames, video_fps=60)
        if profile_snapshot:
            report["profile_snapshot"] = profile_snapshot
        (root / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
    finally:
        scenario_stop.set()
        if scenario_pool is not None:
            if scenario_thread is not None:
                scenario_thread.join(timeout=2.5)
            scenario_pool.close()
            (root / "turn-scenario.json").write_text(json.dumps({"events": scenario_events,
                                                                "errors": scenario_errors}, indent=2) + "\n")
        if recording is not None:
            if not recording.stdin.closed:
                try:
                    recording.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
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
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Preview capture interrupted; isolated processes stopped.", file=sys.stderr)
        raise SystemExit(130)

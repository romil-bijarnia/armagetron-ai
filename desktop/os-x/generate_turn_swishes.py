#!/usr/bin/env python3
"""Synthesize original, deterministic lightcycle turn swishes without playback.

Uses seeded broadband noise only: no recordings, oscillators, tonal attacks or
external samples. Output is mono PCM16 at 44.1/48 kHz, with rounded attacks and
exact silent tails. Project license: GNU GPL version 2 or later.

The default -4 dBFS file peak becomes approximately -18.54 dBFS after the existing
60/128 chunk level and a .4 camera gain, before any further spatial panning.
Requires NumPy; ffmpeg is used only to analyze the original OGG for comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = (
    ("a", 14117, .182, .010, 470, 5100, 290, 2600),
    ("b", 14118, .193, .011, 430, 4800, 320, 2400),
    ("c", 14119, .176, .009, 510, 5400, 310, 2800),
)


def smoothstep(value):
    value = np.clip(value, 0, 1)
    return value * value * value * (value * (value * 6 - 15) + 10)


def coloured_noise(rng, count, rate, low, high):
    # Work in a longer, non-looping block and use its middle. Wide, smooth bands
    # avoid ringing resonances; a mild pink tilt removes the brittle hiss edge.
    padded = 1 << (count * 4 - 1).bit_length()
    white = rng.standard_normal(padded)
    frequency = np.fft.rfftfreq(padded, 1 / rate)
    high_pass = (frequency / low) ** 4
    high_pass /= 1 + high_pass
    low_pass = 1 / np.sqrt(1 + (frequency / high) ** 8)
    tilt = (np.maximum(frequency, 50) / 1000) ** -.28
    shaping = high_pass * low_pass * tilt
    shaping[0] = 0
    shaped = np.fft.irfft(np.fft.rfft(white) * shaping, padded)
    start = (padded - count) // 2
    shaped = shaped[start:start + count].copy()
    return shaped / np.sqrt(np.mean(shaped * shaped))


def swish(seed, duration, attack, early_low, early_high, late_low, late_high, rate, peak_dbfs):
    count = round(duration * rate)
    rng = np.random.Generator(np.random.PCG64(seed))
    t = np.arange(count, dtype=np.float64) / rate
    lead, silence_tail = .0035, .028
    active_end = duration - silence_tail
    progress = smoothstep((t - lead) / (active_end - lead))
    bright = coloured_noise(rng, count, rate, early_low, early_high)
    warm = coloured_noise(rng, count, rate, late_low, late_high)
    # A broad spectral downward breath gives energy motion, without a pitched
    # sweep. Independent noise layers prevent a repeating, mechanical texture.
    sound = np.cos(progress * math.pi / 2) * bright + np.sin(progress * math.pi / 2) * warm
    envelope = smoothstep((t - lead) / attack)
    envelope *= np.exp(-np.maximum(0, t - lead - attack) / .052)
    envelope *= 1 - smoothstep((t - (active_end - .038)) / .038)
    envelope[t >= active_end] = 0
    # Remove residual DC under the envelope rather than adding a DC correction
    # to the silent ends. Both value and slope finish at zero.
    sound -= np.sum(sound * envelope) / np.sum(envelope)
    sound *= envelope
    sound *= 10 ** (peak_dbfs / 20) / np.max(np.abs(sound))
    sound[t < lead] = 0
    sound[t >= active_end] = 0
    return sound


def write_wav(path, samples, rate):
    if not np.all(np.isfinite(samples)) or np.max(np.abs(samples)) >= 1:
        raise ValueError(f"Nonfinite or clipped samples for {path}")
    pcm = np.rint(samples * 32767).astype("<i2")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with wave.open(str(temporary), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm.tobytes())
    temporary.replace(path)
    return pcm.astype(np.float64) / 32768


def decode(path, rate):
    output = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path),
                                      "-f", "f32le", "-ac", "1", "-ar", str(rate), "pipe:1"])
    return np.frombuffer(output, dtype="<f4").astype(np.float64)


def db(value):
    return 20 * math.log10(max(float(value), 1.e-15))


def statistics(path, samples, rate):
    peak = np.max(np.abs(samples))
    energy = np.cumsum(samples * samples)
    nonzero = np.flatnonzero(samples != 0)
    frequency = np.fft.rfftfreq(len(samples), 1 / rate)
    power = np.abs(np.fft.rfft(samples * np.hanning(len(samples)))) ** 2
    audible = frequency >= 50
    audible_power = power[audible]
    return {
        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sample_rate": rate, "channels": 1, "duration_ms": len(samples) / rate * 1000,
        "peak_dbfs": db(peak), "rms_dbfs": db(np.sqrt(np.mean(samples * samples))),
        "first_5ms_peak_dbfs": db(np.max(np.abs(samples[:round(rate * .005)]))),
        "energy_10pct_ms": np.searchsorted(energy, energy[-1] * .1) / rate * 1000,
        "energy_90pct_ms": np.searchsorted(energy, energy[-1] * .9) / rate * 1000,
        "spectral_centroid_above_50hz": float(np.sum(frequency[audible] * audible_power) / np.sum(audible_power)),
        "leading_exact_silence_ms": int(nonzero[0]) / rate * 1000 if len(nonzero) else len(samples) / rate * 1000,
        "trailing_exact_silence_ms": (len(samples) - 1 - int(nonzero[-1])) / rate * 1000 if len(nonzero) else len(samples) / rate * 1000,
        "first_sample": float(samples[0]), "last_sample": float(samples[-1]),
        "dc_mean": float(np.mean(samples)), "max_adjacent_step": float(np.max(np.abs(np.diff(samples)))),
        "clipped_samples": int(np.count_nonzero(np.abs(samples) >= 1)),
    }


def sequence(variants, rate, spacing, count):
    start = round(rate * .20)
    length = start + round(rate * spacing * (count - 1)) + max(len(v) for v in variants) + round(rate * .30)
    result = np.zeros(length)
    for index in range(count):
        offset = start + round(rate * spacing * index)
        sound = variants[index % len(variants)]
        result[offset:offset + len(sound)] += sound
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, default=ROOT / "sound")
    parser.add_argument("--preview-dir", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, choices=(44100, 48000), default=44100)
    parser.add_argument("--peak-dbfs", type=float, default=-4)
    args = parser.parse_args()
    if not math.isfinite(args.peak_dbfs) or not -36 <= args.peak_dbfs <= -1:
        parser.error("--peak-dbfs must be between -36 and -1")
    args.asset_dir = args.asset_dir.resolve()
    args.preview_dir = args.preview_dir.resolve()
    args.asset_dir.mkdir(parents=True, exist_ok=True)
    args.preview_dir.mkdir(parents=True, exist_ok=True)
    rate = args.sample_rate
    rendered, reports = [], []
    original = ROOT / "sound/cycle_turn.ogg"
    original_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    for letter, seed, duration, attack, lo0, hi0, lo1, hi1 in VARIANTS:
        sound = swish(seed, duration, attack, lo0, hi0, lo1, hi1, rate, args.peak_dbfs)
        path = args.asset_dir / f"cycle_turn_air_{letter}.wav"
        actual = write_wav(path, sound, rate)
        report = statistics(path, actual, rate)
        report.update(seed=seed, rounded_attack_ms=attack * 1000,
                      estimated_peak_dbfs_at_chunk60_camera04=report["peak_dbfs"] + db(60 / 128 * .4))
        assert report["clipped_samples"] == 0 and report["trailing_exact_silence_ms"] >= 27
        assert report["first_sample"] == report["last_sample"] == 0
        rendered.append(actual)
        reports.append(report)
    preview_paths = []
    for name, spacing, count in (("turn-swishes-spaced.wav", .55, 6), ("turn-swishes-rapid.wav", .14, 12)):
        path = args.preview_dir / name
        write_wav(path, sequence(rendered, rate, spacing, count), rate)
        preview_paths.append(str(path))
    baseline = statistics(original, decode(original, rate), rate)
    assert hashlib.sha256(original.read_bytes()).hexdigest() == original_hash
    report = {
        "synthesis": "Original seeded broadband-noise spectral crossfade, rounded finite envelope; no sampled recordings or oscillators",
        "script": str(Path(__file__).resolve()), "numpy_version": np.__version__,
        "original_unchanged_sha256": original_hash, "original": baseline,
        "assets": reports, "preview_sequences": preview_paths,
        "packaging_entries": [f"sound/cycle_turn_air_{letter}.wav" for letter, *_ in VARIANTS],
        "runtime_integration": "PCM16 WAV loads through existing SDL_mixer Mix_LoadWAV, with no Vorbis padding or decoder requirement",
        "runtime_gain_assumption": "Peak estimate uses chunk60/128 × camera .4; spatial panning may reduce it further",
        "license": "Original procedural audio; GNU GPL version 2 or later, matching the project",
        "playback_performed": False,
    }
    (args.preview_dir / "turn-swish-analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

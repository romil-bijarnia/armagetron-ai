# Armagetron AI

Armagetron Advanced with a V4 combat policy trained through direct self-play reinforcement learning.

The engine's neural bridge (`src/tron/gNeural.cpp`) controls AI cycles through normal steering and braking commands. The `ai/` folder contains the single supported network (`PolicyNet` in `arma_ai/model.py`), the GPU learner, a C++ self-play simulator, evaluation tools, and a play server for racing the trained policy from the normal game client.

Training mixes ordinary rounds with varied close encounters. Simulator behavior is checked against recorded engine rounds; fixed-opponent evaluation uses the real dedicated engine. Exported V4 weights also run directly in the game's C++ inference code.

## Getting started

Build the dedicated server with the bridge, then train or play. The details, including the full configure line, are in [ai/README.md](ai/README.md).

```bash
cd ai && uv run arma-background
```

```bash
cd ai && uv run arma-play
```

With the play server running, open Armagetron, go to Play Game, Multiplayer, Custom Connect, and connect to `127.0.0.1` on port 4534.

## Native Mac app

This repository builds the game as a native app for the chip of the Mac it is built on (arm64 on Apple Silicon), with its runtime libraries inside the bundle:

```bash
desktop/os-x/build_native_app.sh
```

The script updates `/Applications/Armagetron Advanced.app`, the single installed
app. `build-macos/` keeps the `.dmg`, release `.zip`, and `previous-installed.zip`
rollback archive. Temporary bundles stay in an unindexed staging folder and are
removed after installation, so builds do not add another game to Spotlight.
The script lists the Homebrew packages it needs at the top. Without an Apple
developer ID the app gets an ad-hoc signature, which is enough to run it on the
Mac that built it.

The included `blacklight-visuals.cfg` preset provides the authored black lightcycle
and rider, a dark reflective grid, flowing translucent ribbons and a compact
cockpit. The [24 October graphics checkpoint](documentation/graphics/accepted-2026-10-24.md)
contains the accepted previews and validation. The [editable Blender source and
export workflow](art/lightcycle-authored/README.md) are included, together with
[the original asset credits and license](art/lightcycle-authored/CREDITS.md).

The ribbon has a coloured transparent interior, moving soft highlights and narrow
hot edges. Its rear connection stays grounded at constant height through turns,
including profiles with Simple Trails enabled. Construction and crash effects
remain presentation features; steering and collision rules are unchanged.
The preset requests 4x anti-aliasing and 16x anisotropic filtering.

To reproduce the reviewed local presentation, put these settings in the user's
`autoexec.cfg`:

```text
INCLUDE blacklight-visuals.cfg
INCLUDE blacklight-handling.cfg
MAX_FPS 120
KEEP_WINDOW_ACTIVE 0
ARMAGETRON_VSYNC 2
```

To verify the actual renderer without opening a window, run
`ai/.venv/bin/python desktop/os-x/capture_preview.py --output ai/runtime/graphics-preview/check`.
The helper uses an isolated spectator arena and saves real screenshots. `--classic`
selects the original appearance; `--load-only` renders without framebuffer readback
for concurrent inference timing. The hidden-rendering path logs active sample count,
drawable dimensions, and rendered throughput.

The Codex Run action uses `script/build_and_run.sh` to build and launch the native
app. For an incremental build and an isolated hidden capture, use
`script/build_and_run.sh --preview --output ai/runtime/graphics-preview/check`.
The preview helper also accepts `--video`, `--capture-at`, and camera settings.
Use `--profile-config /path/to/user_3_1_utf8.cfg` to reproduce saved preferences
in an isolated copy; capture overrides never alter the source profile.
`--pixels 8k` renders an exact 7680×4320 framebuffer and checks the measured size;
`--pixels 4k` selects 3840×2160. These are native rendered RGB8 stills. Still captures
sample readback once per second to avoid streaming every 8K frame; video keeps
every-frame readback. The live game uses the display's native resolution.

The high-detail materials use a 4096-pixel floor atlas, filtered moving trail
detail, and world-oriented clear-coat highlights. `FLOOR_REFLECTION_SOFTNESS`
controls reflection filtering (0 disables its framebuffer copy). Model yaw and
banking preserve velocity through rapid turns; the local Blacklight custom
camera follows time consistently across frame rates. Steering rules are unchanged.

## Documentation

- [Graphics checkpoint and review workflow](documentation/graphics/README.md):
  current native captures, editable Blender source, regression checks and rollback instructions.
- [ai/README.md](ai/README.md): building, training, playing and measuring.
- [ai/docs/V4.md](ai/docs/V4.md): the current architecture, observations, controls, training targets and verification.
- The original Armagetron Advanced documentation is unchanged: `README`, `README-DEVELOPER`, `INSTALL` and `documentation/`.

## License

Armagetron Advanced is free software under the GNU General Public License, version 2 or later; see `COPYING.txt`. The engine changes here are released under the same license.

The authored lightcycle asset is separately licensed under Creative Commons Attribution 3.0.
Retain the [LiNk2.0 and base-human credits](art/lightcycle-authored/CREDITS.md) and supplied license when redistributing it.

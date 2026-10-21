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

The official Mac releases are Intel-only and run through Rosetta. This repository builds the game as a native app for the chip of the Mac it is built on (arm64 on Apple Silicon), with every library inside the bundle, so it runs without Homebrew installed:

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

The included `blacklight-visuals.cfg` preset provides sculpted black lightcycles,
smoked glass with curved circuit inlays, flowing translucent light ribbons,
an illuminated stadium, and a compact cockpit. Optical wheel registers rotate
with distance, reactor light travels through the inlays, wheel hoops and chassis traces assemble bikes
at spawn before their dark shells resolve, and crashes progress from a contact
flash into directional shell fragments. Light walls distort a fresh scene snapshot
with persistent broad folds; nearby silhouettes are protected by scene depth. All are presentation effects.
Enable it with `INCLUDE blacklight-visuals.cfg` in the user's `autoexec.cfg`.
It requests 4× anti-aliasing and 16× anisotropic filtering; unsupported multisampling
falls back to an ordinary framebuffer. These settings affect presentation only.

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
  native captures, editable Blender scene, rollback instructions and known trail issue.
- [ai/README.md](ai/README.md): building, training, playing and measuring.
- [ai/docs/V4.md](ai/docs/V4.md): the current architecture, observations, controls, training targets and verification.
- The original Armagetron Advanced documentation is unchanged: `README`, `README-DEVELOPER`, `INSTALL` and `documentation/`.

## License

Armagetron Advanced is free software under the GNU General Public License, version 2 or later; see `COPYING.txt`. The engine changes here are released under the same license.

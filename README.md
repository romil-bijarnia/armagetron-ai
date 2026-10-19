# Armagetron AI

Armagetron Advanced with a V4 combat policy trained through search-guided self-play.

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

The result is `build-macos/Armagetron Advanced.app`, with a `.dmg` and a `.zip` of it beside it. The script lists the Homebrew packages it needs at the top. Without an Apple developer ID the app gets an ad-hoc signature, which is enough to run it on the Mac that built it.

The included `blacklight-visuals.cfg` preset provides a dark procedural arena grid,
clean light trails and perimeter panels, subdued reflections, and a compact cockpit.
Enable it with `INCLUDE blacklight-visuals.cfg` in the user's `autoexec.cfg`.
It requests 4× anti-aliasing and 8× anisotropic filtering; unsupported multisampling
falls back to an ordinary framebuffer. These settings affect presentation only.

To verify the actual renderer without opening a window, run
`ai/.venv/bin/python desktop/os-x/capture_preview.py --output ai/runtime/graphics-preview/check`.
The helper uses an isolated spectator arena and saves real screenshots. `--classic`
selects the original appearance; `--load-only` renders without framebuffer readback
for concurrent inference timing. The hidden-rendering path logs active sample count,
drawable dimensions, and rendered throughput.

## Documentation

- [ai/README.md](ai/README.md): building, training, playing and measuring.
- [ai/docs/V4.md](ai/docs/V4.md): the current architecture, observations, controls, training targets and verification.
- The original Armagetron Advanced documentation is unchanged: `README`, `README-DEVELOPER`, `INSTALL` and `documentation/`.

## License

Armagetron Advanced is free software under the GNU General Public License, version 2 or later; see `COPYING.txt`. The engine changes here are released under the same license.

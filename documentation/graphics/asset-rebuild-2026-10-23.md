# Authored asset rebuild and local-input trail correction

The preceding refinement was too small, and its spectator test missed the trail
problem reported during local play. This pass uses the supplied Tron: Legacy
film frames as the primary visual reference and authors the model in the native
mesh source. No purchased or downloaded bike/rider model is used.

Installed captures: [bike, glass trail and floor](assets/rebuild-cyan-native.png)
and [rapid turns at the production camera](assets/rebuild-gameplay-native.png).
The close view uses an isolated inspection camera; it does not change the user's
camera. [Capture and asset metadata](asset-rebuild-2026-10-23.json) records the
geometry and installed executable hashes.

## Asset and scene changes

The chassis is a continuous closed shell with curved side panels, transverse
seams and inset power-unit housings. The wheelbase is longer, the hubless openings
are larger, and the wheel inlays are broader. The rider has a lower forward pose,
revised shoulders, arms and hands, a flatter helmet and a larger back disc.
Foot rests connect to the fairing. The visual scale increases from 1.4 to 1.65;
the physics dimensions and production camera remain unchanged.

The wheel axle positions are shared consistently by the model, trail emitter,
construction effect and impact fragments. Coated paint, metal and cloth receive
different microfacet highlights; stadium lamp reflections replace the old
diagonal streaks in the wheel barrels. The floor has visible dark panels and
rounded circuit inlays. Animated trail glass retains refraction and optical
folds, with quieter borders and floor spill and more colour in its distant face.
Bloom intensity decreases from 1.15 to 0.95.

The now-retired Blender export contained the actual
native geometry: 145288 triangles with linked wheel meshes and ten material
groups. Export validation found finite positions and normals, with maximum
normal-length error below 0.0000002. The source of truth remains
`src/tron/gLightcycleModel.h`; Blender export is one way and Blender edits are
not automatically imported back into the game.

Local film reference frames inspected include `003820.png`, `004080.png`,
`006720.png`, `007860.png` and `008800.png` under
`/Users/romilbijarnia/Documents/Tron Legacy Clip Analysis/hdr4k/`. The CGTrader
example was a secondary shape reference. Film frames and marketplace models are
not bundled in the repository or application.

## What caused the remaining trail defect

The prior cubic connection had the correct endpoint and final tangent but could
still cross the rear tyre when the fixed wall lay beside or ahead of the turning
bike. A recorded local-input run contained 32 affected frames. Six captured
geometric cases now form regression fixtures; they failed against the earlier
curve and pass with outboard routing. Mirrored cases and both visual scales are
covered. Another regression guards against a discontinuous side switch as the
emitter crosses the wall centreline.

Some extreme reversals put the physical cut point itself inside the drawn tyre.
Curve routing alone cannot move that fixed point. The wall material therefore
clips only the newborn overlap against the rear-wheel footprint, in the main,
reflection and floor-spill passes. The bike and wall use the same sampled visual
pose. Established collision walls and turn timing remain controlled by the
unchanged simulation.

`capture_preview.py --local-turn-demo` injects SDL left/right press/release events
into an isolated hidden human client, through the same input routing as normal
play. It includes single turns, alternating taps at 75 ms and taps at 40 ms,
faster than the configured cooldown. The capture distinguishes 68 emitted key
events from accepted turns; cooldown and death reject some inputs. It requires
accepted input in both bursts. The oblique final run accepted 39 turns before
self-collision; the final installed normal-camera run accepted 38. This is
materially different from the old neural-socket spectator scenario.

Record video with the single startup still, then extract review frames after
recording. Writing several PNGs during the burst can stall readback and distort
the timing being investigated.

```bash
ai/.venv/bin/python desktop/os-x/capture_preview.py \
  --output ai/runtime/graphics-preview/local-turn-review \
  --local-turn-demo --pixels 1920x1080 --seconds 20 \
  --video ai/runtime/graphics-preview/local-turn-review.mp4 --capture-at 0.1 \
  --set START_CAM_1=1 --set START_FOV_1=90 \
  --set INCLUDE=blacklight-handling.cfg --set MAX_FPS=60 \
  --server-set BLACKLIGHT_VISUALS=1 \
  --server-set SIZE_FACTOR=1 --server-set SP_SIZE_FACTOR=1 \
  --server-set CYCLE_DELAY=0.075 --server-set CYCLE_TURN_SPEED_FACTOR=0.99 \
  --server-set CYCLE_TURN_RADIUS=0
```

## Verification and limits

The cycle-motion, wall-attachment, camera-motion and impact-animation C++ checks
pass. Native gameplay and close inspection views render with both material
shaders linked. The local-input video includes construction, rapid alternating
turns, tight reversals, reflections and a self-collision/fragment sequence.
The classic renderer also passed an installed launch/render smoke check.
The installed application passed its signature verification. Immediately after
installation, the user configuration, autoexec and handling-preset hashes matched
the pre-installation snapshot. The previous app is retained as
`build-macos/previous-installed.zip`. An already running game continues to use its
old executable until reopened.

One 1280×720 installed capture logged a GL 1281 error already pending when scene
sampling began. Both material shaders linked and frame capture continued. The
source of that one diagnostic has not been isolated; it is not described as an
error-free run.

The rider anatomy, environment lighting and film-level material richness remain
below the visual reference. A functioning native render and finite mesh are not
evidence of Disney-quality art. These captures also do not establish universal
trail correctness under network correction, arbitrary maps or longer human play.

An initial 3024×1964 frame-rate comparison was discarded because another installed
game was running concurrently. After that client exited, one clean sequential
pair used 4× MSAA, a 240 FPS cap and no framebuffer readback. Excluding the initial
partial game window, the mean was 122.8 FPS for the preceding installed build and
134.2 FPS for this build. Full-game-window p99 times were 9.3–10.4 ms before and
8.5–12.6 ms after. This is one short pair, not deterministic fight replay or a
repeatable performance improvement. Other desktop work remains uncontrolled.

Temporary working captures and exports were removed in the later 23 October
cleanup; selected captures and metadata remain in this documentation directory.
The saved user config
was rewritten during a concurrent game session; installation checks use a fresh
snapshot and preserve that version rather than restoring an older snapshot.

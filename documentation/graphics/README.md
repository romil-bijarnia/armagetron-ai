# Graphics checkpoints

The [23 October refinement](refinement-2026-10-23.md) updates the bike lighting,
floor grid and trail attachment, with native captures and measured limitations.

The `graphics-checkpoint-2026-10-21` tag preserves the earlier Tron: Legacy-inspired
renderer, animation helpers, native Mac installer, capture utilities and Blender
export. It is a recoverable checkpoint with known issues, not a claim that every
visual defect has been resolved.

This repository is Romil's independent GitHub copy of the original
[Armagetron Advanced repository](https://gitlab.com/armagetronad/armagetronad).
The Git history and the `upstream` remote retain that ancestry. GitHub does not
display a fork-network badge for the GitLab upstream.

## Preserved behaviour

- One flat playing field and the existing gameplay camera height and field of view.
- Sculpted bikes with dark coated fairings, differentiated metal and suit finishes,
  matching bike/trail emission, and white front lamps.
- Animated translucent walls with optical folds, scene refraction, reflected
  detail and filtering for distant views.
- A lower stadium profile, seating detail, floor circuits and filtered reflections.
- Staged bike construction, directional shell fragments and curved wheel fragments.
- Continuous visual yaw and banking through turns, with a bounded positional
  sweep shared by the model and the trail attachment. Movement and collision
  simulation remain unchanged; `CYCLE_TURN_RADIUS` remains zero.
- Muted per-turn sound events through `CYCLE_TURN_SOUND_VOLUME 0`.
- Installation updates only `/Applications/Armagetron Advanced.app`; build outputs
  retain archives, including `previous-installed.zip` for rollback.

## Reference artifacts

These are actual native-renderer captures and a Blender export of the same source
mesh. They are not movie frames or generated substitutes for the game.

| Artifact | Purpose |
| --- | --- |
| [Amber bike and trail](assets/amber-native.png) | Saturated warm emission and transparent wall interior |
| [Blue bike and trail](assets/blue-native.png) | Blue rim/suit emission, without forced-white wheel cores |
| [Blender rendering](assets/amber-blender.png) | Bike material and geometry inspection |
| [Editable Blender scene](assets/lightcycle.blend) | Current native bike and rider, 160456 triangles |
| [Capture metadata](baseline.json) | Source hashes, colour settings and rendering context |

The native colour captures use an isolated inspection camera. Their settings are
recorded in the metadata and do not change the production camera. Both the normal
gameplay camera and inspection view are required for subsequent visual review.
The game captures are RGB8 SDR. The supplied 4K HDR film is an external visual
reference; it is not required to build or run the game.

## Baseline regression: trail spilling or jumping at turns

During repeated 90-degree turns, the fresh cyan sheet can appear to swing or spill
across the bike's rear wheel and then jump to the next segment. Romil reported
this on 21 October 2026. See the [in-game screenshot](assets/trail-turn-spill.png).

The 21 October checkpoint includes the reported behaviour. It does not include a fix.
The attachment interpolation in `gWall.cpp` and the shared visual pose in
`gLightcycleMotion.h` are investigation targets, not a confirmed diagnosis.

A fix must keep the sheet attached to the rear of the rendered bike through a
single turn, rapid repeated turns and reversals. Check the top/bottom edges,
floor reflection and head-on view at the normal camera height. Preserve the
close-up liquid-glass effect, colour consistency and collision behaviour.

## Change and verification workflow

1. Branch from the desired checkpoint before changing graphics. Keep renderer,
   handling and training changes in separate commits or PRs.
2. Run the existing logic tests and build the native client. GitHub Actions runs
   the actual C++ motion, camera and impact checks on Linux and macOS, plus syntax
   checks for the build/capture/export utilities.
3. Capture before and after from the same renderer, resolution, colour, camera
   and scenario. Review close and distant walls, rapid turns, construction,
   impacts, reflections and blue/amber/cyan emission.
4. Run a frame-time check without framebuffer readback or another render job.
   Record other active GPU workloads. GPU power is system-wide; use per-process
   counters before attributing it to the game.
5. Put the evidence and remaining defects in the PR. Tag a reviewed checkpoint
   before replacing it, and do not move an existing checkpoint tag.

Automated logic checks do not establish visual quality or eliminate the open
trail defect. Native in-game comparison remains a required review step.

Run the focused checks locally:

```bash
ai/.venv/bin/python -m pytest \
  ai/tests/test_lightcycle_motion.py \
  ai/tests/test_lightwall_attachment.py \
  ai/tests/test_camera_motion.py \
  ai/tests/test_impact_animation.py -q
```

Build and install the one native application:

```bash
desktop/os-x/build_native_app.sh
```

Capture a separate match using the installed application:

```bash
ai/.venv/bin/python desktop/os-x/capture_preview.py \
  --output ai/runtime/graphics-preview/review \
  --pixels 1920x1080 --seconds 25 --capture-at 15 --capture-at 22 \
  --set START_CAM_1=1 --set START_FOV_1=90 \
  --set INCLUDE=blacklight-handling.cfg --set MAX_FPS=60
```

To restore a checkpoint, first commit or otherwise save any current work, then
create a new branch at the tag and rebuild. Do not reset or clean a dirty tree:

```bash
git switch -c codex/restore-graphics graphics-checkpoint-2026-10-21
desktop/os-x/build_native_app.sh
```

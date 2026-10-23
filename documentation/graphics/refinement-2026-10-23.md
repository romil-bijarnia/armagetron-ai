# Bike, grid and trail refinement — 23 October 2026

This pass improves the existing renderer using its native output as the visual
baseline. It changes bike lighting, the floor pattern, the visible trail surface
and its attachment. The existing geometry, simulation, camera preset, team setup
and training checkpoint are retained.

Installed native captures: [cyan bike and grid](assets/refinement-cyan-native.png)
and [normal-camera turn](assets/refinement-turn-native.png).

## Visible changes

Bike paint receives broader, filtered highlights and more fill illumination.
The fairing and rider have more shape at the existing camera distance while
emissive inlays retain their original colour policy.

The floor uses a regular grid with major panel seams, finer subdivisions and
small junction marks. Line coverage is integrated over the pixel footprint,
using derivatives of world coordinates before the repeating grid wraps. This
prevents thin seams gaining brightness or flashing at distant views. Reflections
use a normalized five-sample filter instead of nine samples.

Distant trails carry a continuous coloured membrane and a broad travelling
wave. The lower border and distant reflected border are quieter. Close walls
retain animated optical folds, refraction and reflected detail. A shorter
reflection ray walk and an earlier transition to the simpler distant material
bound the work per pixel.

## Attachment correction

Previously every recent wall segment blended toward the current rear fender.
Several corners could therefore move at once during rapid turns. The new path
leaves settled segments fixed and draws one short cubic connection from a cut
behind the bike to its rendered rear fender. Its final tangent follows the bike.
The starting handle goes to zero on either side of a physical corner so changing
the owning segment does not jump a control point.

The floor glow also uses shared strip vertices. Independent rectangles for
each curve segment produced a bright fan at tight bends. The continuous strip
narrows with curvature to avoid folding over itself.

Collision walls and turn timing still come from the unchanged simulation.
The short visual connection is presentation geometry, not a new collision path.

## Verification and limits

The four focused C++ checks pass: cycle motion, wall attachment, camera motion
and impact animation. The attachment check covers corner continuity, exact
emitter placement, rearward approach, bounded floor-glow width and rapid turns
at 30, 60, 120 and 240 rendering samples per second.

The new `capture_preview.py --turn-demo` uses scripted actions in a separate
arena. It loads no policy or training checkpoint. The normal-camera and oblique
captures each completed 13 accepted action requests, including alternating
100 ms taps and two turns in the same direction. Native frames were inspected
for the attachment, its reflection, the bright fan and the grid. Both material
shaders linked successfully; no rendering fallback was reported.

The packaged `/Applications/Armagetron Advanced.app` passed its signature check
and completed the same 13-action scenario. Cyan, blue and orange native captures
preserve their respective emission colours. The classic renderer also passed a
launch/render smoke check. SHA-256 checks confirm the saved user configuration,
autoexec and handling preset are unchanged. The prior installed application is
retained in `build-macos/previous-installed.zip`.

Two sets of 26-second hidden-renderer measurements used 3024×1964, 4× MSAA, a
240 FPS cap and no framebuffer readback. In the denser comparison the mean of
logged active-game windows was 152.8 FPS before and 150.4 FPS after. Average
throughput is similar; there is **no demonstrated performance improvement**.
Candidate p99 frame times were 11–19 ms versus 8–10.5 ms for the baseline in this
sample. Frame-time spikes remain an open issue. A preceding sparse-arena sample
also had worse early candidate windows; it is retained with the evidence.

The recordings are short controlled checks. They do not establish aesthetic
acceptance or replace a human playtest of handling and difficult crowded fights.
In particular, network correction, unusual maps, trail holes and extreme camera
angles can expose cases beyond the scripted sequence.

Local captures, commands, raw logs, before/after performance data and installation
logs are kept under `ai/runtime/graphics-preview/refinement-2026-10-23/`.
The original 21 October checkpoint and its reference assets remain available.

Reproduce the turn review with the installed client:

```bash
ai/.venv/bin/python desktop/os-x/capture_preview.py \
  --output ai/runtime/graphics-preview/turn-review \
  --turn-demo --pixels 1920x1080 --seconds 18 \
  --video ai/runtime/graphics-preview/turn-review.mp4 \
  --capture-at 8 --capture-at 9 --capture-at 10 --capture-at 12 \
  --set START_CAM_1=1 --set START_FOV_1=90 \
  --set MAX_FPS=60 --set LAG_O_METER=0 \
  --server-set BLACKLIGHT_VISUALS=1 \
  --server-set SIZE_FACTOR=2 --server-set SP_SIZE_FACTOR=2 \
  --server-set CYCLE_DELAY=0.075 --server-set CYCLE_TURN_RADIUS=0
```

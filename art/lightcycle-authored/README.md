# Authored lightcycle

`lightcycle.blend` is the editable LiNk2.0 cycle and rider used by the game. The
unchanged download is preserved as `source/original.blend`; see [CREDITS.md](CREDITS.md)
and [the supplied license](source/BLENDSWAP_LICENSE.txt). The derivative retains
original control geometry, UVs and the posed rig beneath reversible smoothing,
rear-shell clearance and rear-fin edits. Its evaluated export has 51,744 triangles.

The finish uses black non-emissive paint and suit materials, repaired optical
markings and a flush rear stripe. Blender's **Cycle and trail colour** node group
controls the shared colour and intensity. The saved file uses Standard display
with linear equivalents of the selected display colour. All required textures
are packed into the source scene.

The native renderer loads `models/lightcycle-authored.bin` and the scalar
`textures/lightcycle_suit_mask.png`. Geometry uploads once per graphics context;
the wheels rotate separately using their exported radii. The rider uses the
exported pose. `src/tron/gLightcycleProfile.h` fits the live ribbon to the same
rear silhouette. [game-export.json](game-export.json) records the asset hashes,
wheel centres, profile and triangle counts.

The live ribbon is a translucent coloured membrane with flowing highlights,
a thin upper lip and a stronger lower light band. Pale highlights are confined
to its narrow optical cores. Its geometry remains grounded at constant height,
with bounded surface rows through turns. Blacklight uses this connection even
when an existing profile has `SIMPLE_TRAIL 1`. Classic rendering keeps its own
Simple Trails behaviour. The static Blender ribbon is a material study; the game
constructs live trail geometry and does not export that static strip.

## Re-export after editing

From the repository root, use Blender 5.1 or a compatible release:

```sh
/Applications/Blender.app/Contents/MacOS/Blender \
  --background --disable-autoexec --threads 4 --python-exit-code 1 \
  --python art/lightcycle-authored/export_game.py
desktop/os-x/build_native_app.sh
```

The exporter reads the saved scene, evaluates its modifiers and pose, and bakes
emission coverage. It does not save changes to the Blender source. The resulting
mesh, mask, generated profile and export manifest should be versioned together.

The [accepted graphics checkpoint](../../documentation/graphics/accepted-2026-10-24.md)
contains current native previews and verification. Local render caches, machine
configuration dumps, Blender backup files and one-time migration scripts remain
ignored. Blender and the game use different lighting engines; their reflections
and bloom are not pixel-identical. Fabric normal-map detail is not baked into the
native mesh.

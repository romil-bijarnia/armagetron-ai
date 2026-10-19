#!/usr/bin/env python3
"""Export the actual native lightcycle cache to an editable Blender studio scene.

The C++ helper uses the configured native build's compiler/flags and calls only
Geometry(); no game process, OpenGL context, runtime camera or config is touched.
The .blend contains the original triangles and split normals, ten named material
groups, linked wheel meshes at the source axle constants, and two studio cameras.
Blender's physical materials approximate the game's custom lighting; this is a
one-way export, so saving edits in Blender does not update the C++ renderer.

Examples:
  python3 desktop/os-x/export_lightcycle_blender.py --output /path/model-revision --skip-render
  python3 desktop/os-x/export_lightcycle_blender.py --output /path/model-revision --force
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HELPER = Path(__file__).with_name("dump_lightcycle_mesh.cpp")
HEADER = ROOT / "src/tron/gLightcycleModel.h"
BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")
FLAG_NAMES = ("CXX", "CPPFLAGS", "CXXFLAGS", "DEFAULT_INCLUDES", "libtron_a_CXXFLAGS")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="directory for mesh JSON, .blend, renders and logs")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-macos", help="configured native build directory")
    parser.add_argument("--blender", type=Path, default=BLENDER)
    parser.add_argument("--dump-only", action="store_true", help="only compile/run the CPU mesh exporter")
    parser.add_argument("--skip-render", action="store_true", help="save the editable .blend without rendering images")
    parser.add_argument("--force", action="store_true", help="replace this utility's existing generated .blend and renders")
    parser.add_argument("--color", default="0.01,0.86,1.0", help="linear team emission RGB, comma separated")
    parser.add_argument("--world-scale", type=float, default=.5, help="root scale; .5 matches the native caller")
    parser.add_argument("--rear-angle", type=float, default=0, help="rear wheel angle in radians")
    parser.add_argument("--front-angle", type=float, default=0, help="front wheel angle in radians")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--samples", type=int, default=32, help="Cycles CPU samples per studio render")
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--blender-stage", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        args.team_color = tuple(float(value) for value in args.color.split(","))
    except ValueError:
        parser.error("--color must contain three numbers")
    if len(args.team_color) != 3 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in args.team_color):
        parser.error("--color must contain three finite RGB values between zero and one")
    if not math.isfinite(args.world_scale) or args.world_scale <= 0:
        parser.error("--world-scale must be positive and finite")
    if not all(math.isfinite(value) for value in (args.rear_angle, args.front_angle)):
        parser.error("wheel angles must be finite")
    if not 128 <= args.width <= 8192 or not 128 <= args.height <= 8192:
        parser.error("render dimensions must be between 128 and 8192")
    if not 1 <= args.samples <= 4096 or not 1 <= args.threads <= 128:
        parser.error("samples or thread count is out of range")
    args.output = args.output.expanduser().resolve()
    args.build_dir = args.build_dir.expanduser().resolve()
    args.blender = args.blender.expanduser().resolve()
    return args


def native_flags(makefile: Path, scratch: Path):
    """Ask make to expand a variable-only snapshot, never a game build target."""
    if not makefile.is_file():
        raise RuntimeError(f"No configured native Makefile at {makefile}")
    logical = makefile.read_text().replace("\\\n", " ")
    assignments = [line for line in logical.splitlines()
                   if re.match(r"^[A-Za-z_][A-Za-z_0-9]*\s*[:+?]?=", line)]
    snapshot = scratch / "native-flags.mk"
    snapshot.write_text("\n".join(assignments) + "\n"
                        + "\n".join(f"$(info __LIGHTCYCLE_{name}__=$({name}))" for name in FLAG_NAMES)
                        + "\n.PHONY: lightcycle-export-flags\nlightcycle-export-flags:\n\t@:\n")
    result = subprocess.run(["make", "-rR", "--no-print-directory", "-f", str(snapshot), "lightcycle-export-flags"],
                            cwd=makefile.parent, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    flags = {}
    for line in result.stdout.splitlines():
        match = re.fullmatch(r"__LIGHTCYCLE_([A-Za-z_0-9]+)__=(.*)", line)
        if match:
            flags[match.group(1)] = match.group(2)
    if set(flags) != set(FLAG_NAMES) or not flags["CXX"]:
        raise RuntimeError("Could not expand the native compiler configuration")
    return flags


def export_native(args):
    args.output.mkdir(parents=True, exist_ok=True)
    makefile = args.build_dir / "src/Makefile"
    header_hash = digest(HEADER)
    with tempfile.TemporaryDirectory(prefix=".native-export-", suffix=".noindex", dir=args.output) as temporary:
        scratch = Path(temporary)
        flags = native_flags(makefile, scratch)
        executable = scratch / "dump-lightcycle"
        command = shlex.split(flags["CXX"])
        for name in FLAG_NAMES[1:]:
            command.extend(shlex.split(flags[name]))
        command.extend([str(HELPER), "-Wl,-dead_strip", "-o", str(executable)])
        with (args.output / "compile.log").open("w") as log:
            result = subprocess.run(command, cwd=makefile.parent, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"CPU dump helper did not compile; see {args.output / 'compile.log'}")
        temporary_mesh = scratch / "native-mesh.json"
        with temporary_mesh.open("wb") as output, (args.output / "dump.log").open("wb") as log:
            result = subprocess.run([str(executable)], stdout=output, stderr=log)
        if result.returncode:
            raise RuntimeError(f"CPU mesh export failed; see {args.output / 'dump.log'}")
        if digest(HEADER) != header_hash:
            raise RuntimeError("The source header changed during export; rerun after the edit finishes")
        # Validate before replacing any prior mesh artifact.
        mesh = json.loads(temporary_mesh.read_text())
        if mesh.get("schema_version") != 1 or len(mesh.get("materials", [])) != 10:
            raise RuntimeError("Unexpected native mesh schema/material count")
        shutil.copyfile(temporary_mesh, args.output / "native-mesh.json")
    manifest = {
        "source_header": str(HEADER), "source_header_sha256": header_hash,
        "dump_helper_sha256": digest(HELPER), "native_makefile": str(makefile),
        "native_makefile_sha256": digest(makefile), "compiler_flags": flags,
        "compiler_command": command, "mesh_sha256": digest(args.output / "native-mesh.json"),
        "constants": mesh["constants"], "native_summary": mesh["summary"],
        "team_emission_rgb": args.team_color, "world_scale": args.world_scale,
        "rear_angle_radians": args.rear_angle, "front_angle_radians": args.front_angle,
        "geometry_source": "gLightcycleModel::Detail::Geometry() CPU cache",
        "notes": ["Exact source triangles and split normals; no substitute/proxy model.",
                  "Blender physical materials approximate the custom native shader.",
                  "One-way export: Blender edits do not update the game source."],
    }
    (args.output / "export-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def studio_materials(bpy, names, color):
    # All ten native groups stay separate, with dark polished fairings, readable
    # machined silver, fabric suit and exact-hue emission without compositor glow.
    specification = {
        "Shell": ((.025, .033, .043), .68, .22),
        "Armour": ((.055, .071, .088), .55, .32),
        "Metal": ((.34, .39, .45), .88, .26),
        "Rubber": ((.009, .012, .016), 0, .54),
        "Recess": ((.005, .007, .010), .20, .55),
        "Suit": ((.017, .022, .030), 0, .77),
        "Visor": ((.008, .020, .029), .62, .12),
    }
    materials = []
    for index, name in enumerate(names):
        material = bpy.data.materials.new(f"{index:02d} {name}")
        material.use_nodes = True
        shader = material.node_tree.nodes.get("Principled BSDF")
        if name in ("Light", "SubtleLight", "Headlight"):
            hue = (.77, .87, .95) if name == "Headlight" else color
            power = .35 if name == "SubtleLight" else 1.0
            shader.inputs["Base Color"].default_value = (0, 0, 0, 1)
            shader.inputs["Roughness"].default_value = .30
            shader.inputs["Emission Color"].default_value = (*hue, 1)
            shader.inputs["Emission Strength"].default_value = power
            material.diffuse_color = (*hue, 1)
        else:
            base, metallic, roughness = specification[name]
            shader.inputs["Base Color"].default_value = (*base, 1)
            shader.inputs["Metallic"].default_value = metallic
            shader.inputs["Roughness"].default_value = roughness
            material.diffuse_color = (*base, 1)
            if name in ("Shell", "Visor"):
                shader.inputs["Coat Weight"].default_value = .45
                shader.inputs["Coat Roughness"].default_value = .16
            if name == "Suit":
                shader.inputs["Sheen Weight"].default_value = .32
        material["native_material_index"] = index
        material["native_material_name"] = name
        materials.append(material)
    return materials


def native_mesh(bpy, group, name, material):
    """Weld identical positions, preserving every native triangle and loop normal."""
    vertices, faces, loop_normals = [], [], []
    lookup = {}
    source = group["vertices"]
    for start in range(0, len(source), 3):
        face = []
        for row in source[start:start + 3]:
            position = tuple(row[:3])
            if position not in lookup:
                lookup[position] = len(vertices)
                vertices.append(position)
            face.append(lookup[position])
            loop_normals.append(tuple(row[3:]))
        faces.append(face)
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    if mesh.validate(clean_customdata=False):
        raise RuntimeError(f"Blender validation changed the native topology of {name}")
    mesh.update(calc_edges=True)
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    mesh.normals_split_custom_set(loop_normals)
    mesh.materials.append(material)
    mesh["native_material_index"] = group["material"]
    mesh["native_triangle_count"] = len(faces)
    mesh["native_vertex_count"] = len(source)
    return mesh


def blender_scene(args):
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.filepaths.save_version = 0
    scene = bpy.context.scene
    data = json.loads((args.output / "native-mesh.json").read_text())
    manifest = json.loads((args.output / "export-manifest.json").read_text())
    materials = studio_materials(bpy, data["materials"], args.team_color)
    model_collection = bpy.data.collections.new("Native Lightcycle")
    studio_collection = bpy.data.collections.new("Studio — export review only")
    scene.collection.children.link(model_collection)
    scene.collection.children.link(studio_collection)

    def empty(name, parent=None, location=(0, 0, 0)):
        obj = bpy.data.objects.new(name, None)
        model_collection.objects.link(obj)
        obj.parent = parent
        obj.location = location
        obj.empty_display_size = .1
        return obj

    root = empty("Native Lightcycle · edit this hierarchy")
    root.scale = (args.world_scale,) * 3
    root["source_header_sha256"] = manifest["source_header_sha256"]
    root["native_mesh_sha256"] = manifest["mesh_sha256"]
    root["export_note"] = "Exact native triangles; saving edits here does not modify C++ source."
    body = empty("Body", root)
    constants = data["constants"]
    rear = empty("Rear wheel pivot", root, (constants["rear_axle"], 0, constants["wheel_radius"]))
    front = empty("Front wheel pivot", root, (constants["front_axle"], 0, constants["wheel_radius"]))
    rear.rotation_euler.y = -args.rear_angle
    front.rotation_euler.y = -args.front_angle
    native_objects = []
    source_triangles = 0
    for kind, parents in (("body", (body,)), ("wheel", (rear, front))):
        for group in data["parts"][kind]:
            if not group["vertices"]:
                continue
            index = group["material"]
            name = data["materials"][index]
            mesh = native_mesh(bpy, group, f"Native {kind} · {name}", materials[index])
            for parent in parents:
                obj = bpy.data.objects.new(f"{parent.name} · {name}", mesh)
                model_collection.objects.link(obj)
                obj.parent = parent
                native_objects.append(obj)
                source_triangles += len(mesh.polygons)
    if source_triangles != data["summary"]["complete"]["triangles"]:
        raise RuntimeError("Blender/native triangle counts differ")
    bpy.context.view_layer.update()
    points = [obj.matrix_world @ Vector(corner) for obj in native_objects for corner in obj.bound_box]
    low = Vector(tuple(min(point[i] for point in points) for i in range(3)))
    high = Vector(tuple(max(point[i] for point in points) for i in range(3)))
    center = (low + high) * .5
    size = high - low
    extent = max(size)

    # Neutral, physical studio lighting. No arena, native post-processing, glare,
    # perspective distortion or camera tuning is carried into the review scene.
    world = bpy.data.worlds.new("Neutral studio world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (.18, .18, .18, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = .45
    scene.world = world
    ground = bpy.data.meshes.new("Studio ground mesh")
    half = extent * 5
    ground.from_pydata([(-half, -half, low.z - .005), (half, -half, low.z - .005),
                       (half, half, low.z - .005), (-half, half, low.z - .005)], [], [(0, 1, 2, 3)])
    floor = bpy.data.objects.new("Neutral ground", ground)
    studio_collection.objects.link(floor)
    floor_material = bpy.data.materials.new("Studio matte grey")
    floor_material.use_nodes = True
    floor_shader = floor_material.node_tree.nodes.get("Principled BSDF")
    floor_shader.inputs["Base Color"].default_value = (.16, .16, .16, 1)
    floor_shader.inputs["Roughness"].default_value = .64
    ground.materials.append(floor_material)

    def area(name, offset, energy, width, height=None):
        light = bpy.data.lights.new(name, "AREA")
        light.energy = energy
        light.shape = "RECTANGLE" if height else "DISK"
        light.size = width
        if height:
            light.size_y = height
        obj = bpy.data.objects.new(name, light)
        studio_collection.objects.link(obj)
        obj.location = center + Vector(offset) * extent
        obj.rotation_euler = (center - obj.location).to_track_quat("-Z", "Y").to_euler()

    area("Key softbox", (.1, -1.25, 1.7), 850, extent * 1.3, extent * .65)
    area("Fill softbox", (.9, 1.0, .9), 550, extent * 1.2, extent)
    area("Rear edge softbox", (-1.2, .7, 1.2), 750, extent * .75, extent * 1.2)
    area("Top strip", (0, 0, 2.0), 450, extent * 1.6, extent * .3)
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100

    def camera(name, direction):
        camera_data = bpy.data.cameras.new(name)
        camera_data.type = "ORTHO"
        obj = bpy.data.objects.new(name, camera_data)
        studio_collection.objects.link(obj)
        obj.location = center + Vector(direction).normalized() * extent * 6
        obj.rotation_euler = (center - obj.location).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        inverse = obj.matrix_world.inverted()
        local = [inverse @ point for point in points]
        width = max(point.x for point in local) - min(point.x for point in local)
        height = max(point.y for point in local) - min(point.y for point in local)
        midpoint = Vector(((max(point.x for point in local) + min(point.x for point in local)) * .5,
                           (max(point.y for point in local) + min(point.y for point in local)) * .5, 0))
        obj.location += obj.rotation_euler.to_quaternion() @ midpoint
        camera_data.ortho_scale = max(width, height, .001)
        camera_data.lens = 55
        camera_data.clip_start = .01
        camera_data.clip_end = extent * 100
        bpy.context.view_layer.update()
        # Camera sensor-fit changes the meaning of ortho_scale with the image
        # aspect ratio. Fit Blender's actual projection instead of assuming that
        # ortho_scale is always the vertical size of the frame.
        projected = [world_to_camera_view(scene, obj, point) for point in points]
        span = max(max(point.x for point in projected) - min(point.x for point in projected),
                   max(point.y for point in projected) - min(point.y for point in projected))
        camera_data.ortho_scale *= span * 1.26
        projected = [world_to_camera_view(scene, obj, point) for point in points]
        obj["frame_bounds_uv"] = [min(point.x for point in projected), min(point.y for point in projected),
                                  max(point.x for point in projected), max(point.y for point in projected)]
        return obj

    side = camera("Side orthographic · proportions", (0, -1, 0))
    three_quarter = camera("Front three-quarter orthographic", (1.25, -1.8, .85))
    scene.camera = three_quarter
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    scene.cycles.seed = 0
    scene.cycles.use_denoising = True
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.render.threads_mode = "FIXED"
    scene.render.threads = args.threads
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    scene.use_nodes = False
    scene.unit_settings.system = "METRIC"
    for screen in bpy.data.screens:
        for space in screen.areas:
            if space.type == "VIEW_3D":
                space.spaces.active.shading.type = "SOLID"
                space.spaces.active.shading.color_type = "MATERIAL"
    note = bpy.data.texts.new("NATIVE EXPORT — provenance and limits")
    note.write(json.dumps(manifest, indent=2) + "\n\n"
               "Editable exact native triangles, with source normals and linked wheels.\n"
               "Materials and neutral studio lighting are Blender approximations, not the native GLSL shader.\n"
               "No compositor glow. Side and three-quarter cameras are orthographic.\n"
               "Saving this file does not update the game; edit a copy to preserve future exports.\n")
    blend_path = args.output / "lightcycle-native.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), compress=True)
    rendered = []
    if not args.skip_render:
        for name, view in (("side", side), ("three-quarter", three_quarter)):
            scene.camera = view
            scene.render.filepath = str(args.output / f"{name}.png")
            bpy.ops.render.render(write_still=True)
            rendered.append(scene.render.filepath)
    report = {
        "blend": str(blend_path), "blender_version": bpy.app.version_string,
        "native_triangle_count": data["summary"]["complete"]["triangles"],
        "blender_instanced_triangle_count": source_triangles, "native_mesh_objects": len(native_objects),
        "material_group_count": len(materials), "linked_wheel_meshes": True,
        "world_bounds": [list(low), list(high)], "renders_skipped": args.skip_render,
        "source_header_sha256": manifest["source_header_sha256"], "native_mesh_sha256": manifest["mesh_sha256"],
        "camera_frame_bounds_uv": {"side": list(side["frame_bounds_uv"]),
                                   "three_quarter": list(three_quarter["frame_bounds_uv"])},
        "rendered_files": rendered, "engine": "Cycles CPU", "samples": args.samples,
    }
    (args.output / "scene-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    in_blender = "--blender-stage" in sys.argv
    args = arguments(sys.argv[sys.argv.index("--") + 1:] if in_blender else None)
    if in_blender:
        blender_scene(args)
        return
    if not args.dump_only and (args.output / "lightcycle-native.blend").exists() and not args.force:
        raise RuntimeError("The generated .blend already exists; use another output directory or --force")
    manifest = export_native(args)
    if args.dump_only:
        print(json.dumps({"mesh": str(args.output / "native-mesh.json"),
                          "summary": manifest["native_summary"]}, indent=2))
        return
    if not args.blender.is_file():
        raise RuntimeError(f"Blender executable is missing: {args.blender}")
    command = [str(args.blender), "--background", "--factory-startup", "--threads", str(args.threads),
               "--python-exit-code", "1", "--python", str(Path(__file__).resolve()), "--",
               "--blender-stage", "--output", str(args.output), "--color", args.color,
               "--world-scale", str(args.world_scale), "--rear-angle", str(args.rear_angle),
               "--front-angle", str(args.front_angle), "--width", str(args.width),
               "--height", str(args.height), "--samples", str(args.samples), "--threads", str(args.threads)]
    if args.skip_render:
        command.append("--skip-render")
    with (args.output / "blender.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Blender export failed; see {args.output / 'blender.log'}")
    print((args.output / "scene-report.json").read_text())


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
        print(f"Lightcycle export: {error}", file=sys.stderr)
        raise SystemExit(1)

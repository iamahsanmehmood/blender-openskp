"""Run with Blender's own headless mode:

    blender --background --python tests/test_export.py

Builds a real Blender cube, exports it via export_skp.export_skp(), then
reads the result back through openskp's OWN Python reader (not this
addon) and checks the exact face count, per-face vertex count, and
world-space coordinates (after the metres->inches conversion) round-trip
losslessly.

The per-face vertex count check is the regression test for a real bug: an
earlier version exported every quad face as two triangles sharing a
diagonal, so a plain cube exported with a spurious "middle line" across
every side - caught by testing a real export in SketchUp/FreeCAD, not
found here first. A cube's six faces must each come back as a single
4-vertex loop, not two 3-vertex ones.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _load_addon import _load_module, ensure_openskp_importable  # noqa: E402

ensure_openskp_importable()
export_skp = _load_module("export_skp")

import bpy  # noqa: E402

METRES_TO_INCHES = 1.0 / 0.0254


def main():
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(1.0, 2.0, 3.0))
    cube = bpy.context.active_object
    cube.select_set(True)
    bpy.context.view_layer.objects.active = cube

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_export_test.skp"
    )
    stats = export_skp.export_skp(out_path, bpy.context)
    print("export stats:", stats)
    assert stats == {
        "faces_written": 6,
        "faces_skipped": 0,
        "objects_skipped": 0,
        "objects_exported": 1,
        "materials_written": 0,
        "layers_written": 0,
        "definitions_written": 0,
        "instances_written": 0,
    }, stats

    ensure_openskp_importable()
    import openskp

    model = openskp.SkpFile.open(out_path).parse()
    os.remove(out_path)

    assert len(model.root.faces) == 6, f"expected 6 faces, got {len(model.root.faces)}"

    # Each face must be a single 4-vertex loop (the cube's actual quad
    # boundary) - not two 3-vertex loops from an unwanted triangulation.
    for f in model.root.faces.values():
        assert len(f.loops) == 1, f"expected 1 loop per face, got {len(f.loops)}"
        assert len(f.loops[0]) == 4, f"expected a 4-vertex face, got {len(f.loops[0])}"

    # A 2m cube at (1, 2, 3) spans, in inches: x [0, 78.74], y [39.37,
    # 118.11], z [78.74, 157.48]. Check every stored vertex lands exactly
    # on that box (exact, not approximate - this is a pure unit-scale
    # conversion, so it should be lossless to float precision).
    lo = (0.0, 1.0 * METRES_TO_INCHES, 2.0 * METRES_TO_INCHES)
    hi = (2.0 * METRES_TO_INCHES, 3.0 * METRES_TO_INCHES, 4.0 * METRES_TO_INCHES)
    for v in model.root.vertices.values():
        for coord, lo_c, hi_c in zip((v.x, v.y, v.z), lo, hi):
            assert lo_c - 1e-6 <= coord <= hi_c + 1e-6, (v.x, v.y, v.z, lo, hi)

    print("test_export: OK (6 faces, each a clean 4-vertex quad, coordinates exact)")


def test_layer_export():
    """Regression test for layer (SketchUp "tag") export: an object's
    layer is whichever Collection it's linked into, other than the
    scene's own default/root ones (see export_skp.py's
    _object_layer_name docstring for the exact exclusion rules).
    Selects only the three cubes this creates - explicitly, not "every
    scene object" - since a background Blender session starts with its
    own default Cube/Light/Camera already in the scene (confirmed
    directly) that must NOT be swept into this export."""
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
    default_obj = bpy.context.active_object
    default_obj.name = "DefaultLayerCube"

    studs_coll = bpy.data.collections.new("Studs")
    bpy.context.scene.collection.children.link(studs_coll)
    bpy.context.view_layer.active_layer_collection = (
        bpy.context.view_layer.layer_collection.children["Studs"]
    )
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(5, 0, 0))
    studs_obj = bpy.context.active_object
    studs_obj.name = "StudCube"

    plates_coll = bpy.data.collections.new("Plates")
    bpy.context.scene.collection.children.link(plates_coll)
    bpy.context.view_layer.active_layer_collection = (
        bpy.context.view_layer.layer_collection.children["Plates"]
    )
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(10, 0, 0))
    plates_obj = bpy.context.active_object
    plates_obj.name = "PlateCube"

    for obj in (default_obj, studs_obj, plates_obj):
        obj.select_set(True)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_layer_export_test.skp")
    stats = export_skp.export_skp(out_path, bpy.context)
    print("layer export stats:", stats)
    assert stats == {
        "faces_written": 18, "faces_skipped": 0, "objects_skipped": 0,
        "objects_exported": 3, "materials_written": 0, "layers_written": 2,
        "definitions_written": 0, "instances_written": 0,
    }, stats

    import openskp

    model = openskp.SkpFile.open(out_path).parse()
    os.remove(out_path)

    layer_names = sorted(l.name for l in model.layers)
    assert layer_names == ["Layer0", "Plates", "Studs"], layer_names

    # Faces split into exactly 3 distinct Face.layer ids (SketchUp's own
    # per-face layer field, `None` reading back as the implicit default)
    # - one per object, 6 faces (one cube) each - not just "some layer
    # got written," a real per-face assignment cross-check.
    by_layer: dict = {}
    for f in model.root.faces.values():
        by_layer.setdefault(f.layer, 0)
        by_layer[f.layer] += 1
    assert len(by_layer) == 3, f"expected 3 distinct Face.layer groups, got {by_layer}"
    assert sorted(by_layer.values()) == [6, 6, 6], by_layer
    # openskp's writer encodes "no explicit layer" as a raw 0 (SketchUp's
    # own default-layer sentinel, matching how a 0 material_id means "no
    # material") - confirmed directly, not assumed from Face.layer's own
    # docstring (which describes the READER's `None` convention for a
    # face that never had the field touched at all, a different case).
    assert by_layer.get(0, 0) == 6, f"the unorganized cube's faces should carry the default-layer sentinel: {by_layer}"

    print(f"test_layer_export: OK ({layer_names}, 6 faces each, correctly split by Face.layer)")


def test_material_export():
    """Regression test for material export: a polygon's own material
    slot (poly.material_index into obj.data.materials) becomes a
    SketchUp material read from Material.diffuse_color. Two cubes, one
    opaque red, one 50%-alpha translucent blue, sharing no materials -
    plus a plain cube with no material at all, to confirm an unpainted
    object still exports cleanly."""
    # test_layer_export() (run just before this, in the same session)
    # leaves the active layer collection pointed at "Plates" - reset it
    # to the scene's own root, or these new cubes would silently land
    # inside "Plates" too and inflate layers_written below.
    bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection

    red = bpy.data.materials.new(name="ExportRed")
    red.diffuse_color = (1.0, 0.0, 0.0, 1.0)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(20, 0, 0))
    red_obj = bpy.context.active_object
    red_obj.name = "RedCube"
    red_obj.data.materials.append(red)

    blue = bpy.data.materials.new(name="ExportBlue")
    blue.diffuse_color = (0.0, 0.0, 1.0, 0.5)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(25, 0, 0))
    blue_obj = bpy.context.active_object
    blue_obj.name = "BlueCube"
    blue_obj.data.materials.append(blue)

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(30, 0, 0))
    plain_obj = bpy.context.active_object
    plain_obj.name = "PlainCube"

    for obj in (red_obj, blue_obj, plain_obj):
        obj.select_set(True)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_material_export_test.skp")
    stats = export_skp.export_skp(out_path, bpy.context)
    print("material export stats:", stats)
    assert stats == {
        "faces_written": 18, "faces_skipped": 0, "objects_skipped": 0,
        "objects_exported": 3, "materials_written": 2, "layers_written": 0,
        "definitions_written": 0, "instances_written": 0,
    }, stats

    import openskp

    model = openskp.SkpFile.open(out_path).parse()
    os.remove(out_path)

    materials_by_name = {m.name: m for m in model.materials}
    assert set(materials_by_name) == {"ExportRed", "ExportBlue"}, materials_by_name

    red_mat = materials_by_name["ExportRed"]
    assert red_mat.color[:3] == (255, 0, 0), red_mat.color
    assert red_mat.transparency == 1.0, red_mat.transparency

    blue_mat = materials_by_name["ExportBlue"]
    assert blue_mat.color[:3] == (0, 0, 255), blue_mat.color
    assert abs(blue_mat.transparency - 0.5) < 1e-3, blue_mat.transparency

    # Cross-check per-face assignment, not just "the material records
    # exist": faces split into 3 groups by material_id - red's, blue's,
    # and the unpainted cube's `None` - 6 faces each.
    by_material: dict = {}
    for f in model.root.faces.values():
        by_material.setdefault(f.material_id, 0)
        by_material[f.material_id] += 1
    assert len(by_material) == 3, f"expected 3 distinct Face.material_id groups, got {by_material}"
    assert sorted(by_material.values()) == [6, 6, 6], by_material
    assert by_material.get(None, 0) == 6, f"the unpainted cube's faces should carry no material_id: {by_material}"

    print("test_material_export: OK (red opaque, blue 50% alpha, plain unpainted - all correctly split)")


def test_material_colour_round_trips():
    """Blender colours are scene-linear and SketchUp's are sRGB: the importer
    converts SketchUp's colour to linear, so the exporter must convert it
    back - SketchUp's grey (126, 127, 127) leaves Blender as the same grey,
    not as the much darker (52, 53, 53) a straight *255 gives."""
    import_skp = _load_module("import_skp")
    grey = bpy.data.materials.new("Grey")
    grey.diffuse_color = (*import_skp.srgb_to_linear((126 / 255, 127 / 255, 127 / 255)), 1.0)
    rgba, opacity = export_skp._material_rgba_opacity(grey)
    assert rgba == (126, 127, 127) and opacity is None, rgba
    print("test_material_colour_round_trips: OK (SketchUp grey survives import and export)")


def _fold_m13(m13):
    """Folds openskp's 13-float column-major Instance.matrix into 12 floats
    (col0, col1, col2, translation_in)."""
    s = m13[12] if len(m13) > 12 else 1.0
    return tuple(round(v * s, 5) for v in m13[:12])


def _model_world_face_signature(model):
    """Walks a parsed SkpModel's root faces and recursive component instances
    in world space (inches), returning (face_count, total_area_sq_in, vertex_set)."""
    import math
    from openskp._core import multiply_matrices, transform_point

    def poly_area(pts):
        if len(pts) < 3:
            return 0.0
        x0, y0, z0 = pts[0]
        cx = cy = cz = 0.0
        for i in range(1, len(pts) - 1):
            ux, uy, uz = pts[i][0] - x0, pts[i][1] - y0, pts[i][2] - z0
            vx, vy, vz = pts[i + 1][0] - x0, pts[i + 1][1] - y0, pts[i + 1][2] - z0
            cx += uy * vz - uz * vy
            cy += uz * vx - ux * vz
            cz += ux * vy - uy * vx
        return 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)

    def loop_verts(defn, loop):
        pts = []
        for edge_id, sign in loop:
            edge = defn.edges.get(edge_id)
            if edge is None:
                continue
            vid = edge.v1_id if sign > 0 else edge.v2_id
            if vid in defn.vertices:
                v = defn.vertices[vid]
                pts.append((v.x, v.y, v.z))
        return pts

    face_count = 0
    total_area = 0.0
    verts = set()

    def walk_def(defn, world_m):
        nonlocal face_count, total_area
        for face in defn.faces.values():
            if not face.loops:
                continue
            outer = [transform_point(p, world_m) for p in loop_verts(defn, face.loops[0])]
            if len(outer) < 3:
                continue
            area = poly_area(outer)
            for hole in face.loops[1:]:
                hpts = [transform_point(p, world_m) for p in loop_verts(defn, hole)]
                if len(hpts) >= 3:
                    area -= poly_area(hpts)
            face_count += 1
            total_area += max(0.0, area)
            for loop in face.loops:
                for p in loop_verts(defn, loop):
                    px, py, pz = transform_point(p, world_m)
                    verts.add((round(px, 6), round(py, 6), round(pz, 6)))
        for inst in defn.instances:
            child_def = model.definitions.get(inst.ref_idx)
            if child_def is None:
                continue
            walk_def(child_def, multiply_matrices(world_m, inst.matrix))

    ident = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    walk_def(model.root, ident)
    return face_count, round(total_area, 3), sorted(verts)


def _flat_export_scene_to_skp(filepath, objs):
    """Writes ``objs`` flattened into loose root faces (expanding every
    collection-instance Empty into world space, the pre-#28 flat export) so
    tests can verify exact world-space geometry parity against component export."""
    from openskp import create

    builder = create()
    dummy_stats = {"faces_skipped": 0}

    def emit_collection(coll, parent_mw):
        for o in coll.objects:
            if o.type == "MESH" and o.data is not None and len(o.data.polygons) > 0:
                for pts, _spec in export_skp._extract_mesh_faces(o, parent_mw @ o.matrix_local, dummy_stats):
                    builder.add_face(pts)
            elif export_skp._is_collection_empty(o):
                emit_collection(o.instance_collection, parent_mw @ o.matrix_local)

    for obj in objs:
        if obj.type == "MESH" and obj.data is not None and len(obj.data.polygons) > 0:
            for pts, _spec in export_skp._extract_mesh_faces(obj, obj.matrix_world, dummy_stats):
                builder.add_face(pts)
        elif export_skp._is_collection_empty(obj):
            emit_collection(obj.instance_collection, obj.matrix_world)

    with open(filepath, "wb") as f:
        f.write(builder.to_bytes())


def _assert_same_vertices(pts_a, pts_b, tol=1e-3, label=""):
    """Verifies every 3D point in ``pts_a`` lies within ``tol`` inches of a
    point in ``pts_b`` and vice versa (immune to half-ULP rounding boundaries
    from Blender's float32 matrix_world)."""
    def max_min_dist(src, dst):
        # Sort by X for fast narrow-window search
        dst_sorted = sorted(dst)
        import bisect

        worst = 0.0
        worst_pt = None
        xs = [p[0] for p in dst_sorted]
        for px, py, pz in src:
            lo = bisect.bisect_left(xs, px - tol * 2.0)
            hi = bisect.bisect_right(xs, px + tol * 2.0)
            best = float("inf")
            for qx, qy, qz in dst_sorted[lo:hi]:
                d2 = (px - qx) ** 2 + (py - qy) ** 2 + (pz - qz) ** 2
                if d2 < best:
                    best = d2
            dist = best ** 0.5
            if dist > worst:
                worst = dist
                worst_pt = (px, py, pz)
        return worst, worst_pt

    d_ab, pt_ab = max_min_dist(pts_a, pts_b)
    d_ba, pt_ba = max_min_dist(pts_b, pts_a)
    assert d_ab <= tol and d_ba <= tol, (
        f"{label}: vertex mismatch max_dist=({d_ab:.6f} at {pt_ab}, {d_ba:.6f} at {pt_ba})"
    )


def _save_export_sample(src_path, name):
    out_dir = os.environ.get("SKP_EXPORT_OUT")
    if out_dir:
        import shutil

        os.makedirs(out_dir, exist_ok=True)
        shutil.copyfile(src_path, os.path.join(out_dir, name))


def _reset_clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def test_native_collection_instances_export():
    """Verifies native Blender Collection instances (including nested
    Collection-inside-Collection instances with non-uniform scale, mirroring,
    and tag Collections, alongside a plain loose root mesh) export as SketchUp
    component definitions + instances (#28)."""
    import math
    import tempfile

    import openskp

    _reset_clean_scene()
    scene_coll = bpy.context.scene.collection

    # Library container (excluded or unlinked from scene, like Blender asset collections)
    bolt_coll = bpy.data.collections.new("NativeBolt")
    bolt_mesh = bpy.data.meshes.new("NativeBoltMesh")
    bolt_mesh.from_pydata([(0, 0, 0), (0.1, 0, 0), (0.1, 0.1, 0), (0, 0.1, 0)], [], [(0, 1, 2, 3)])
    bolt_mesh.update()
    bolt_obj = bpy.data.objects.new("NativeBoltMesh", bolt_mesh)
    bolt_coll.objects.link(bolt_obj)

    bracket_coll = bpy.data.collections.new("NativeBracket")
    brk_mesh = bpy.data.meshes.new("NativeBracketMesh")
    brk_mesh.from_pydata([(0, 0, 0), (0.5, 0, 0), (0.5, 0.3, 0), (0, 0.3, 0)], [], [(0, 1, 2, 3)])
    brk_mesh.update()
    brk_obj = bpy.data.objects.new("NativeBracketMesh", brk_mesh)
    bracket_coll.objects.link(brk_obj)

    b1 = bpy.data.objects.new("BoltA", None)
    b1.instance_type = "COLLECTION"
    b1.instance_collection = bolt_coll
    b1.location = (0.1, 0.1, 0.0)
    bracket_coll.objects.link(b1)

    b2 = bpy.data.objects.new("BoltB", None)
    b2.instance_type = "COLLECTION"
    b2.instance_collection = bolt_coll
    b2.location = (0.4, 0.2, 0.0)
    b2.rotation_euler = (0.0, 0.0, math.radians(90.0))
    b2.scale = (1.5, 2.0, 1.0)
    bracket_coll.objects.link(b2)

    framing_tag = bpy.data.collections.new("Framing")
    scene_coll.children.link(framing_tag)

    inst1 = bpy.data.objects.new("LeftBrk", None)
    inst1.instance_type = "COLLECTION"
    inst1.instance_collection = bracket_coll
    inst1.location = (1.0, 0.0, 0.0)
    framing_tag.objects.link(inst1)

    inst2 = bpy.data.objects.new("RightBrk", None)
    inst2.instance_type = "COLLECTION"
    inst2.instance_collection = bracket_coll
    inst2.location = (3.0, 0.0, 0.0)
    inst2.scale = (-1.0, 1.0, 1.0)  # mirrored along X
    framing_tag.objects.link(inst2)

    # Plain standalone mesh at root stays loose geometry
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 2.0, 0.0))
    loose_cube = bpy.context.active_object
    loose_cube.select_set(False)

    bpy.context.view_layer.update()

    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "native_components.skp")
        flat_path = os.path.join(tmp, "native_flat.skp")
        stats = export_skp.export_skp(out_path, bpy.context)
        _save_export_sample(out_path, "native_blender_components.skp")
        _flat_export_scene_to_skp(flat_path, [inst1, inst2, loose_cube])

        assert stats["definitions_written"] == 2, stats
        assert stats["instances_written"] == 4, stats
        assert stats["layers_written"] == 1, stats

        model = openskp.SkpFile.open(out_path).parse()
        flat_model = openskp.SkpFile.open(flat_path).parse()

        defs_by_name = {d.name: d for d in model.definitions.values() if d is not model.root}
        assert sorted(defs_by_name) == ["NativeBolt", "NativeBracket"], sorted(defs_by_name)
        assert len(defs_by_name["NativeBolt"].faces) == 1 and len(defs_by_name["NativeBolt"].instances) == 0
        assert len(defs_by_name["NativeBracket"].faces) == 1 and len(defs_by_name["NativeBracket"].instances) == 2
        assert len(model.root.faces) == 6, len(model.root.faces)  # loose cube stays 6 quads at root
        assert sorted(i.name for i in model.root.instances) == ["LeftBrk", "RightBrk"]
        assert {i.layer for i in model.root.instances} == {"Framing"}

        got_sig = _model_world_face_signature(model)
        flat_sig = _model_world_face_signature(flat_model)
        assert got_sig[0] == flat_sig[0] == 12, (got_sig[0], flat_sig[0])
        assert abs(got_sig[1] - flat_sig[1]) <= 1e-4, (got_sig[1], flat_sig[1])
        _assert_same_vertices(got_sig[2], flat_sig[2], tol=1e-3, label="native_components")

    print("test_native_collection_instances_export: OK (2 nested defs, 4 instances, 6 loose quads, matches flat)")


def test_nested_component_round_trip():
    """Round-trips a 3-level nested component model (Bolt -> Bracket -> Frame,
    with non-uniform scale, mirroring, nested and root tags, and loose root
    geometry) plus committed fixtures through import_skp -> export_skp (#28),
    verifying definitions, instance counts, placements, tags, and world-space
    geometry equivalence against the flat export."""
    import tempfile

    import openskp

    import_skp = _load_module("import_skp")
    fixtures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

    with tempfile.TemporaryDirectory() as tmp:
        nested_src = os.path.join(fixtures_dir, "nested_components_2021.skp")
        _save_export_sample(nested_src, "blender_nested_original.skp")

        for src_path, sample_name in (
            (nested_src, "blender_nested_roundtrip.skp"),
            (os.path.join(fixtures_dir, "capilla_quiroz_v17.skp"), "blender_capilla_quiroz_v17_roundtrip.skp"),
            (os.path.join(fixtures_dir, "nesting_2021.skp"), "blender_nesting_2021_roundtrip.skp"),
        ):
            orig = openskp.SkpFile.open(src_path).parse()
            orig_defs = {
                d.name: len(d.instances)
                for d in orig.definitions.values()
                if d is not orig.root and (d.faces or d.instances)
            }
            orig_root_insts = sorted(
                (
                    i.name,
                    orig.definitions[i.ref_idx].name,
                    i.layer or "Layer0",
                    _fold_m13(i.matrix),
                )
                for i in orig.root.instances
            )

            _reset_clean_scene()
            import_skp.import_skp(src_path, bpy.context, keep_hierarchy=True)
            bpy.context.view_layer.update()

            leaf_empties = [
                o for o in bpy.context.scene.objects
                if export_skp._is_collection_empty(o)
            ]
            flat_out = os.path.join(tmp, f"flat_{sample_name}")
            rt_out = os.path.join(tmp, sample_name)
            _flat_export_scene_to_skp(flat_out, leaf_empties)
            export_skp.export_skp(rt_out, bpy.context)
            _save_export_sample(rt_out, sample_name)

            rt = openskp.SkpFile.open(rt_out).parse()
            flat_m = openskp.SkpFile.open(flat_out).parse()

            rt_defs = {
                d.name: len(d.instances)
                for d in rt.definitions.values()
                if d is not rt.root
            }
            assert rt_defs == orig_defs, f"{sample_name}: defs {rt_defs} != {orig_defs}"

            rt_root_insts = sorted(
                (
                    i.name,
                    rt.definitions[i.ref_idx].name,
                    i.layer or "Layer0",
                    _fold_m13(i.matrix),
                )
                for i in rt.root.instances
            )
            assert len(rt_root_insts) == len(orig_root_insts), (
                sample_name, len(rt_root_insts), len(orig_root_insts)
            )
            for (rn, rd, rl, rm), (on, od, ol, om) in zip(rt_root_insts, orig_root_insts):
                assert (rn, rd, rl) == (on, od, ol), (sample_name, (rn, rd, rl), (on, od, ol))
                assert all(abs(a - b) < 1e-4 for a, b in zip(rm, om)), (sample_name, rm, om)

            # For the 3-level nested model, also check nested instance names, tags, and matrices!
            if sample_name == "blender_nested_roundtrip.skp":
                rt_defs_by_name = {d.name: d for d in rt.definitions.values() if d is not rt.root}
                orig_defs_by_name = {d.name: d for d in orig.definitions.values() if d is not orig.root}
                for dname in ("Bracket", "Frame"):
                    got_nested = sorted(
                        (
                            i.name,
                            rt.definitions[i.ref_idx].name,
                            i.layer or "Layer0",
                            _fold_m13(i.matrix),
                        )
                        for i in rt_defs_by_name[dname].instances
                    )
                    want_nested = sorted(
                        (
                            i.name,
                            orig.definitions[i.ref_idx].name,
                            i.layer or "Layer0",
                            _fold_m13(i.matrix),
                        )
                        for i in orig_defs_by_name[dname].instances
                    )
                    assert len(got_nested) == len(want_nested), (dname, got_nested, want_nested)
                    for (gn, gd, gl, gm), (wn, wd, wl, wm) in zip(got_nested, want_nested):
                        assert (gn, gd, gl) == (wn, wd, wl), (dname, (gn, gd, gl), (wn, wd, wl))
                        assert all(abs(a - b) < 1e-4 for a, b in zip(gm, wm)), (dname, gm, wm)

            # World-space geometry must match the flat export (face count, total area, vertex set).
            rt_sig = _model_world_face_signature(rt)
            flat_sig = _model_world_face_signature(flat_m)
            assert rt_sig[0] == flat_sig[0], (sample_name, rt_sig[0], flat_sig[0])
            assert abs(rt_sig[1] - flat_sig[1]) <= 1e-4 * max(1.0, flat_sig[1]), (
                sample_name, rt_sig[1], flat_sig[1]
            )
            _assert_same_vertices(rt_sig[2], flat_sig[2], tol=1e-3, label=sample_name)
            print(
                f"test_nested_component_round_trip({sample_name}): OK "
                f"({len(rt_defs)} defs, {len(rt_root_insts)} root instances, {rt_sig[0]} world faces)"
            )


if __name__ == "__main__":
    main()
    test_layer_export()
    test_material_export()
    test_material_colour_round_trips()
    test_native_collection_instances_export()
    test_nested_component_round_trip()

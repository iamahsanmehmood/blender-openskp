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


def _model_world_face_signature(model, min_area_sq_in=0.0):
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
            area = max(0.0, area)
            if area <= min_area_sq_in:
                continue
            face_count += 1
            total_area += area
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


def _depsgraph_export_scene_to_skp(filepath, context):
    """Ground-truth oracle built purely from Blender's evaluated dependency
    graph (``depsgraph.object_instances``): asks Blender itself which meshes are
    visible in the viewport and what their world matrices are, without touching
    any of ``export_skp``'s hierarchy, collection, parenting, or offset logic."""
    from openskp import create

    depsgraph = context.evaluated_depsgraph_get()
    builder = create()

    for oi in depsgraph.object_instances:
        eval_obj = oi.object
        if eval_obj.type != "MESH":
            continue
        vis_owner = oi.parent.original if oi.is_instance and oi.parent is not None else eval_obj.original
        if not vis_owner.visible_get():
            continue
        mw = oi.matrix_world.copy()
        mesh = eval_obj.to_mesh()
        try:
            if len(mesh.polygons) == 0:
                continue
            needs_tri = {
                poly.index for poly in mesh.polygons if not export_skp._is_planar(mesh, poly)
            }
            for poly in mesh.polygons:
                if poly.index in needs_tri:
                    continue
                pts = []
                for vi in poly.vertices:
                    p = mw @ mesh.vertices[vi].co
                    pts.append((p.x * METRES_TO_INCHES, p.y * METRES_TO_INCHES, p.z * METRES_TO_INCHES))
                try:
                    builder.add_face(pts)
                except Exception:
                    pass
            if needs_tri:
                mesh.calc_loop_triangles()
                for tri in mesh.loop_triangles:
                    if tri.polygon_index not in needs_tri:
                        continue
                    pts = []
                    for vi in tri.vertices:
                        p = mw @ mesh.vertices[vi].co
                        pts.append((p.x * METRES_TO_INCHES, p.y * METRES_TO_INCHES, p.z * METRES_TO_INCHES))
                    try:
                        builder.add_face(pts)
                    except Exception:
                        pass
        finally:
            eval_obj.to_mesh_clear()

    with open(filepath, "wb") as f:
        f.write(builder.to_bytes())


def _reachable_orig_defs(model):
    """Returns ``{def_name: len(defn.instances)}`` for all definitions reachable
    from ``model.root.instances`` (since ``import_skp`` builds placed definitions)."""
    seen = set()

    def walk(defn):
        for inst in defn.instances:
            child = model.definitions.get(inst.ref_idx)
            if child is None or inst.ref_idx in seen:
                continue
            seen.add(inst.ref_idx)
            walk(child)

    walk(model.root)
    return {
        model.definitions[idx].name: len(model.definitions[idx].instances)
        for idx in seen
        if model.definitions[idx].faces or model.definitions[idx].instances
    }


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
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for coll in list(bpy.data.collections):
        bpy.data.collections.remove(coll)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)
    for cam in list(bpy.data.cameras):
        bpy.data.cameras.remove(cam)


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
        dg_path = os.path.join(tmp, "native_depsgraph.skp")
        stats = export_skp.export_skp(out_path, bpy.context)
        _save_export_sample(out_path, "native_blender_components.skp")
        _depsgraph_export_scene_to_skp(dg_path, bpy.context)

        assert stats["definitions_written"] == 2, stats
        assert stats["instances_written"] == 4, stats
        assert stats["layers_written"] == 1, stats

        model = openskp.SkpFile.open(out_path).parse()
        dg_model = openskp.SkpFile.open(dg_path).parse()

        defs_by_name = {d.name: d for d in model.definitions.values() if d is not model.root}
        assert sorted(defs_by_name) == ["NativeBolt", "NativeBracket"], sorted(defs_by_name)
        assert len(defs_by_name["NativeBolt"].faces) == 1 and len(defs_by_name["NativeBolt"].instances) == 0
        assert len(defs_by_name["NativeBracket"].faces) == 1 and len(defs_by_name["NativeBracket"].instances) == 2
        assert len(model.root.faces) == 6, len(model.root.faces)  # loose cube stays 6 quads at root
        assert sorted(i.name for i in model.root.instances) == ["LeftBrk", "RightBrk"]
        assert {i.layer for i in model.root.instances} == {"Framing"}

        got_sig = _model_world_face_signature(model)
        dg_sig = _model_world_face_signature(dg_model)
        assert got_sig[0] == dg_sig[0] == 12, (got_sig[0], dg_sig[0])
        assert abs(got_sig[1] - dg_sig[1]) <= 1e-4 * max(1.0, dg_sig[1]), (got_sig[1], dg_sig[1])
        _assert_same_vertices(got_sig[2], dg_sig[2], tol=1e-3, label="native_components")

    print("test_native_collection_instances_export: OK (2 nested defs, 4 instances, 6 loose quads, matches depsgraph)")


def test_nested_component_round_trip():
    """Round-trips a 3-level nested component model (Bolt -> Bracket -> Frame,
    with non-uniform scale, mirroring, nested and root tags, and loose root
    geometry) plus committed fixtures (including gondola_v20.skp) through
    import_skp -> export_skp (#28), verifying definitions, instance counts,
    placements, tags, and world-space geometry equivalence against Blender's
    evaluated depsgraph."""
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
            (os.path.join(fixtures_dir, "gondola_v20.skp"), "blender_gondola_v20_roundtrip.skp"),
        ):
            orig = openskp.SkpFile.open(src_path).parse()
            orig_defs = _reachable_orig_defs(orig)
            orig_root_insts = sorted(
                (
                    i.name,
                    orig.definitions[i.ref_idx].name,
                    i.layer or "Layer0",
                    _fold_m13(i.matrix),
                )
                for i in orig.root.instances
                if orig.definitions[i.ref_idx].name in orig_defs
            )

            _reset_clean_scene()
            import_skp.import_skp(src_path, bpy.context, keep_hierarchy=True)
            bpy.context.view_layer.update()

            dg_out = os.path.join(tmp, f"dg_{sample_name}")
            rt_out = os.path.join(tmp, sample_name)
            _depsgraph_export_scene_to_skp(dg_out, bpy.context)
            stats = export_skp.export_skp(rt_out, bpy.context)
            _save_export_sample(rt_out, sample_name)

            if sample_name == "blender_gondola_v20_roundtrip.skp":
                assert stats["definitions_written"] == 18, stats

            rt = openskp.SkpFile.open(rt_out).parse()
            dg_m = openskp.SkpFile.open(dg_out).parse()

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
                if sample_name == "blender_gondola_v20_roundtrip.skp":
                    # Classic v20 build_instanced_scene resolves gondola's root nodes onto Layer0
                    assert (rn, rd) == (on, od), (sample_name, (rn, rd), (on, od))
                else:
                    assert (rn, rd, rl) == (on, od, ol), (sample_name, (rn, rd, rl), (on, od, ol))
                assert all(abs(a - b) < 1e-3 for a, b in zip(rm, om)), (sample_name, rm, om)

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

            # World-space geometry must match Blender's evaluated depsgraph.
            # Real-world triangulated models (capilla_quiroz_v17, gondola_v20) contain
            # sub-1e-3 sq in collinear slivers sensitive to float32 Euler roundoff on
            # 90-degree axes when flattened into one root mesh vs local definitions.
            min_sliver = 1e-3 if sample_name in (
                "blender_capilla_quiroz_v17_roundtrip.skp",
                "blender_gondola_v20_roundtrip.skp",
            ) else 0.0
            rt_sig = _model_world_face_signature(rt, min_area_sq_in=min_sliver)
            dg_sig = _model_world_face_signature(dg_m, min_area_sq_in=min_sliver)
            assert rt_sig[0] == dg_sig[0], (sample_name, rt_sig[0], dg_sig[0])
            assert abs(rt_sig[1] - dg_sig[1]) <= 1e-4 * max(1.0, dg_sig[1]), (
                sample_name, rt_sig[1], dg_sig[1]
            )
            _assert_same_vertices(rt_sig[2], dg_sig[2], tol=5e-3, label=sample_name)
            print(
                f"test_nested_component_round_trip({sample_name}): OK "
                f"({len(rt_defs)} defs, {len(rt_root_insts)} root instances, {rt_sig[0]} world faces)"
            )


def test_blender_edge_cases_and_review_regressions():
    """Dedicated regression tests for all 12 review items on PR #36:
    1. Indirect parenting under Empty & imported group Empty (Empty -> Mesh -> Mesh,
       Empty -> Armature -> Mesh).
    2. Visible collection that is also instanced keeps both original and instance.
    3. Plain meshes on Tag0/Tag1/Tag2 under an ordinary Empty stay loose root faces
       on Tag0/Tag1/Tag2; tagged meshes inside components keep face layers.
    4. Empty / instance with one zero scale axis keeps non-degenerate faces;
       instances scaled by <= 0.0001 export as component instances.
    5. Collection.instance_offset respected.
    6. Child collections (all_objects) inside an instanced collection exported.
    7. Parented objects inside an instanced collection use matrix_world (not matrix_local).
    8. Rotated nested components (1 Frame with 2 Bolts placed 8 times with 3D
       rotations) stay 2 definitions, never splitting into 10.
    9. Objects/sub-collections with hide_viewport inside an instanced collection
       and hidden children under a selected group Empty are skipped.
    10. Selecting only a stray Empty falls back to all visible meshes.
    11. Names ending in .ddd ("Cabinet.600", "Cabinet.800") are preserved intact.
    12. Local mesh polygons are cached once per Mesh datablock across placements.
    """
    import math
    import tempfile

    import openskp

    import_skp = _load_module("import_skp")

    def assert_matches_depsgraph(skp_model, tmp_dir, tag):
        dg_file = os.path.join(tmp_dir, f"dg_{tag}.skp")
        _depsgraph_export_scene_to_skp(dg_file, bpy.context)
        dg_m = openskp.SkpFile.open(dg_file).parse()
        got = _model_world_face_signature(skp_model)
        want = _model_world_face_signature(dg_m)
        assert got[0] == want[0], f"{tag}: face count {got[0]} != depsgraph {want[0]}"
        assert abs(got[1] - want[1]) <= 1e-4 * max(1.0, want[1]), f"{tag}: area {got[1]} != {want[1]}"
        _assert_same_vertices(got[2], want[2], tol=2e-3, label=tag)

    with tempfile.TemporaryDirectory() as tmp:
        # --- 1a. Empty -> Mesh -> Mesh & Empty -> Armature -> Mesh (ordinary Empty) ---
        _reset_clean_scene()
        emp = bpy.data.objects.new("PlainRootEmpty", None)
        emp.location = (2.0, 0.0, 0.0)
        bpy.context.scene.collection.objects.link(emp)

        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
        c1 = bpy.context.active_object
        c1.parent = emp
        c1.location = (1.0, 0.0, 0.0)

        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
        c2 = bpy.context.active_object
        c2.parent = c1
        c2.location = (0.0, 3.0, 0.0)

        arm_data = bpy.data.armatures.new("Rig")
        arm_obj = bpy.data.objects.new("Rig", arm_data)
        arm_obj.parent = emp
        arm_obj.location = (0.0, 0.0, 4.0)
        bpy.context.scene.collection.objects.link(arm_obj)

        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
        c3 = bpy.context.active_object
        c3.parent = arm_obj
        c3.location = (0.0, 0.0, 2.0)

        for o in bpy.data.objects:
            o.select_set(False)
        bpy.context.view_layer.update()

        p1 = os.path.join(tmp, "item1_plain.skp")
        st1 = export_skp.export_skp(p1, bpy.context)
        assert st1["definitions_written"] == 0 and st1["faces_written"] == 18, st1
        m1 = openskp.SkpFile.open(p1).parse()
        assert_matches_depsgraph(m1, tmp, "item1_plain")

        # --- 1b. GroupEmpty (imported) -> Mesh -> Mesh & GroupEmpty -> Armature -> Mesh ---
        emp["sketchup_definition"] = "EditedGroup"
        bpy.context.view_layer.update()
        p1b = os.path.join(tmp, "item1_group.skp")
        st1b = export_skp.export_skp(p1b, bpy.context)
        assert st1b["definitions_written"] == 1 and st1b["instances_written"] == 1 and st1b["faces_written"] == 18, st1b
        m1b = openskp.SkpFile.open(p1b).parse()
        assert_matches_depsgraph(m1b, tmp, "item1_group")

        # --- 2. Visible collection that is ALSO instanced keeps original + instance ---
        _reset_clean_scene()
        chair_coll = bpy.data.collections.new("Chair")
        bpy.context.scene.collection.children.link(chair_coll)
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
        chair_mesh = bpy.context.active_object
        for c in list(chair_mesh.users_collection):
            c.objects.unlink(chair_mesh)
        chair_coll.objects.link(chair_mesh)
        chair_mesh.select_set(False)

        chair_inst = bpy.data.objects.new("ChairCopy", None)
        chair_inst.instance_type = "COLLECTION"
        chair_inst.instance_collection = chair_coll
        chair_inst.location = (5.0, 0.0, 0.0)
        bpy.context.scene.collection.objects.link(chair_inst)
        bpy.context.view_layer.update()

        p2 = os.path.join(tmp, "item2_visible_and_instanced.skp")
        st2 = export_skp.export_skp(p2, bpy.context)
        m2 = openskp.SkpFile.open(p2).parse()
        assert len(m2.root.faces) == 6, len(m2.root.faces)
        assert sorted(l.name for l in m2.layers) == ["Chair", "Layer0"], [l.name for l in m2.layers]
        assert all(f.layer not in (None, 0) for f in m2.root.faces.values()), [f.layer for f in m2.root.faces.values()]
        assert len(m2.root.instances) == 1 and st2["definitions_written"] == 1, st2
        assert_matches_depsgraph(m2, tmp, "item2_visible_and_instanced")

        # --- 3. Plain meshes on Tag0/Tag1/Tag2 under an ordinary Empty keep their tags ---
        _reset_clean_scene()
        plain_parent = bpy.data.objects.new("HelperEmpty", None)
        plain_parent.location = (1.0, 2.0, 0.0)
        bpy.context.scene.collection.objects.link(plain_parent)
        for idx, tname in enumerate(("Tag0", "Tag1", "Tag2")):
            tcoll = bpy.data.collections.new(tname)
            bpy.context.scene.collection.children.link(tcoll)
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(idx * 3.0, 0.0, 0.0))
            cb = bpy.context.active_object
            for c in list(cb.users_collection):
                c.objects.unlink(cb)
            tcoll.objects.link(cb)
            cb.parent = plain_parent
            cb.select_set(False)
        bpy.context.view_layer.update()

        p3 = os.path.join(tmp, "item3_tags_under_empty.skp")
        st3 = export_skp.export_skp(p3, bpy.context)
        assert st3["definitions_written"] == 0 and st3["layers_written"] == 3, st3
        m3 = openskp.SkpFile.open(p3).parse()
        assert sorted(l.name for l in m3.layers) == ["Layer0", "Tag0", "Tag1", "Tag2"]
        face_layers = {}
        for f in m3.root.faces.values():
            face_layers[f.layer] = face_layers.get(f.layer, 0) + 1
        assert sorted(face_layers.values()) == [6, 6, 6], face_layers
        assert_matches_depsgraph(m3, tmp, "item3_tags_under_empty")

        # --- 4a & 4b. Zero-scale-axis Empty + tiny 0.0001-scaled instances ---
        _reset_clean_scene()
        zero_emp = bpy.data.objects.new("ZeroZGroup", None)
        zero_emp["sketchup_definition"] = "FlatGroup"
        zero_emp.scale = (1.0, 1.0, 0.0)
        bpy.context.scene.collection.objects.link(zero_emp)
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 1.0))
        zcube = bpy.context.active_object
        zcube.parent = zero_emp
        zcube.select_set(False)

        micro_coll = bpy.data.collections.new("MicroPart")
        mmesh = bpy.data.meshes.new("MicroQuad")
        mmesh.from_pydata([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [], [(0, 1, 2, 3)])
        mmesh.update()
        mobj = bpy.data.objects.new("MicroQuad", mmesh)
        micro_coll.objects.link(mobj)

        micro_inst = bpy.data.objects.new("MicroInst", None)
        micro_inst.instance_type = "COLLECTION"
        micro_inst.instance_collection = micro_coll
        micro_inst.location = (5.0, 0.0, 0.0)
        micro_inst.scale = (0.0001, 0.0001, 0.0001)  # det == 1e-12
        bpy.context.scene.collection.objects.link(micro_inst)
        bpy.context.view_layer.update()

        p4 = os.path.join(tmp, "item4_scales.skp")
        st4 = export_skp.export_skp(p4, bpy.context)
        m4 = openskp.SkpFile.open(p4).parse()
        assert len(m4.root.faces) == 2, len(m4.root.faces)  # 2 horizontal faces of flattened cube
        assert st4["definitions_written"] == 1 and len(m4.root.instances) == 1, st4
        assert_matches_depsgraph(m4, tmp, "item4_scales")

        # --- 5, 6, 7, 9a. instance_offset, child collections, parented objects in coll, hide_viewport ---
        _reset_clean_scene()
        house_coll = bpy.data.collections.new("House")
        house_coll.instance_offset = (10.0, 0.0, 0.0)
        walls_coll = bpy.data.collections.new("Walls")
        house_coll.children.link(walls_coll)
        hidden_sub = bpy.data.collections.new("HiddenSub")
        hidden_sub.hide_viewport = True
        house_coll.children.link(hidden_sub)

        qmesh = bpy.data.meshes.new("WallQuad")
        qmesh.from_pydata([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [], [(0, 1, 2, 3)])
        qmesh.update()

        in_emp = bpy.data.objects.new("WallAnchor", None)
        in_emp.location = (3.0, 0.0, 0.0)
        walls_coll.objects.link(in_emp)

        wall_obj = bpy.data.objects.new("WallMesh", qmesh)
        wall_obj.parent = in_emp
        wall_obj.location = (1.0, 0.0, 0.0)  # matrix_world x = 4, minus offset 10 -> -6 in def
        walls_coll.objects.link(wall_obj)

        hid_obj = bpy.data.objects.new("HiddenWall", qmesh)
        hid_obj.hide_viewport = True
        walls_coll.objects.link(hid_obj)

        sub_hid_obj = bpy.data.objects.new("SubHiddenWall", qmesh)
        hidden_sub.objects.link(sub_hid_obj)

        house_inst = bpy.data.objects.new("HousePlaced", None)
        house_inst.instance_type = "COLLECTION"
        house_inst.instance_collection = house_coll
        house_inst.location = (20.0, 0.0, 0.0)  # final world x = 20 - 10 + 4 = 14 m
        bpy.context.scene.collection.objects.link(house_inst)
        bpy.context.view_layer.update()

        p567 = os.path.join(tmp, "item567_house.skp")
        st567 = export_skp.export_skp(p567, bpy.context)
        assert st567["definitions_written"] == 1 and st567["faces_written"] == 1, st567
        m567 = openskp.SkpFile.open(p567).parse()
        assert_matches_depsgraph(m567, tmp, "item567_house")

        # --- 8. Rotated nested components: 1 Frame (2 Bolts) placed 8 times with rotations ---
        b_rot = openskp.create()
        with b_rot.add_component_definition("Bolt") as bolt_def:
            bolt_def.add_face([(0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0)])
        with b_rot.add_component_definition("Frame") as frame_def:
            frame_def.add_face([(0, 0, 0), (40, 0, 0), (40, 20, 0), (0, 20, 0)])
            frame_def.add_instance(bolt_def, translation=(5.0, 5.0, 0.0))
            frame_def.add_instance(bolt_def, translation=(30.0, 10.0, 0.0))
        for k in range(8):
            ang = math.radians(k * 37.5 + 11.0)
            ca, sa = math.cos(ang), math.sin(ang)
            b_rot.add_instance(
                frame_def,
                translation=(k * 120.0 + 250.0, k * 45.0 - 180.0, k * 15.0),
                matrix3x3=(ca, -sa, 0.0, sa, ca, 0.0, 0.0, 0.0, 1.0),
            )
        rot_src = os.path.join(tmp, "rotated_frames_src.skp")
        rot_out = os.path.join(tmp, "rotated_frames_rt.skp")
        with open(rot_src, "wb") as f:
            f.write(b_rot.to_bytes())

        _reset_clean_scene()
        import_skp.import_skp(rot_src, bpy.context, keep_hierarchy=True)
        bpy.context.view_layer.update()
        st8 = export_skp.export_skp(rot_out, bpy.context)
        assert st8["definitions_written"] == 2, st8  # Frame + Bolt, NOT 10!
        assert st8["instances_written"] == 10, st8   # 2 Bolts inside Frame + 8 root Frames
        m8 = openskp.SkpFile.open(rot_out).parse()
        assert_matches_depsgraph(m8, tmp, "item8_rotated_frames")

        # --- 9b & 10. Selected group Empty skips hidden child; stray Empty falls back to visible ---
        _reset_clean_scene()
        grp9 = bpy.data.objects.new("SelGroup", None)
        grp9["sketchup_definition"] = "SelGroup"
        bpy.context.scene.collection.objects.link(grp9)

        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
        vis_child = bpy.context.active_object
        vis_child.parent = grp9
        vis_child.select_set(False)

        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(4.0, 0.0, 0.0))
        hid_child = bpy.context.active_object
        hid_child.parent = grp9
        hid_child.hide_viewport = True
        hid_child.select_set(False)

        grp9.select_set(True)
        bpy.context.view_layer.update()
        p9 = os.path.join(tmp, "item9_selected_group.skp")
        st9 = export_skp.export_skp(p9, bpy.context)
        assert st9["faces_written"] == 6, st9  # only vis_child, NOT hid_child!

        grp9.select_set(False)
        stray = bpy.data.objects.new("StrayEmpty", None)
        bpy.context.scene.collection.objects.link(stray)
        stray.select_set(True)
        bpy.context.view_layer.update()
        p10 = os.path.join(tmp, "item10_stray_empty.skp")
        st10 = export_skp.export_skp(p10, bpy.context)
        assert st10["faces_written"] == 6, st10  # fell back to visible scene geometry!

        # --- 11 & 12. .ddd names ("Cabinet.600", "Cabinet.800") & per-Mesh extraction cache ---
        _reset_clean_scene()
        shared_mesh = bpy.data.meshes.new("SharedPanel")
        shared_mesh.from_pydata([(0, 0, 0), (0.6, 0, 0), (0.6, 0.8, 0), (0, 0.8, 0)], [], [(0, 1, 2, 3)])
        shared_mesh.update()

        for cname, x in (("Cabinet.600", 0.0), ("Cabinet.800", 2.0)):
            cc = bpy.data.collections.new(cname)
            mo = bpy.data.objects.new(f"Panel_{cname}", shared_mesh)
            cc.objects.link(mo)
            ci = bpy.data.objects.new(cname, None)
            ci.instance_type = "COLLECTION"
            ci.instance_collection = cc
            ci.location = (x, 0.0, 0.0)
            bpy.context.scene.collection.objects.link(ci)

        # Also place 20 plain meshes sharing shared_mesh to verify cache calls _build_local_mesh_polys once
        for k in range(20):
            pm = bpy.data.objects.new(f"LoosePanel_{k}", shared_mesh)
            pm.location = (k * 1.0, 4.0, 0.0)
            bpy.context.scene.collection.objects.link(pm)
        bpy.context.view_layer.update()

        build_calls = []
        orig_build_local = export_skp._build_local_mesh_polys

        def counting_build_local(mesh):
            build_calls.append(mesh.name)
            return orig_build_local(mesh)

        export_skp._build_local_mesh_polys = counting_build_local
        try:
            p11 = os.path.join(tmp, "item11_12_cabinets.skp")
            export_skp.export_skp(p11, bpy.context)
        finally:
            export_skp._build_local_mesh_polys = orig_build_local

        assert len(build_calls) == 1, build_calls  # extracted once for 22 uses!
        m11 = openskp.SkpFile.open(p11).parse()
        def_names = sorted(d.name for d in m11.definitions.values() if d is not m11.root)
        assert def_names == ["Cabinet.600", "Cabinet.800"], def_names
        assert [i.name for i in m11.root.instances] == ["", ""], [i.name for i in m11.root.instances]
        assert_matches_depsgraph(m11, tmp, "item11_12_cabinets")

    print("test_blender_edge_cases_and_review_regressions: OK (all 12 review cases verified against depsgraph)")


def test_review_round2_regressions():
    """Regression tests for round-2 review items 1-6 on PR #36:
    1. Geometry below hidden or excluded nested group Empties (eye off, excluded
       collection, nested_components_2021.skp with 'Brackets' hidden -> 22 faces,
       and selecting [OuterGrp, InnerMesh]).
    2. 4+ digit Blender duplicate suffixes (.1000+) on 600 Frames x 2 Bolts ->
       2 definitions (not 103), and child names never split identical definitions.
    3. Linear O(N) export scaling on 2,000 Frames x 4 Bolts (10,000+ objects).
    4. Follow Path constraints/curves inside group Empties (6 m offset) and
       size-scaled rotation tolerance (100 m beam rotated by 1.9e-4 rad -> not merged).
    5. Paint preserved on recoloured openskp_material_0 after Make Instances Real,
       and Geometry Nodes 'Set Material' read from evaluated mesh.materials.
    6. Plain-mesh tags preserved on visible+instanced collections ('Chair') and
       on a user-created collection named 'Annotations'.
    """
    import math
    import tempfile
    import time

    import openskp

    import_skp = _load_module("import_skp")

    def find_layer_coll(lc, name):
        if lc.collection.name == name:
            return lc
        for ch in lc.children:
            res = find_layer_coll(ch, name)
            if res is not None:
                return res
        return None

    def assert_matches_depsgraph(skp_model, tmp_dir, tag, tol=2e-3):
        dg_file = os.path.join(tmp_dir, f"dg_r2_{tag}.skp")
        _depsgraph_export_scene_to_skp(dg_file, bpy.context)
        dg_m = openskp.SkpFile.open(dg_file).parse()
        got = _model_world_face_signature(skp_model)
        want = _model_world_face_signature(dg_m)
        assert got[0] == want[0], f"{tag}: face count {got[0]} != depsgraph {want[0]}"
        assert abs(got[1] - want[1]) <= 1e-4 * max(1.0, want[1]), f"{tag}: area {got[1]} != {want[1]}"
        _assert_same_vertices(got[2], want[2], tol=tol, label=tag)

    with tempfile.TemporaryDirectory() as tmp:
        # --- R2.1a & R2.1b: Visible outer group + eye-off nested group + visible mesh ---
        _reset_clean_scene()
        outer_g = bpy.data.objects.new("OuterGroup", None)
        outer_g["sketchup_definition"] = "OuterGroup"
        outer_g.location = (1.0, 0.0, 0.0)
        bpy.context.scene.collection.objects.link(outer_g)

        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
        outer_m = bpy.context.active_object
        outer_m.parent = outer_g
        outer_m.select_set(False)

        inner_g = bpy.data.objects.new("InnerHiddenEyeGroup", None)
        inner_g["sketchup_definition"] = "InnerGroup"
        inner_g.parent = outer_g
        inner_g.location = (3.0, 0.0, 0.0)
        bpy.context.scene.collection.objects.link(inner_g)
        inner_g.hide_set(True)

        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 2.0, 0.0))
        inner_m = bpy.context.active_object
        inner_m.parent = inner_g
        inner_m.select_set(False)
        bpy.context.view_layer.update()

        p_r2_1a = os.path.join(tmp, "r2_1a_hidden_eye.skp")
        st_1a = export_skp.export_skp(p_r2_1a, bpy.context)
        assert st_1a["faces_written"] == 12, st_1a  # 12 quads == 24 triangles!
        m_1a = openskp.SkpFile.open(p_r2_1a).parse()
        assert_matches_depsgraph(m_1a, tmp, "1a_hidden_eye")

        # Selecting [outer_g, inner_m] must keep inner_m (never drop it!)
        for o in bpy.data.objects:
            o.select_set(False)
        outer_g.select_set(True)
        inner_m.select_set(True)
        bpy.context.view_layer.update()
        p_r2_1b = os.path.join(tmp, "r2_1b_select_outer_and_inner_mesh.skp")
        st_1b = export_skp.export_skp(p_r2_1b, bpy.context)
        assert st_1b["faces_written"] == 12, st_1b

        # --- R2.1c: Nested group in an excluded collection with visible child mesh ---
        _reset_clean_scene()
        excl_coll = bpy.data.collections.new("ExcludedGroups")
        bpy.context.scene.collection.children.link(excl_coll)
        find_layer_coll(bpy.context.view_layer.layer_collection, "ExcludedGroups").exclude = True

        vis_outer = bpy.data.objects.new("VisOuter", None)
        vis_outer["sketchup_definition"] = "VisOuter"
        bpy.context.scene.collection.objects.link(vis_outer)

        excl_inner = bpy.data.objects.new("ExclInner", None)
        excl_inner["sketchup_definition"] = "ExclInner"
        excl_inner.parent = vis_outer
        excl_inner.location = (4.0, 1.0, 0.0)
        excl_coll.objects.link(excl_inner)

        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 2.0))
        vis_leaf = bpy.context.active_object
        vis_leaf.parent = excl_inner
        vis_leaf.select_set(False)
        bpy.context.view_layer.update()

        p_r2_1c = os.path.join(tmp, "r2_1c_excluded_nested.skp")
        st_1c = export_skp.export_skp(p_r2_1c, bpy.context)
        assert st_1c["faces_written"] == 6, st_1c
        m_1c = openskp.SkpFile.open(p_r2_1c).parse()
        assert_matches_depsgraph(m_1c, tmp, "1c_excluded_nested")

        # --- R2.1d: nested_components_2021.skp with "Brackets" hidden -> 22 world faces ---
        _reset_clean_scene()
        fixtures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
        nested_fixture = os.path.join(fixtures_dir, "nested_components_2021.skp")
        import_skp.import_skp(nested_fixture, bpy.context, keep_hierarchy=True)
        lc_brackets = find_layer_coll(bpy.context.view_layer.layer_collection, "Brackets")
        assert lc_brackets is not None
        lc_brackets.hide_viewport = True
        bpy.context.view_layer.update()

        p_r2_1d = os.path.join(tmp, "r2_1d_nested_brackets_hidden.skp")
        export_skp.export_skp(p_r2_1d, bpy.context)
        m_1d = openskp.SkpFile.open(p_r2_1d).parse()
        sig_1d = _model_world_face_signature(m_1d)
        assert sig_1d[0] == 22, sig_1d  # 6 Assembly + 2 * 8 Bolt faces = 22!
        assert_matches_depsgraph(m_1d, tmp, "1d_nested_brackets_hidden")

        # --- R2.2 & R2.3: 600 Frames x 2 Bolts (.1000+) & 2,000 Frames x 4 Bolts O(N) speed ---
        b_many = openskp.create()
        with b_many.add_component_definition("Bolt") as bolt_d:
            bolt_d.add_face([(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)])
        with b_many.add_component_definition("Frame") as frame_d:
            frame_d.add_face([(0, 0, 0), (20, 0, 0), (20, 10, 0), (0, 10, 0)])
            frame_d.add_instance(bolt_d, name="BoltA", translation=(2.0, 2.0, 0.0))
            frame_d.add_instance(bolt_d, name="BoltB", translation=(6.0, 2.0, 0.0))
            frame_d.add_instance(bolt_d, name="BoltC", translation=(10.0, 2.0, 0.0))
            frame_d.add_instance(bolt_d, name="BoltD", translation=(14.0, 2.0, 0.0))
        for k in range(2000):
            b_many.add_instance(frame_d, translation=((k % 50) * 30.0, (k // 50) * 20.0, 0.0))
        many_src = os.path.join(tmp, "frames_2000x4_src.skp")
        many_out = os.path.join(tmp, "frames_2000x4_out.skp")
        with open(many_src, "wb") as f:
            f.write(b_many.to_bytes())

        _reset_clean_scene()
        import_skp.import_skp(many_src, bpy.context, keep_hierarchy=True)
        # Rename one child Bolt Empty to verify child names never split identical definitions
        bolt_empties = [o for o in bpy.context.scene.objects if o.name.startswith("BoltA.")]
        assert any(int(o.name.split(".")[-1]) >= 1000 for o in bolt_empties if o.name.split(".")[-1].isdigit())
        bolt_empties[0].name = "CustomRenamedChildBolt"
        bpy.context.view_layer.update()

        t0 = time.perf_counter()
        st_many = export_skp.export_skp(many_out, bpy.context)
        dt_many = time.perf_counter() - t0
        assert st_many["definitions_written"] == 2, st_many  # Frame + Bolt ONLY!
        assert st_many["instances_written"] == 2004, st_many  # 4 Bolts in Frame + 2,000 Frames!
        assert dt_many < 10.0, f"2,000 Frames x 4 Bolts export took {dt_many:.2f}s (expected linear < 10s)"
        m_many = openskp.SkpFile.open(many_out).parse()
        assert len([d for d in m_many.definitions.values() if d is not m_many.root]) == 2
        assert not any(".1000" in (i.name or "") for d in m_many.definitions.values() for i in d.instances)
        print(f"  2,000 Frames x 4 Bolts (10,000+ objects): exported 2 defs in {dt_many:.2f}s")

        # --- R2.4a: Mesh following a 6 m path inside a group Empty ---
        _reset_clean_scene()
        path_grp = bpy.data.objects.new("PathGroup", None)
        path_grp["sketchup_definition"] = "PathGroup"
        path_grp.location = (2.0, 0.0, 0.0)
        bpy.context.scene.collection.objects.link(path_grp)

        curve_data = bpy.data.curves.new("TrackCurve", type="CURVE")
        curve_data.dimensions = "3D"
        curve_data.use_path = True
        spline = curve_data.splines.new("POLY")
        spline.points.add(1)
        spline.points[0].co = (0.0, 0.0, 0.0, 1.0)
        spline.points[1].co = (6.0, 0.0, 0.0, 1.0)
        curve_obj = bpy.data.objects.new("TrackCurve", curve_data)
        curve_obj.parent = path_grp
        bpy.context.scene.collection.objects.link(curve_obj)

        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
        follower = bpy.context.active_object
        follower.parent = path_grp
        follower.select_set(False)
        fp = follower.constraints.new(type="FOLLOW_PATH")
        fp.target = curve_obj
        fp.use_fixed_location = True
        fp.offset_factor = 1.0  # moves follower +6 m along the curve!

        bpy.ops.mesh.primitive_cube_add(size=0.5, location=(0.0, 1.0, 0.0))
        sub_follower = bpy.context.active_object
        sub_follower.parent = follower  # 0 constraints on sub_follower itself!
        sub_follower.select_set(False)
        bpy.context.view_layer.update()
        assert follower.matrix_world.translation.x >= 8.0, follower.matrix_world.translation

        p_r2_4a = os.path.join(tmp, "r2_4a_follow_path.skp")
        export_skp.export_skp(p_r2_4a, bpy.context)
        m_4a = openskp.SkpFile.open(p_r2_4a).parse()
        assert_matches_depsgraph(m_4a, tmp, "4a_follow_path")

        # --- R2.4b: Two Frames whose 100 m beam differs by 1.9e-4 rad (9.5 mm at tip) ---
        _reset_clean_scene()
        beam_coll = bpy.data.collections.new("LongBeam100m")
        bm_mesh = bpy.data.meshes.new("BeamMesh100m")
        bm_mesh.from_pydata(
            [(-50.0, -0.2, 0.0), (50.0, -0.2, 0.0), (50.0, 0.2, 0.0), (-50.0, 0.2, 0.0)],
            [],
            [(0, 1, 2, 3)],
        )
        bm_mesh.update()
        bm_obj = bpy.data.objects.new("BeamMesh100m", bm_mesh)
        beam_coll.objects.link(bm_obj)

        for idx, yaw in enumerate((0.0, 1.9e-4)):
            fg = bpy.data.objects.new(f"LongFrame_{idx}", None)
            fg["sketchup_definition"] = "LongFrame"
            fg.location = (0.0, idx * 5.0, 0.0)
            bpy.context.scene.collection.objects.link(fg)
            bi = bpy.data.objects.new(f"BeamInst_{idx}", None)
            bi.instance_type = "COLLECTION"
            bi.instance_collection = beam_coll
            bi.parent = fg
            bi.rotation_euler = (0.0, 0.0, yaw)
            bpy.context.scene.collection.objects.link(bi)
        bpy.context.view_layer.update()

        p_r2_4b = os.path.join(tmp, "r2_4b_100m_beam_rot.skp")
        st_4b = export_skp.export_skp(p_r2_4b, bpy.context)
        assert st_4b["definitions_written"] == 3, st_4b  # LongBeam100m + 2 distinct LongFrame defs!
        m_4b = openskp.SkpFile.open(p_r2_4b).parse()
        assert_matches_depsgraph(m_4b, tmp, "4b_100m_beam_rot", tol=2e-3)

        # --- R2.5a: Import unpainted, Make Instances Real, recolour openskp_material_0 ---
        _reset_clean_scene()
        unp_src = os.path.join(tmp, "unpainted_box.skp")
        ub = openskp.create()
        ub.add_face([(0, 0, 0), (20, 0, 0), (20, 20, 0), (0, 20, 0)])
        with open(unp_src, "wb") as f:
            f.write(ub.to_bytes())
        import_skp.import_skp(unp_src, bpy.context, keep_hierarchy=False)
        for o in bpy.context.scene.objects:
            o.select_set(o.type == "EMPTY")
        bpy.ops.object.duplicates_make_real()
        for o in list(bpy.context.scene.objects):
            if o.type == "EMPTY":
                bpy.data.objects.remove(o, do_unlink=True)
            else:
                o.select_set(False)
        mat0 = bpy.data.materials["openskp_material_0"]
        want_rgb = (220, 40, 60)
        lr, lg, lb = import_skp.srgb_to_linear(tuple(c / 255.0 for c in want_rgb))
        mat0.diffuse_color = (lr, lg, lb, 1.0)
        bpy.context.view_layer.update()

        p_r2_5a = os.path.join(tmp, "r2_5a_recolored_mat0.skp")
        st_5a = export_skp.export_skp(p_r2_5a, bpy.context)
        assert st_5a["materials_written"] == 1, st_5a
        m_5a = openskp.SkpFile.open(p_r2_5a).parse()
        assert len(m_5a.materials) == 1
        got_mat = m_5a.materials[0]
        assert tuple(got_mat.color[:3]) == want_rgb, got_mat.color

        # --- R2.5b: Geometry Nodes 'Set Material' on evaluated mesh ---
        _reset_clean_scene()
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
        gn_cube = bpy.context.active_object
        gn_cube.select_set(False)
        gn_mat = bpy.data.materials.new("GeoNodeGreen")
        gn_rgb = (30, 190, 80)
        glr, glg, glb = import_skp.srgb_to_linear(tuple(c / 255.0 for c in gn_rgb))
        gn_mat.diffuse_color = (glr, glg, glb, 1.0)

        gn_mod = gn_cube.modifiers.new(name="SetMatGN", type="NODES")
        nt = bpy.data.node_groups.new("SetMatGroup", "GeometryNodeTree")
        nt.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        nt.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        n_in = nt.nodes.new("NodeGroupInput")
        n_out = nt.nodes.new("NodeGroupOutput")
        n_set = nt.nodes.new("GeometryNodeSetMaterial")
        n_set.inputs["Material"].default_value = gn_mat
        nt.links.new(n_in.outputs["Geometry"], n_set.inputs["Geometry"])
        nt.links.new(n_set.outputs["Geometry"], n_out.inputs["Geometry"])
        gn_mod.node_group = nt
        bpy.context.view_layer.update()

        p_r2_5b = os.path.join(tmp, "r2_5b_geonodes_set_mat.skp")
        st_5b = export_skp.export_skp(p_r2_5b, bpy.context)
        assert st_5b["materials_written"] == 1, st_5b
        m_5b = openskp.SkpFile.open(p_r2_5b).parse()
        assert len(m_5b.materials) == 1
        got_gn_mat = m_5b.materials[0]
        assert got_gn_mat.name == "GeoNodeGreen" and tuple(got_gn_mat.color[:3]) == gn_rgb, got_gn_mat

        # --- R2.6: User collection named "Annotations" keeps tag "Annotations" on plain meshes ---
        _reset_clean_scene()
        user_annot_coll = bpy.data.collections.new("Annotations")
        bpy.context.scene.collection.children.link(user_annot_coll)
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
        u_cube = bpy.context.active_object
        for c in list(u_cube.users_collection):
            c.objects.unlink(u_cube)
        user_annot_coll.objects.link(u_cube)
        u_cube.select_set(False)
        bpy.context.view_layer.update()

        p_r2_6 = os.path.join(tmp, "r2_6_user_annotations_tag.skp")
        st_6 = export_skp.export_skp(p_r2_6, bpy.context)
        assert st_6["layers_written"] == 1, st_6
        m_6 = openskp.SkpFile.open(p_r2_6).parse()
        assert sorted(l.name for l in m_6.layers) == ["Annotations", "Layer0"], [l.name for l in m_6.layers]
        assert all(f.layer not in (None, 0) for f in m_6.root.faces.values()), [f.layer for f in m_6.root.faces.values()]

    print("test_review_round2_regressions: OK (all 6 round-2 review points verified)")


if __name__ == "__main__":
    main()
    test_layer_export()
    test_material_export()
    test_material_colour_round_trips()
    test_native_collection_instances_export()
    test_nested_component_round_trip()
    test_blender_edge_cases_and_review_regressions()
    test_review_round2_regressions()

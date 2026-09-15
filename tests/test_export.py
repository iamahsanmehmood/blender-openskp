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
from _load_addon import _load_module  # noqa: E402

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
    }, stats

    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor")
    )
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


if __name__ == "__main__":
    main()
    test_layer_export()
    test_material_export()

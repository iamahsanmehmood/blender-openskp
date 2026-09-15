"""Run with Blender's own headless mode, e.g.:

    blender --background --python tests/test_import.py

Unlike FreeCAD's freecadcmd (which runs a script with __name__ set to the
filename, not "__main__"), Blender's --python behaves like a normal
Python interpreter here - verified directly, not assumed - so this can
safely use a __main__ guard.

Verifies import_skp.import_skp() against real fixture files, checking the
exact unique-mesh/placement/triangle AND loose-edge (curve) counts against
known-good values - not just "it didn't crash." The curve counts are the
regression guard for a real gap found testing against a real structural-
framing file: a definition made entirely of loose edges (light-gauge-
steel/structural members drawn as construction lines, not solids)
contributed nothing to the import at all before openskp's
build_instanced_scene() gained curve-resource support.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _load_addon import _load_module  # noqa: E402

import_skp = _load_module("import_skp")

import bpy  # noqa: E402

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

# (unique_meshes, placements, triangles, unique_curve_meshes, curve_placements,
#  curve_runs, unique_layers, hidden_layers, textures_loaded) per fixture. Both
# committed fixtures happen to use only the default "Layer0" (unhidden) - real
# multi-layer/hidden-layer grouping was verified separately against a real
# external structural-framing file (13 layers, 2 genuinely hidden - see
# import_skp.py's module docstring and the blender-openskp README) since no
# committed fixture exercises that. capilla_quiroz_v17.skp DOES carry 3 real
# textures (glass/concrete/roofing materials from a SketchUp material library)
# - a genuine, not synthetic, cross-check for texture import.
EXPECTED = {
    "SU_File.skp": (1, 1, 104, 0, 0, 0, 1, 0, 0),
    "capilla_quiroz_v17.skp": (3, 4, 871, 2, 2, 20, 1, 0, 3),
}


def check(fixture_name):
    path = os.path.join(FIXTURES_DIR, fixture_name)
    # Several fixtures share a generic definition name ("ROOT_MODEL", for
    # a root that has no real name of its own) - a plain bpy.data.objects
    # lookup by that name would silently find an EARLIER fixture's
    # leftover object once a later one gets auto-suffixed ".001" by
    # Blender (confirmed directly: this is exactly what happened before
    # the objects-created-by-THIS-import filter below was added). So
    # every check below that looks an object up by name is restricted to
    # objects that didn't exist prior to this specific import call.
    objects_before = set(bpy.data.objects)
    stats = import_skp.import_skp(path)
    new_objects = set(bpy.data.objects) - objects_before

    got = (
        stats["unique_meshes"], stats["placements"], stats["triangles"],
        stats["unique_curve_meshes"], stats["curve_placements"], stats["curve_runs"],
        stats["unique_layers"], stats["hidden_layers"], stats["textures_loaded"],
    )
    print(f"{fixture_name}: {got}")
    expected = EXPECTED.get(fixture_name)
    if expected is not None:
        assert got == expected, f"{fixture_name}: expected {expected}, got {got}"
    if stats["unique_meshes"]:
        check_source_geometry_is_discoverable_but_excluded(fixture_name)
        check_materials_match_gltf_materials(fixture_name, new_objects)
        check_layer_collection(fixture_name)
        check_textures(fixture_name, new_objects)
    return got


def _find_layer_collection(layer_collection, target_collection):
    if layer_collection.collection == target_collection:
        return layer_collection
    for child in layer_collection.children:
        found = _find_layer_collection(child, target_collection)
        if found is not None:
            return found
    return None


def check_source_geometry_is_discoverable_but_excluded(fixture_name):
    """Regression test for github#3: a placement Empty has no mesh data
    of its own, so before this fix there was no way to even find - let
    alone select and Tab-edit - the real master mesh object, since its
    Collection was never linked anywhere. Checks the fix's two halves:
    discoverable (linked into the Outliner) but excluded (no duplicate
    render at the origin) by default. Called from check() itself (not
    re-importing) - collection names aren't unique across repeat imports
    of the same file, so a second import here would silently look up a
    ".001"-suffixed collection instead."""
    root_name = os.path.splitext(fixture_name)[0]
    sources_name = f"{root_name} (source geometry)"
    sources_coll = bpy.data.collections.get(sources_name)
    assert sources_coll is not None, f"{sources_name} was never created/linked"

    layer_coll = _find_layer_collection(bpy.context.view_layer.layer_collection, sources_coll)
    assert layer_coll is not None, f"{sources_name} is not reachable from the View Layer at all"
    assert layer_coll.exclude, f"{sources_name} should be excluded by default"

    # Every master object should be inside it (directly or via a nested
    # per-definition collection), reachable via Outliner navigation, and
    # NOT part of the active view layer's evaluated set while excluded.
    all_source_objects = {o.name for c in sources_coll.children for o in c.objects}
    assert all_source_objects, f"{sources_name} has no source objects at all"
    evaluated = {o.name for o in bpy.context.view_layer.objects}
    assert not (all_source_objects & evaluated), (
        "source objects must not be part of the evaluated view layer while excluded "
        f"(would render duplicated at the origin): {all_source_objects & evaluated}"
    )
    print(f"{fixture_name}: {len(all_source_objects)} source objects discoverable and excluded - OK")


def check_materials_match_gltf_materials(fixture_name, new_objects):
    """Cross-validates the Blender materials/per-polygon assignment
    against openskp's own (already-tested) InstancedScene.gltf_materials
    + LocalPrimitive.material_index - the source of truth this addon
    reads from, rather than asserting a color in isolation. Called from
    check() itself (not re-importing), same reason as the discoverability
    check above.

    `new_objects` restricts the by-name lookup below to objects created
    by THIS fixture's own import call. Several fixtures share a generic
    definition name ("ROOT_MODEL", used when a definition has no real
    name of its own) - a plain bpy.data.objects.get(name) lookup would
    silently match an earlier fixture's leftover object once this
    fixture's own object gets auto-suffixed ".001" by Blender for
    colliding with it (confirmed directly: this is exactly what happened
    before new_objects was threaded through here)."""
    vendor_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor")
    if vendor_dir not in sys.path:
        sys.path.insert(0, vendor_dir)
    import openskp

    path = os.path.join(FIXTURES_DIR, fixture_name)
    scene = openskp.SkpFile.open(path).build_instanced_scene()
    if not scene.gltf_materials:
        return

    # Blender auto-suffixes a colliding name ("ROOT_MODEL" -> "ROOT_MODEL.001")
    # rather than erroring - and a generic definition name like "ROOT_MODEL"
    # (used when a definition has no real name of its own) is common enough
    # to collide with an EARLIER fixture's own object in this same session
    # (confirmed directly: this is exactly what SU_File.skp's "ROOT_MODEL"
    # does against capilla_quiroz_v17.skp's own "ROOT_MODEL"). So match by
    # base name (suffix stripped) within this fixture's own new_objects
    # only, rather than requiring an exact, unsuffixed name - but restrict
    # this to actual mesh data-objects (the master/source objects this
    # test cares about), since placement Empties of the same collection
    # get auto-suffixed the same way and would otherwise collide with -
    # and, depending on dict insertion order, silently shadow - the real
    # mesh object in this lookup (confirmed directly: without the type
    # filter this failed with "'NoneType' object has no attribute
    # 'materials'", i.e. it had matched an Empty instead).
    new_objects_by_base_name = {
        re.sub(r"\.\d{3}$", "", o.name): o for o in new_objects if o.type == "MESH" and o.data is not None
    }

    checked_any = False
    for resource in scene.mesh_resources:
        obj_name = resource.definition_name or resource.id
        obj = new_objects_by_base_name.get(obj_name)
        assert obj is not None, (
            f"{fixture_name}: no newly-created Blender object named {obj_name!r} "
            f"(created this import: {sorted(o.name for o in new_objects)})"
        )

        expected_colors = []
        expected_indices = []
        for prim in resource.primitives:
            pbr = scene.gltf_materials[prim.material_index].get("pbrMetallicRoughness", {})
            color = tuple(round(c, 3) for c in pbr.get("baseColorFactor", [0.8, 0.8, 0.8, 1.0]))
            expected_colors.append(color)
            expected_indices.extend([len(expected_colors) - 1] * (len(prim.indices) // 3))

        got_colors = [tuple(round(c, 3) for c in mat.diffuse_color) for mat in obj.data.materials]
        assert got_colors == expected_colors, (
            f"{fixture_name}/{obj_name}: material colors {got_colors} != expected {expected_colors}"
        )
        got_indices = [p.material_index for p in obj.data.polygons]
        assert got_indices == expected_indices, (
            f"{fixture_name}/{obj_name}: per-polygon material assignment doesn't match source primitives"
        )
        checked_any = True

    assert checked_any, f"{fixture_name}: has gltf_materials but no mesh resources to check against"
    print(f"{fixture_name}: materials match openskp's own gltf_materials - OK")


def check_layer_collection(fixture_name):
    """Both committed fixtures use only SketchUp's default "Layer0" - not
    a real cross-file multi-layer test, but does verify the mechanics:
    a "Layer0" Collection exists directly under the root Collection
    (not inside sources_collection - a layer is a placement property,
    not a definition property), holds every placement Empty, and isn't
    hidden (matching layer_hidden's own "Layer0": False for these
    files). Called from check() itself, same reason as the checks above."""
    root_name = os.path.splitext(fixture_name)[0]
    root_coll = bpy.data.collections.get(root_name)
    assert root_coll is not None, f"{root_name}: root collection was never created"

    # Same cross-fixture name-collision reason as check_materials_match_gltf_materials:
    # a second fixture's own "Layer0" collection gets Blender-auto-suffixed
    # ("Layer0.001") when an EARLIER fixture already created one in this same
    # session - match by base name (suffix stripped) rather than requiring
    # an exact "Layer0" (confirmed directly: this is exactly what happened
    # here on SU_File.skp, processed after capilla_quiroz_v17.skp).
    layer0 = None
    for child in root_coll.children:
        if re.sub(r"\.\d{3}$", "", child.name) == "Layer0":
            layer0 = child
            break
    assert layer0 is not None, f"{fixture_name}: no 'Layer0' collection directly under {root_name!r}"
    assert not layer0.hide_viewport and not layer0.hide_render, (
        f"{fixture_name}: 'Layer0' should be visible (layer_hidden['Layer0'] is False in both fixtures)"
    )

    placement_empties = [o for o in layer0.objects if o.instance_type == "COLLECTION"]
    assert placement_empties, f"{fixture_name}: 'Layer0' collection has no placement Empties in it"
    print(f"{fixture_name}: {len(placement_empties)} placements grouped under visible 'Layer0' - OK")


def check_textures(fixture_name, new_objects):
    """Cross-validates textured-material import against openskp's own
    InstancedScene.textures/gltf_materials, the same way
    check_materials_match_gltf_materials does for solid colors: every
    material with a pbrMetallicRoughness.baseColorTexture must have a
    real Image Texture node in Blender, wired to an Image with the
    correct pixel dimensions (decoded independently from the source
    file's own raw bytes, not just "some image got created") - and the
    same source texture shared by two different materials (a real case
    in capilla_quiroz_v17.skp: the translucent-glass material appears on
    both windows and the door) must load as the SAME Blender Image
    datablock, not two separate copies. Also checks every mesh with a
    textured material actually has a populated UV layer, since a texture
    with no UVs would just look wrong, not fail outright."""
    import openskp

    path = os.path.join(FIXTURES_DIR, fixture_name)
    scene = openskp.SkpFile.open(path).build_instanced_scene()
    textures = getattr(scene, "textures", None) or []
    if not textures:
        return

    def expected_pixel_size(data):
        # Minimal PNG/JPEG header dimension decode - avoids depending on
        # PIL, which isn't guaranteed present in Blender's own bundled
        # Python interpreter.
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            w = int.from_bytes(data[16:20], "big")
            h = int.from_bytes(data[20:24], "big")
            return (w, h)
        if data[:2] == b"\xff\xd8":
            i = 2
            while i < len(data):
                if data[i] != 0xFF:
                    i += 1
                    continue
                marker = data[i + 1]
                if marker in (0xC0, 0xC1, 0xC2, 0xC3):
                    h = int.from_bytes(data[i + 5:i + 7], "big")
                    w = int.from_bytes(data[i + 7:i + 9], "big")
                    return (w, h)
                seg_len = int.from_bytes(data[i + 2:i + 4], "big")
                i += 2 + seg_len
        return None

    new_objects_by_base_name = {
        re.sub(r"\.\d{3}$", "", o.name): o for o in new_objects if o.type == "MESH" and o.data is not None
    }

    image_by_texture_index = {}
    checked_any = False
    for resource in scene.mesh_resources:
        obj_name = resource.definition_name or resource.id
        obj = new_objects_by_base_name.get(obj_name)
        assert obj is not None, f"{fixture_name}: no newly-created Blender object named {obj_name!r}"

        textured_material_indices = {
            prim.material_index for prim in resource.primitives
            if scene.gltf_materials[prim.material_index].get("pbrMetallicRoughness", {}).get("baseColorTexture")
        }
        if not textured_material_indices:
            continue

        assert obj.data.uv_layers, f"{fixture_name}/{obj_name}: has a textured material but no UV layer at all"
        assert len(obj.data.uv_layers[0].data) == len(obj.data.loops), (
            f"{fixture_name}/{obj_name}: UV layer doesn't cover every loop"
        )

        for material_index in textured_material_indices:
            pbr = scene.gltf_materials[material_index]["pbrMetallicRoughness"]
            texture_index = pbr["baseColorTexture"]["index"]
            expected_size = expected_pixel_size(textures[texture_index].data)

            bl_mat = bpy.data.materials.get(f"openskp_material_{material_index}")
            assert bl_mat is not None, f"{fixture_name}: no Blender material named openskp_material_{material_index}"
            tex_nodes = [n for n in bl_mat.node_tree.nodes if n.type == "TEX_IMAGE"]
            assert len(tex_nodes) == 1, (
                f"{fixture_name}/{bl_mat.name}: expected exactly 1 Image Texture node, got {len(tex_nodes)}"
            )
            image = tex_nodes[0].image
            assert image is not None, f"{fixture_name}/{bl_mat.name}: Image Texture node has no image assigned"
            assert tuple(image.size) == expected_size, (
                f"{fixture_name}/{bl_mat.name}: image size {tuple(image.size)} != expected {expected_size}"
            )

            if texture_index in image_by_texture_index:
                assert image is image_by_texture_index[texture_index], (
                    f"{fixture_name}: texture {texture_index} loaded as two different Blender Images "
                    "(should be deduplicated and shared across materials)"
                )
            else:
                image_by_texture_index[texture_index] = image
            checked_any = True

    assert checked_any, f"{fixture_name}: has textures but no textured mesh resources to check against"
    print(
        f"{fixture_name}: {len(image_by_texture_index)} textures correctly wired "
        f"(matching pixel sizes, deduplicated) - OK"
    )


def main():
    checked = 0
    for name in os.listdir(FIXTURES_DIR):
        if name.endswith(".skp"):
            check(name)
            checked += 1
    assert checked == len(EXPECTED), (
        f"expected to check {len(EXPECTED)} fixtures, found {checked} .skp files in {FIXTURES_DIR}"
    )
    print("test_import: all fixtures OK (including source-geometry discoverability)")


if __name__ == "__main__":
    main()

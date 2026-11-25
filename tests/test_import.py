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
from _load_addon import _load_module, ensure_openskp_importable  # noqa: E402

ensure_openskp_importable()
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
    # OpenSKP's own nesting fixture: groups Outer > Shelf > Inner, each with faces.
    "nesting_2021.skp": (3, 3, 6, 0, 0, 0, 1, 0, 0),
    # One face each with a TIFF, TGA or GIF textured material. The last number, textures_loaded, is
    # 0 with an openskp that leaves such a texture out (before 1.6.0, which is bundled now) and, with one that
    # keeps it, 1 for TIFF and TGA but still 0 for GIF, which Blender cannot use as an image
    # (_UNLOADABLE_TEXTURES); check() sets it from the library in use. See
    # check_original_textures_are_loaded.
    "texture_tiff_2021.skp": (1, 1, 4, 0, 0, 0, 1, 0, 0),
    "texture_tga_2021.skp": (1, 1, 4, 0, 0, 0, 1, 0, 0),
    "texture_gif_2021.skp": (1, 1, 4, 0, 0, 0, 1, 0, 0),
    # Scene camera fixtures (#26): gondola_v20.skp (1 parallel scene "Cena1")
    # and two_scenes_2021.skp (SDK-written: perspective + parallel scenes).
    "gondola_v20.skp": (12, 81, 3663, 4, 18, 4105, 1, 0, 0),
    "two_scenes_2021.skp": (1, 1, 12, 0, 0, 0, 1, 0, 0),
    # 3-level nested components (Bolt -> Bracket -> Frame) across 4 tags (#28).
    "nested_components_2021.skp": (4, 15, 8, 0, 0, 0, 4, 0, 0),
    # Annotations fixtures (#27): 36x12" root slab + 24x6" Beam component on
    # Framing, 1 leader text, 1 screen text, and 2 linear dimensions (2017 & 2021).
    "annotations_2017.skp": (2, 2, 4, 0, 0, 0, 1, 0, 0),
    "annotations_2021.skp": (2, 2, 4, 0, 0, 0, 2, 0, 0),
}
TEXTURE_FIXTURES = ("texture_tiff_2021.skp", "texture_tga_2021.skp", "texture_gif_2021.skp")


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
    import openskp

    orig_parse = openskp.SkpFile.parse
    parse_calls = 0

    def count_parse(self, *args, **kwargs):
        nonlocal parse_calls
        parse_calls += 1
        return orig_parse(self, *args, **kwargs)

    openskp.SkpFile.parse = count_parse
    try:
        objects_before = set(bpy.data.objects)
        stats = import_skp.import_skp(path)
        new_objects = set(bpy.data.objects) - objects_before
    finally:
        openskp.SkpFile.parse = orig_parse
    assert parse_calls == 0, (
        f"{fixture_name}: import_skp() called SkpFile.parse() {parse_calls} times (must not parse twice)"
    )
    assert stats["texts_imported"] == 0 and stats["dimensions_imported"] == 0, (
        f"{fixture_name}: annotations must stay off by default, got {stats}"
    )
    assert not [o for o in new_objects if o.type == "FONT"], (
        f"{fixture_name}: unexpected FONT objects when import_annotations=False"
    )

    got = (
        stats["unique_meshes"], stats["placements"], stats["triangles"],
        stats["unique_curve_meshes"], stats["curve_placements"], stats["curve_runs"],
        stats["unique_layers"], stats["hidden_layers"], stats["textures_loaded"],
    )
    print(f"{fixture_name}: {got}")
    expected = EXPECTED.get(fixture_name)
    if fixture_name in TEXTURE_FIXTURES:
        loaded = 1 if _library_keeps_original_textures() and fixture_name != "texture_gif_2021.skp" else 0
        expected = expected[:8] + (loaded,)
    elif fixture_name == "annotations_2017.skp":
        sh = openskp.SkpFile.open(path).build_instanced_scene().scene_hierarchy
        expected_layers = len({n.layer or "Layer0" for n in [sh, *sh.children]})
        expected = expected[:6] + (expected_layers,) + expected[7:]
    if expected is not None:
        assert got == expected, f"{fixture_name}: expected {expected}, got {got}"
    if stats["unique_meshes"]:
        check_source_geometry_is_discoverable_but_excluded(fixture_name)
        check_materials_match_gltf_materials(fixture_name, new_objects)
        check_layer_collection(fixture_name)
        check_textures(fixture_name, new_objects)
        check_material_names_and_sides(fixture_name, new_objects)
    check_scene_cameras_for_fixture(fixture_name, new_objects, stats)
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
    ensure_openskp_importable()
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

        if _twin_pairs(_source_triangles(resource)):
            check_front_back_pairs(fixture_name, scene, resource, obj)
            checked_any = True
            continue

        expected_colors = []
        expected_indices = []
        for prim in resource.primitives:
            pbr = scene.gltf_materials[prim.material_index].get("pbrMetallicRoughness", {})
            r, g, b, a = pbr.get("baseColorFactor", [0.8, 0.8, 0.8, 1.0])
            color = tuple(round(c, 3) for c in (*import_skp.srgb_to_linear((r, g, b)), a))
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


def _rounded(point):
    return tuple(round(c, 4) for c in point)


def _source_triangles(resource):
    """[(rounded position set, winding, material index)] for a resource's
    triangles, in primitive order. Winding is the parity of the corner
    order against sorted order: rotating the corners keeps it, reversing
    them flips it. Written independently of import_skp's own twin search
    (which compares positions after rotating to the smallest corner)."""
    out = []
    for prim in resource.primitives:
        pos, idx = prim.positions, prim.indices
        for i in range(0, len(idx), 3):
            corners = [_rounded(pos[3 * k:3 * k + 3]) for k in idx[i:i + 3]]
            winding = sum(corners[a] > corners[b] for a in range(3) for b in range(a + 1, 3)) % 2
            out.append((frozenset(corners), winding, prim.material_index, len(set(corners)) == 3))
    return out


def _twin_pairs(source):
    """[(front, back) source indices]: the same face seen from its two
    sides (same corners, opposite winding) painted with different materials."""
    first_of = {}
    pairs = []
    paired = set()
    for t, (shape, winding, material, proper) in enumerate(source):
        if not proper:
            continue
        for other in first_of.get(shape, []):
            if other not in paired and source[other][1] != winding and source[other][2] != material:
                pairs.append((other, t))
                paired.update((other, t))
                break
        else:
            first_of.setdefault(shape, []).append(t)
    return pairs


def check_front_back_pairs(fixture_name, scene, resource, obj):
    """A face painted differently on its two sides arrives as two
    single-sided copies on the same plane, wound opposite ways. The import
    keeps ONE polygon per such pair, with a material that mixes the two
    paints on Geometry > Backfacing; the twin is gone, so nothing overlaps
    and Solid shading, EEVEE and Cycles all agree."""
    source = _source_triangles(resource)
    pairs = _twin_pairs(source)
    mesh = obj.data
    label = f"{fixture_name}/{obj.name}"

    assert len(mesh.polygons) == len(source) - len(pairs), (
        f"{label}: {len(mesh.polygons)} polygons, expected {len(source)} source triangles - {len(pairs)} pairs"
    )

    by_shape = {}
    for t, (shape, winding, material, proper) in enumerate(source):
        by_shape.setdefault(shape, []).append((winding, material, t))
    paired_materials = {}  # shape -> {winding: material} for the pairs
    for front, back in pairs:
        shape = source[front][0]
        paired_materials[shape] = {source[front][1]: source[front][2], source[back][1]: source[back][2]}

    seen = {}  # shape -> (winding, material) of the lone polygons, for the leftover check
    paired_polygons = 0
    for poly in mesh.polygons:
        corners = [_rounded(tuple(mesh.vertices[v].co)) for v in poly.vertices]
        shape = frozenset(corners)
        winding = sum(corners[a] > corners[b] for a in range(3) for b in range(a + 1, 3)) % 2
        mat = mesh.materials[poly.material_index]
        front_index = mat["openskp_material_index"]
        back_index = mat.get("openskp_back_material_index")
        if back_index is not None:
            paired_polygons += 1
            assert shape in paired_materials, f"{label}: a front/back material on a face that has no twin"
            sides = paired_materials[shape]
            assert {front_index, back_index} == set(sides.values()), (
                f"{label}: paired face has materials {front_index}/{back_index}, source has {sorted(sides.values())}"
            )
            assert sides[winding] == front_index, (
                f"{label}: the polygon's front paint {front_index} is not the copy wound like it ({sides[winding]})"
            )
        else:
            # A shape can also keep a lone copy (a same-paint duplicate stacked on a pair).
            assert any(w == winding and m == front_index for w, m, _ in by_shape[shape]), (
                f"{label}: polygon material {front_index} matches no source triangle with that winding"
            )
        if back_index is None:
            seen.setdefault(shape, []).append((winding, front_index))
    assert paired_polygons == len(pairs), f"{label}: {paired_polygons} paired polygons, expected {len(pairs)}"

    # Nothing that could have been paired is left over: no two lone copies of
    # a shape with opposite winding and different paints.
    for shape, kept in seen.items():
        for i, (w1, m1) in enumerate(kept):
            for w2, m2 in kept[i + 1:]:
                assert not (w1 != w2 and m1 != m2), f"{label}: two opposite-wound copies with different paints remain"

    pair_materials = [m for m in mesh.materials if m.get("openskp_back_material_index") is not None]
    assert pair_materials, f"{label}: pairs found but no front/back material"
    for mat in pair_materials:
        front_gltf = scene.gltf_materials[mat["openskp_material_index"]]
        back_gltf = scene.gltf_materials[mat["openskp_back_material_index"]]
        nodes = mat.node_tree.nodes
        assert not mat.use_backface_culling, f"{label}/{mat.name}: both sides must be drawn"
        assert sum(n.type == "BSDF_PRINCIPLED" for n in nodes) == 2, f"{label}/{mat.name}: expected two surfaces"
        mix = next(n for n in nodes if n.type == "MIX_SHADER")
        assert mix.inputs["Fac"].links[0].from_socket.name == "Backfacing", f"{label}/{mat.name}: mix not on Backfacing"
        out = next(n for n in nodes if n.type == "OUTPUT_MATERIAL")
        assert out.inputs["Surface"].links[0].from_node.name == mix.name
        front_in = mix.inputs[1].links[0].from_node
        back_in = mix.inputs[2].links[0].from_node
        assert front_in.name != back_in.name and front_in.type == back_in.type == "BSDF_PRINCIPLED"
        front_painted = bool(front_gltf.get("name")) or bool(
            front_gltf.get("pbrMetallicRoughness", {}).get("baseColorTexture")
        )
        back_painted = bool(back_gltf.get("name")) or bool(
            back_gltf.get("pbrMetallicRoughness", {}).get("baseColorTexture")
        )
        solid_gltf = back_gltf if back_painted and not front_painted else front_gltf
        r, g, b, a = solid_gltf.get("pbrMetallicRoughness", {}).get("baseColorFactor", [0.8, 0.8, 0.8, 1.0])
        expected = tuple(round(c, 3) for c in (*import_skp.srgb_to_linear((r, g, b)), a))
        assert tuple(round(c, 3) for c in mat.diffuse_color) == expected, (
            f"{label}/{mat.name}: Solid shading should show the painted side (or front when both are painted)"
        )
        if back_gltf.get("pbrMetallicRoughness", {}).get("baseColorTexture"):
            assert import_skp._BACK_UV_LAYER in mesh.uv_layers, f"{label}: textured back paint but no back UV layer"
            back_tex = next(n for n in nodes if n.type == "TEX_IMAGE" and n.inputs["Vector"].links)
            uv_node = back_tex.inputs["Vector"].links[0].from_node
            assert uv_node.type == "UVMAP" and uv_node.uv_map == import_skp._BACK_UV_LAYER, (
                f"{label}/{mat.name}: back texture must read {import_skp._BACK_UV_LAYER}"
            )
    print(
        f"{label}: {len(pairs)} front/back pairs kept as one polygon each, "
        f"{len(pair_materials)} front/back materials, no overlapping copies - OK"
    )


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

        def is_textured(index):
            return bool(scene.gltf_materials[index].get("pbrMetallicRoughness", {}).get("baseColorTexture"))

        textured_material_indices = {
            prim.material_index for prim in resource.primitives if is_textured(prim.material_index)
        }
        if not textured_material_indices:
            continue

        assert obj.data.uv_layers, f"{fixture_name}/{obj_name}: has a textured material but no UV layer at all"
        assert len(obj.data.uv_layers[0].data) == len(obj.data.loops), (
            f"{fixture_name}/{obj_name}: UV layer doesn't cover every loop"
        )

        # Every Blender material of this mesh: a single paint (one surface) or
        # a front/back pair (two surfaces; the back one reads UVMap_back).
        seen_materials = set()
        for bl_mat in obj.data.materials:
            front_index = bl_mat["openskp_material_index"]
            back_index = bl_mat.get("openskp_back_material_index")
            roles = [(front_index, False)] + ([(back_index, True)] if back_index is not None else [])
            textured_roles = [(index, is_back) for index, is_back in roles if is_textured(index)]
            tex_nodes = [n for n in bl_mat.node_tree.nodes if n.type == "TEX_IMAGE"]
            assert len(tex_nodes) == len(textured_roles), (
                f"{fixture_name}/{bl_mat.name}: expected {len(textured_roles)} Image Texture node(s), "
                f"got {len(tex_nodes)}"
            )
            for index, is_back in textured_roles:
                seen_materials.add(index)
                texture_index = scene.gltf_materials[index]["pbrMetallicRoughness"]["baseColorTexture"]["index"]
                expected_size = expected_pixel_size(textures[texture_index].data)
                # The back paint's texture is the one fed by a UV Map node.
                tex_node = next(n for n in tex_nodes if bool(n.inputs["Vector"].links) == is_back)
                image = tex_node.image
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
        assert textured_material_indices <= seen_materials, (
            f"{fixture_name}/{obj_name}: textured materials {sorted(textured_material_indices - seen_materials)} "
            "have no Image Texture in any Blender material"
        )

    assert checked_any, f"{fixture_name}: has textures but no textured mesh resources to check against"
    print(
        f"{fixture_name}: {len(image_by_texture_index)} textures correctly wired "
        f"(matching pixel sizes, deduplicated) - OK"
    )


def _material_for_index(obj, material_index):
    return next(
        (m for m in obj.data.materials if m.get("openskp_material_index") == material_index), None
    )


def check_material_names_and_sides(fixture_name, new_objects):
    """Every Blender material is named after its SketchUp material when
    openskp provides one (openskp 1.6+), and a single-sided material - one
    side of a face whose two sides are painted differently, emitted twice on
    the same plane - is invisible from behind, so the other side's copy is
    the one that shows (backface culling for the viewport/EEVEE, a
    Backfacing mix for Cycles). Double-sided materials stay visible from
    both sides."""
    import openskp

    path = os.path.join(FIXTURES_DIR, fixture_name)
    scene = openskp.SkpFile.open(path).build_instanced_scene()
    seen = {}
    pairs = {}
    for obj in new_objects:
        if obj.type != "MESH" or obj.data is None:
            continue
        for mat in obj.data.materials:
            if mat.get("openskp_back_material_index") is not None:
                pairs[(mat["openskp_material_index"], mat["openskp_back_material_index"])] = mat
            else:
                seen[mat.get("openskp_material_index")] = mat
    assert seen or pairs, f"{fixture_name}: no imported materials"
    for (front, back), mat in pairs.items():
        def label(i):
            return scene.gltf_materials[i].get("name") or f"openskp_material_{i}"

        assert re.sub(r"\.\d{3}$", "", mat.name) == f"{label(front)} | {label(back)}", (
            f"{fixture_name}: front/back material named {mat.name!r}"
        )
        assert not mat.use_backface_culling, f"{fixture_name}/{mat.name}: both sides must be drawn"
        assert any(n.type == "NEW_GEOMETRY" for n in mat.node_tree.nodes), f"{fixture_name}/{mat.name}: no Backfacing"
    named = 0
    for index, mat in seen.items():
        gltf = scene.gltf_materials[index]
        base_name = re.sub(r"\.\d{3}$", "", mat.name)
        assert base_name == (gltf.get("name") or f"openskp_material_{index}"), (
            f"{fixture_name}: material {index} named {mat.name!r}, expected {gltf.get('name')!r}"
        )
        named += bool(gltf.get("name"))
        hides_back = any(n.type == "NEW_GEOMETRY" for n in mat.node_tree.nodes)
        if gltf.get("doubleSided"):
            assert not mat.use_backface_culling and not hides_back, f"{fixture_name}/{mat.name}: double-sided"
        else:
            assert mat.use_backface_culling and hides_back, f"{fixture_name}/{mat.name}: back side not hidden"
            out = next(n for n in mat.node_tree.nodes if n.type == "OUTPUT_MATERIAL")
            assert out.inputs["Surface"].links[0].from_node.type == "MIX_SHADER"
    print(
        f"{fixture_name}: {len(seen)} materials ({named} with SketchUp names) and "
        f"{len(pairs)} front/back materials, back sides correct - OK"
    )


def check_colorized_texture_is_tinted():
    """A colorized material (SketchUp's "[Name]1" copy, openskp's
    extras.sketchup.colorized) re-tints the shared image toward its own
    colour, instead of showing the untinted source image."""
    import openskp

    scene = openskp.SkpFile.open(os.path.join(FIXTURES_DIR, "capilla_quiroz_v17.skp")).build_instanced_scene()
    textures = scene.textures
    gltf_materials = [
        {
            "name": f"[Veneer]1 type {colorize_type}",
            "pbrMetallicRoughness": {"baseColorFactor": [68 / 255, 40 / 255, 21 / 255, 1.0],
                                     "baseColorTexture": {"index": 0}},
            "extras": {"sketchup": {"colorized": True, "colorize_type": colorize_type}},
            "doubleSided": True,
        }
        for colorize_type in (0, 1)
    ]
    for index, colorize_type in enumerate((0, 1)):
        mat = import_skp._get_or_build_material(index, gltf_materials, {}, textures, {})
        nodes = mat.node_tree.nodes
        bsdf = nodes["Principled BSDF"]
        tint = bsdf.inputs["Base Color"].links[0].from_node
        assert tint.type == "MIX" and tint.blend_type == "MULTIPLY", f"{mat.name}: Base Color isn't tinted"
        feeds = {link.from_node.type for s in tint.inputs if s.type == "RGBA" for link in s.links}
        assert feeds == ({"TEX_IMAGE"} if colorize_type == 0 else {"RGBTOBW"}), f"{mat.name}: {feeds}"
        scale = next(s for s in tint.inputs if s.name == "B" and s.type == "RGBA").default_value
        assert scale[0] > scale[1] > scale[2], f"{mat.name}: tint {tuple(scale)} isn't toward the brown colour"
    print("colorized texture: re-tinted toward its colour (shift and tint) - OK")


def _png(width, height, rows):
    """A minimal PNG, RGBA when the pixels have 4 values and RGB when they
    have 3, so the alpha test needs no image file: none of the committed
    fixtures has a texture with its own alpha channel."""
    import struct
    import zlib

    channels = len(rows[0][0])
    raw = b"".join(b"\x00" + bytes(v for px in row for v in px) for row in rows)

    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    header = struct.pack(">IIBBBBB", width, height, 8, 6 if channels == 4 else 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def check_alpha_materials_use_the_right_render_method():
    """A texture with its own alpha channel (a fence, a lattice, a leaf)
    renders with EEVEE's Dithered method, which needs no sorting, so the
    holes show what is behind them from every angle; a flat translucent
    colour (SketchUp's glass) stays Blended, which is smooth where Dithered
    is grainy; and a front/back pair takes Dithered when either paint is a
    cut-out texture, whichever order the paints come in."""

    class Texture:
        def __init__(self, data):
            self.data, self.mime_type, self.filename = data, "image/png", ""

    clear, solid = (255, 255, 255, 0), (255, 255, 255, 255)
    textures = [
        Texture(_png(2, 2, [[clear, solid], [solid, clear]])),  # 0: cut-out
        Texture(_png(2, 2, [[(255, 255, 255)] * 2] * 2)),  # 1: opaque image
    ]

    def painted(name, texture_index=None, alpha=1.0):
        pbr = {"baseColorFactor": [0.5, 0.5, 0.5, alpha]}
        if texture_index is not None:
            pbr["baseColorTexture"] = {"index": texture_index}
        return {"name": name, "pbrMetallicRoughness": pbr, "doubleSided": True}

    gltf = [
        painted("cut-out", 0),
        painted("opaque image", 1),
        painted("glass", alpha=0.5),
        painted("flat"),
        painted("glass image", 1, alpha=0.5),
    ]

    def build(index):
        return import_skp._get_or_build_material(index, gltf, {}, textures, {})

    cutout = build(0)
    alpha_links = cutout.node_tree.nodes["Principled BSDF"].inputs["Alpha"].links
    assert alpha_links and alpha_links[0].from_node.type == "TEX_IMAGE", "cut-out alpha isn't wired to Alpha"
    assert cutout.surface_render_method == "DITHERED", "a cut-out texture must use the Dithered render method"
    assert build(2).surface_render_method == "BLENDED", "flat translucent glass must stay Blended"
    assert build(4).surface_render_method == "BLENDED", "a translucent material with an opaque image stays Blended"
    assert build(1).surface_render_method == "DITHERED" and build(3).surface_render_method == "DITHERED"

    for front, back in ((0, 2), (2, 0), (0, 4), (4, 0)):
        pair = import_skp._get_or_build_pair_material(front, back, gltf, {}, textures, {})
        assert pair.surface_render_method == "DITHERED", f"pair ({front}, {back}) has a cut-out, must be Dithered"
    glass_pair = import_skp._get_or_build_pair_material(2, 4, gltf, {}, textures, {})
    assert glass_pair.surface_render_method == "BLENDED", "a pair of flat translucent paints must stay Blended"
    print("alpha materials: cut-outs Dithered, flat glass Blended, pairs Dithered when either paint cuts out - OK")


def check_vertices_are_welded():
    """openskp gives every face its own copy of its vertices; the importer
    welds the ones at the same position, so a box is 8 vertices (what Merge by
    Distance leaves) and not one detached set per face, while each corner keeps
    its own UV so a texture seam survives the weld (blender-openskp#16)."""
    import tempfile

    import bmesh
    import openskp

    s = 100.0

    def P(x, y, z):
        return (x * s, y * s, z * s)

    # An open box (no top), faces wound outward, textured on the front side.
    faces = [
        [P(0, 0, 0), P(0, 1, 0), P(1, 1, 0), P(1, 0, 0)],
        [P(0, 0, 0), P(1, 0, 0), P(1, 0, 1), P(0, 0, 1)],
        [P(1, 0, 0), P(1, 1, 0), P(1, 1, 1), P(1, 0, 1)],
        [P(1, 1, 0), P(0, 1, 0), P(0, 1, 1), P(1, 1, 1)],
        [P(0, 1, 0), P(0, 0, 0), P(0, 0, 1), P(0, 1, 1)],
    ]
    with tempfile.TemporaryDirectory() as tmp:
        image = os.path.join(tmp, "grid.png")
        with open(image, "wb") as f:
            f.write(_png(2, 2, [[(255, 0, 0), (0, 255, 0)], [(0, 0, 255), (255, 255, 0)]]))
        skp = os.path.join(tmp, "box.skp")
        builder = openskp.SkpBuilder()
        front = builder.add_texture_material("Grid", image, 10, 10)
        back = builder.add_material("Orange", (255, 128, 0, 255))
        for face in faces:
            builder.add_face(face, material=front, back_material=back)
        builder.save(skp)

        resource = openskp.SkpFile.open(skp).build_instanced_scene().mesh_resources[0]
        source = {
            (tuple(round(c, 3) for c in prim.positions[i * 3:i * 3 + 3]),
             tuple(round(c, 4) for c in prim.uvs[i * 2:i * 2 + 2]))
            for prim in resource.primitives
            for i in range(len(prim.positions) // 3)
        }
        source_vertices = sum(len(prim.positions) // 3 for prim in resource.primitives)
        before = set(bpy.data.objects)
        import_skp.import_skp(skp)
        (obj,) = [o for o in set(bpy.data.objects) - before if o.type == "MESH"]

    mesh = obj.data
    assert source_vertices > 8, "the test box must arrive with detached per-face vertices"
    assert len(mesh.vertices) == 8, f"box: {len(mesh.vertices)} vertices, expected the 8 corners"
    assert len(mesh.polygons) == 10, f"box: {len(mesh.polygons)} polygons, expected 5 faces x 2 triangles"
    bm = bmesh.new()
    bm.from_mesh(mesh)
    before_count = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-4)
    assert before_count == len(bm.verts), "Merge by Distance still finds vertices to remove"
    bm.free()

    # Each corner keeps the UV of the source vertex it came from, so a vertex
    # shared by faces that map it differently has several UVs, none invented.
    uv_data = mesh.uv_layers["UVMap"].data
    per_vertex = {}
    for loop in mesh.loops:
        position = tuple(round(c, 3) for c in mesh.vertices[loop.vertex_index].co)
        uv = tuple(round(c, 4) for c in uv_data[loop.index].uv)
        assert (position, uv) in source, f"corner {position} has UV {uv}, which the source never gives it"
        per_vertex.setdefault(loop.vertex_index, set()).add(uv)
    assert any(len(uvs) > 1 for uvs in per_vertex.values()), "no welded vertex kept more than one UV (seams lost)"
    print(f"vertices welded: {source_vertices} source vertices -> 8, per-corner UVs kept - OK")


def check_library_warnings_are_collected():
    """Warnings the openskp library logs while it builds a scene (a texture
    format a scene cannot carry, ...) are collected during the import, each
    distinct message once, and nothing else: not info messages, not other
    loggers' warnings, and the handler is gone afterwards."""
    import logging

    library = logging.getLogger("openskp")
    handlers_before = list(library.handlers)
    with import_skp._capture_library_warnings() as messages:
        logging.getLogger("openskp.scene").warning("Texture of material %r is TIFF", "Wall")
        logging.getLogger("openskp.instanced_scene").warning("Texture of material %r is TIFF", "Wall")
        logging.getLogger("openskp.scene").warning("Texture of material %r is TGA", "Floor")
        logging.getLogger("openskp.scene").info("not a warning")
        logging.getLogger("somebody_else").warning("not openskp's")
    assert messages == ["Texture of material 'Wall' is TIFF", "Texture of material 'Floor' is TGA"], messages
    assert list(library.handlers) == handlers_before, "the capture handler was not removed"
    print("library warnings: collected once each, nothing else, handler removed - OK")


def check_import_reports_no_warnings_for_a_clean_file():
    stats = import_skp.import_skp(os.path.join(FIXTURES_DIR, "capilla_quiroz_v17.skp"))
    assert stats["warnings"] == [], stats["warnings"]
    assert stats["unread_root_entities"] == 0, stats
    print("a file with PNG/JPEG textures: no warnings - OK")


def check_partial_read_warning_is_reported():
    """When openskp reports a classic file whose root entity list stopped early
    (openskp.legacy warning or InstancedScene.unread_root_entities, openskp#552),
    import_skp collects the warning even if the root logger was silenced, orders
    it ahead of texture notes so Blender's one-line status bar shows it, and
    records unread_root_entities in stats."""
    import functools
    import logging
    import tempfile

    import openskp

    orig_build = openskp.SkpFile.build_instanced_scene
    root_logger = logging.getLogger()
    saved_root_level = root_logger.level
    try:
        # 1. openskp.legacy warning during build_instanced_scene with root logger
        # silenced to ERROR, on a file that also produces a GIF texture warning:
        # partial-read warning is still captured and placed first in stats["warnings"].
        @functools.wraps(orig_build)
        def build_with_legacy_warning(self, *args, **kwargs):
            scene = orig_build(self, *args, **kwargs)
            logging.getLogger("openskp.legacy").warning(
                "Classic .skp root entity list stopped early after 4 entities "
                "(6 unread): this file was read only in part"
            )
            return scene

        root_logger.setLevel(logging.ERROR)
        openskp.SkpFile.build_instanced_scene = build_with_legacy_warning
        stats = import_skp.import_skp(os.path.join(FIXTURES_DIR, "texture_gif_2021.skp"))
        assert stats["unread_root_entities"] == 6, stats
        assert len(stats["warnings"]) >= 2, stats["warnings"]
        assert "read only in part" in stats["warnings"][0], stats["warnings"]
        assert any("GIF" in m for m in stats["warnings"][1:]), stats["warnings"]

        # 2. Attribute-only signal (hasattr(scene, "unread_root_entities")).
        @functools.wraps(orig_build)
        def build_with_scene_attr(self, *args, **kwargs):
            scene = orig_build(self, *args, **kwargs)
            scene.unread_root_entities = 4
            return scene

        openskp.SkpFile.build_instanced_scene = build_with_scene_attr
        stats = import_skp.import_skp(os.path.join(FIXTURES_DIR, "SU_File.skp"))
        assert stats["unread_root_entities"] == 4, stats
        assert len(stats["warnings"]) == 1 and "read only in part" in stats["warnings"][0], stats

        # 3. End-to-end check once a post-openskp#552 wheel is bundled.
        openskp.SkpFile.build_instanced_scene = orig_build
        root_logger.setLevel(saved_root_level)
        if hasattr(openskp.SkpModel, "unread_root_entities"):
            builder = openskp.create()
            builder.add_face([(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0)])
            builder.add_text("First", (0, 0, 0))
            builder.add_text("Broken", (5, 0, 0))
            builder.add_face([(20, 0, 0), (30, 0, 0), (30, 10, 0), (20, 10, 0)])
            raw = bytearray(builder.to_bytes())
            at = raw.find(b"\x01\xff\xfe\xff\x06B\x00")
            assert at >= 0
            raw[at : at + 4] = b"\x00\x00\x00\x00"
            with tempfile.TemporaryDirectory() as tmp:
                broken = os.path.join(tmp, "partial.skp")
                with open(broken, "wb") as f:
                    f.write(raw)
                stats = import_skp.import_skp(broken)
                assert stats["unread_root_entities"] > 0, stats
                assert stats["warnings"] and "read only in part" in stats["warnings"][0], stats
    finally:
        openskp.SkpFile.build_instanced_scene = orig_build
        root_logger.setLevel(saved_root_level)
    print("partial-read warning: captured, ordered first, unread count recorded - OK")


def _library_keeps_original_textures():
    """Whether the openskp library in use can keep a TIFF, TGA or GIF texture in a scene (1.6.0 and
    newer, which is what this add-on bundles); the texture fixtures import differently without it."""
    import inspect

    import openskp

    return "original_textures" in inspect.signature(openskp.SkpFile.build_instanced_scene).parameters


def check_a_skipped_texture_is_reported_by_the_import():
    """With an openskp that logs a texture it leaves out but cannot keep it (a build between 1.5.0
    and 1.6.0; the bundled library is past that), a TIFF-textured material comes in without its texture and the import says
    so (blender-openskp#17). Neither an older library (nothing logged) nor a newer one (which
    keeps the texture, see check_original_textures_are_loaded) has anything to check here."""
    import openskp.scene

    if not hasattr(openskp.scene, "_warn_skipped_texture"):
        print("skipped-texture warning: the bundled openskp does not log it yet - not checked")
        return
    if _library_keeps_original_textures():
        print("skipped-texture warning: this openskp keeps the texture instead - not checked")
        return
    stats = import_skp.import_skp(os.path.join(FIXTURES_DIR, "texture_tiff_2021.skp"))
    assert stats["textures_loaded"] == 0
    assert len(stats["warnings"]) == 1, stats["warnings"]
    assert "Tex_TIF" in stats["warnings"][0] and "TIFF" in stats["warnings"][0], stats["warnings"]
    print("a TIFF texture: skipped, and the import reports it - OK")


def check_original_textures_are_loaded():
    """A TIFF or TGA texture comes in as a real, packed Blender image wired into the material,
    where it used to be dropped (blender-openskp#17). Needs an openskp that can keep it."""
    if not _library_keeps_original_textures():
        print("original textures: the bundled openskp cannot keep them yet - not checked")
        return
    for name in ("texture_tiff_2021.skp", "texture_tga_2021.skp"):
        before = set(bpy.data.images)
        stats = import_skp.import_skp(os.path.join(FIXTURES_DIR, name))
        assert stats["textures_loaded"] == 1 and stats["warnings"] == [], (name, stats)
        (image,) = set(bpy.data.images) - before
        assert tuple(image.size) == (64, 64), (name, tuple(image.size))
        assert image.packed_file is not None, f"{name}: the image is not packed into the .blend"
        shown = [
            mat.name
            for mat in bpy.data.materials
            if mat.use_nodes and any(node.type == "TEX_IMAGE" and node.image == image for node in mat.node_tree.nodes)
        ]
        assert shown, f"{name}: no material shows the loaded image"
    # A GIF is kept by the library but Blender opens it as a movie: no image, and the import says so.
    before = set(bpy.data.images)
    stats = import_skp.import_skp(os.path.join(FIXTURES_DIR, "texture_gif_2021.skp"))
    assert stats["textures_loaded"] == 0 and set(bpy.data.images) == before
    assert len(stats["warnings"]) == 1 and "GIF" in stats["warnings"][0], stats["warnings"]
    print("original textures: TIFF and TGA come in as packed images in the material, a GIF is reported - OK")


def _world_matrices(objects):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    return sorted(
        tuple(round(v, 4) for row in o.evaluated_get(depsgraph).matrix_world for v in row)
        for o in objects if o.instance_type == "COLLECTION"
    )


def check_hierarchy(fixture_name, expected_groups):
    """keep_hierarchy keeps SketchUp's nesting as parented Empties without
    moving anything: the same placements end up at the same world matrices
    as the flat import, and every placement inside a group is parented
    under it."""
    path = os.path.join(FIXTURES_DIR, fixture_name)
    before = set(bpy.data.objects)
    flat_stats = import_skp.import_skp(path)
    flat = set(bpy.data.objects) - before
    before = set(bpy.data.objects)
    stats = import_skp.import_skp(path, keep_hierarchy=True)
    nested = set(bpy.data.objects) - before

    assert stats["placements"] == flat_stats["placements"]
    assert stats["groups"] == expected_groups, f"{fixture_name}: {stats['groups']} groups, expected {expected_groups}"
    assert _world_matrices(nested) == _world_matrices(flat), f"{fixture_name}: hierarchy moved placements"
    parented = [o for o in nested if o.parent is not None]
    if expected_groups:
        assert parented, f"{fixture_name}: nothing parented"
        assert all(o.parent in nested and o.parent.instance_type == "NONE" for o in parented)
    print(f"{fixture_name}: hierarchy kept ({stats['groups']} groups, {len(parented)} parented), "
          "same world placements as flat - OK")
    return nested


def check_nesting_tree():
    """nesting_2021.skp: groups Outer > Shelf > Inner, each with its own faces."""
    nested = check_hierarchy("nesting_2021.skp", expected_groups=2)
    by_name = {re.sub(r"\.\d{3}$", "", o.name): o for o in nested if o.type == "EMPTY"}
    groups = [o for o in nested if o.type == "EMPTY" and o.instance_type == "NONE"]
    outer_group = next(o for o in groups if o.parent is None)
    shelf_group = next(o for o in groups if o.parent == outer_group)
    assert re.sub(r"\.\d{3}$", "", outer_group.name) == "Outer"
    assert re.sub(r"\.\d{3}$", "", shelf_group.name) == "Shelf"
    inner = by_name["Inner"]
    assert inner.parent == shelf_group and inner.instance_type == "COLLECTION"
    assert inner["sketchup_definition"] == "Group"
    print("nesting_2021.skp: Outliner tree is Outer > Shelf > Inner - OK")




def check_missing_bundled_library_is_a_clear_error():
    """A bundled library that cannot be imported (what Blender 4.2.0-4.2.3 causes by
    installing the cp313 wheel into Python 3.11) stops the import with an actionable
    message, instead of openskp falling back to fan triangulation and importing
    meshes with their holes filled in."""
    import_skp.check_bundled_libraries()  # fine when everything loads
    saved = sys.modules.get("mapbox_earcut")
    sys.modules["mapbox_earcut"] = None  # makes `import mapbox_earcut` raise ImportError
    try:
        for call in (
            import_skp.check_bundled_libraries,
            lambda: import_skp.import_skp(os.path.join(FIXTURES_DIR, "SU_File.skp")),
        ):
            try:
                call()
            except RuntimeError as exc:
                assert "mapbox_earcut" in str(exc) and "4.2.4" in str(exc), str(exc)
            else:
                raise AssertionError("expected a RuntimeError naming mapbox_earcut and Blender 4.2.4")
    finally:
        if saved is None:
            del sys.modules["mapbox_earcut"]
        else:
            sys.modules["mapbox_earcut"] = saved
    print("missing bundled library: clear error naming it and Blender 4.2.4 - OK")


def check_solid_view_shows_painted_side():
    """In Solid viewport shading (`Material.diffuse_color`), a face painted on
    only one side shows that painted side's colour even when the painted side
    faces into the mesh (e.g. a box wall painted only on the inside, with its
    outward polygon side unpainted, blender-openskp#16), while keeping outward
    polygon normals and the EEVEE/Cycles front/back shader mix intact. When
    both sides are painted (or neither is), Solid shading shows the polygon's
    front (outward) paint."""
    import tempfile

    import mathutils
    import openskp

    s = 100.0
    sm = s * 0.0254

    def P(x, y, z):
        return (x * s, y * s, z * s)

    aluminum_rgb = (180, 190, 205)
    blue_rgb = (20, 90, 220)
    orange_rgb = (240, 120, 20)

    def linear_rgba(rgb):
        return tuple(round(c, 3) for c in (*import_skp.srgb_to_linear(tuple(v / 255 for v in rgb)), 1.0))

    with tempfile.TemporaryDirectory() as tmp:
        skp = os.path.join(tmp, "solid_sides_box.skp")
        builder = openskp.SkpBuilder()
        alum = builder.add_material("Metal Aluminum Anodized", (*aluminum_rgb, 255))
        blue = builder.add_material("Blue", (*blue_rgb, 255))
        orange = builder.add_material("Orange", (*orange_rgb, 255))

        # +X wall: wound INWARD, painted Aluminum on its SketchUp front (inside)
        # and unpainted on its SketchUp back (outside) - reproduces box.skp in #16.
        builder.add_face([P(1, 0, 1), P(1, 1, 1), P(1, 1, 0), P(1, 0, 0)], material=alum)
        # -Y wall: wound outward, painted Blue on outside, unpainted inside.
        builder.add_face([P(0, 0, 0), P(1, 0, 0), P(1, 0, 1), P(0, 0, 1)], material=blue)
        # +Y wall: wound outward, painted Blue outside and Orange inside.
        builder.add_face([P(1, 1, 0), P(0, 1, 0), P(0, 1, 1), P(1, 1, 1)], material=blue, back_material=orange)
        # -X wall: wound INWARD, painted Orange on front (inside) and Blue on back (outside).
        builder.add_face([P(0, 1, 0), P(0, 1, 1), P(0, 0, 1), P(0, 0, 0)], material=orange, back_material=blue)
        # Unpainted bottom (-Z) and top (+Z) close the box.
        builder.add_face([P(0, 0, 0), P(0, 1, 0), P(1, 1, 0), P(1, 0, 0)])
        builder.add_face([P(0, 0, 1), P(1, 0, 1), P(1, 1, 1), P(0, 1, 1)])
        builder.save(skp)

        before = set(bpy.data.objects)
        import_skp.import_skp(skp)
        (obj,) = [o for o in set(bpy.data.objects) - before if o.type == "MESH"]

    mesh = obj.data
    assert len(mesh.polygons) == 12, f"expected 6 quads x 2 triangles, got {len(mesh.polygons)}"
    centre = sum((v.co for v in mesh.vertices), mathutils.Vector()) / len(mesh.vertices)

    wall_colors = {"+X": set(), "-X": set(), "-Y": set(), "+Y": set()}
    plus_x_mats = set()
    for poly in mesh.polygons:
        assert poly.normal.dot(poly.center - centre) > 0, (
            f"polygon at {tuple(poly.center)} has inward normal {tuple(poly.normal)}"
        )
        mat = mesh.materials[poly.material_index]
        diffuse = tuple(round(c, 3) for c in mat.diffuse_color)
        # Local mesh coordinates are glTF Y-up metres: (x_m, z_m, -y_m).
        if abs(poly.center.x - sm) < 1e-4:
            wall_colors["+X"].add(diffuse)
            plus_x_mats.add(mat)
        elif abs(poly.center.x) < 1e-4:
            wall_colors["-X"].add(diffuse)
        elif abs(poly.center.z) < 1e-4:
            wall_colors["-Y"].add(diffuse)
        elif abs(poly.center.z + sm) < 1e-4:
            wall_colors["+Y"].add(diffuse)

    assert wall_colors["+X"] == {linear_rgba(aluminum_rgb)}, (
        f"+X wall (painted inside, unpainted outside) should show Aluminum in Solid view, got {wall_colors['+X']}"
    )
    assert wall_colors["-Y"] == {linear_rgba(blue_rgb)}, f"-Y wall got {wall_colors['-Y']}"
    assert wall_colors["+Y"] == {linear_rgba(blue_rgb)}, f"+Y wall (both painted) got {wall_colors['+Y']}"
    assert wall_colors["-X"] == {linear_rgba(blue_rgb)}, f"-X wall (both painted, wound inward) got {wall_colors['-X']}"

    # On the +X wall, EEVEE/Cycles still keep unpainted on front (outside) and Aluminum on back (inside).
    (plus_x_mat,) = plus_x_mats
    assert plus_x_mat.name.endswith("| Metal Aluminum Anodized"), plus_x_mat.name
    mix = next(n for n in plus_x_mat.node_tree.nodes if n.type == "MIX_SHADER")
    back_bsdf = mix.inputs[2].links[0].from_node
    assert tuple(round(c, 3) for c in back_bsdf.inputs["Base Color"].default_value) == linear_rgba(aluminum_rgb)

    # When neither side is painted (two unnamed fallback colours, as openskp#533 will emit for
    # style front/back colours), Solid shading shows the polygon's front colour.
    unpainted_gltf = [
        {"pbrMetallicRoughness": {"baseColorFactor": [0.95, 0.95, 0.95, 1.0]}},
        {"pbrMetallicRoughness": {"baseColorFactor": [0.9, 0.4, 0.1, 1.0]}},
    ]
    unpainted_pair = import_skp._get_or_build_pair_material(0, 1, unpainted_gltf, {}, [], {})
    want_front = tuple(round(c, 3) for c in (*import_skp.srgb_to_linear((0.95, 0.95, 0.95)), 1.0))
    assert tuple(round(c, 3) for c in unpainted_pair.diffuse_color) == want_front
    print("Solid view painted side: painted inside/outside/both/neither all show expected colour - OK")


def check_concave_surfaces_keep_one_orientation():
    """A torus painted on the outside only becomes one front/back material
    with every polygon facing outward, whichever side SketchUp calls the
    front: the inner ring faces the mesh's centre, which used to turn it
    inside out and give the mesh a second, reversed material (#33). A single
    flat face keeps SketchUp's own front."""
    import math
    import tempfile

    import bmesh
    import openskp

    segments, sides = 16, 10
    big, small = 100.0, 30.0

    def point(i, j):
        u, v = 2 * math.pi * (i % segments) / segments, 2 * math.pi * (j % sides) / sides
        r = big + small * math.cos(v)
        return (r * math.cos(u), r * math.sin(u), small * math.sin(v))

    quads = [
        [point(i, j), point(i + 1, j), point(i + 1, j + 1), point(i, j + 1)]
        for i in range(segments)
        for j in range(sides)
    ]

    def import_faces(faces, paint_front):
        with tempfile.TemporaryDirectory() as tmp:
            image = os.path.join(tmp, "grid.png")
            with open(image, "wb") as f:
                f.write(_png(2, 2, [[(255, 0, 0), (0, 255, 0)], [(0, 0, 255), (255, 255, 0)]]))
            skp = os.path.join(tmp, "shape.skp")
            builder = openskp.SkpBuilder()
            paint = builder.add_texture_material("Blinds", image, 10, 10)
            for face in faces:
                builder.add_face(face, **{"material" if paint_front else "back_material": paint})
            builder.save(skp)
            before = set(bpy.data.objects)
            import_skp.import_skp(skp)
        (obj,) = [o for o in set(bpy.data.objects) - before if o.type == "MESH"]
        return obj.data

    for label, faces, paint_front in (
        ("front outside", quads, True),
        ("front inside, back painted", [q[::-1] for q in quads], False),
    ):
        mesh = import_faces(faces, paint_front)
        names = [m.name for m in mesh.materials]
        assert len(names) == 1 and names[0].startswith("Blinds |"), f"torus ({label}): materials {names}"
        bm = bmesh.new()
        bm.from_mesh(mesh)
        turned = [e for e in bm.edges if len(e.link_loops) == 2 and e.link_loops[0].vert == e.link_loops[1].vert]
        assert all(len(e.link_faces) == 2 for e in bm.edges), f"torus ({label}): not closed"
        assert not turned, f"torus ({label}): {len(turned)} edges where the surface turns over"
        assert bm.calc_volume(signed=True) > 0, f"torus ({label}): faces inward"
        bm.free()
        # The paint is on the polygons' front, so it uses the main UV layer.
        assert [layer.name for layer in mesh.uv_layers] == ["UVMap"], list(mesh.uv_layers.keys())

    for flat in ([(0, 0, 7), (9, 0, 7), (9, 5, 7), (0, 5, 7)], [(0, 5, 7), (9, 5, 7), (9, 0, 7), (0, 0, 7)]):
        mesh = import_faces([flat], True)
        names = [m.name for m in mesh.materials]
        assert len(names) == 1 and names[0].startswith("Blinds |"), f"flat face: {names}"
    print("concave surfaces: one orientation, one front/back material - OK")


def check_scene_cameras_for_fixture(fixture_name, new_objects, stats):
    """Cross-validates imported Blender Camera objects against SkpModel.pages
    for every fixture (#26): position, view direction (-Z), up vector (+Y),
    perspective vertical FOV vs orthographic vertical height, 'Scenes'
    collection membership, and active scene camera.

    When the bundled openskp wheel does not yet expose ``InstancedScene.pages``
    (openskp <= 1.6.0, before openskp#593), verifies that ``import_skp()`` left
    ``scenes_imported == 0`` without parsing twice, then passes
    ``SkpFile.parse().pages`` to ``_import_scene_cameras`` directly so the
    camera construction stays covered today."""
    import math
    import mathutils
    import openskp

    path = os.path.join(FIXTURES_DIR, fixture_name)
    has_scene_pages = hasattr(openskp.SkpFile.open(path).build_instanced_scene(), "pages")
    pages = openskp.SkpFile.open(path).parse().pages
    valid_pages = [
        p for p in pages
        if p.eye is not None and p.target is not None
        and (mathutils.Vector(p.target) - mathutils.Vector(p.eye)).length > 1e-12
    ]

    root_name = os.path.splitext(fixture_name)[0]
    root_coll = bpy.data.collections.get(root_name)
    assert root_coll is not None

    if has_scene_pages:
        assert stats["scenes_imported"] == len(valid_pages), (
            f"{fixture_name}: scenes_imported={stats['scenes_imported']}, expected {len(valid_pages)}"
        )
        cam_objs = [o for o in new_objects if o.type == "CAMERA"]
    else:
        assert stats["scenes_imported"] == 0, (
            f"{fixture_name}: bundled scene has no .pages, expected scenes_imported=0, got {stats['scenes_imported']}"
        )
        assert not [o for o in new_objects if o.type == "CAMERA"], (
            f"{fixture_name}: unexpected cameras created without InstancedScene.pages"
        )
        before_direct = set(bpy.data.objects)
        created = import_skp._import_scene_cameras(pages, root_coll, bpy.context)
        assert created == len(valid_pages), (fixture_name, created, len(valid_pages))
        cam_objs = [o for o in (set(bpy.data.objects) - before_direct) if o.type == "CAMERA"]

    scenes_colls = [c for c in root_coll.children if re.sub(r"\.\d{3}$", "", c.name) == "Scenes"]

    if not valid_pages:
        assert not scenes_colls, f"{fixture_name}: unexpected 'Scenes' collection when file has no scenes"
        return

    assert len(scenes_colls) == 1, f"{fixture_name}: expected 1 'Scenes' collection, got {scenes_colls}"
    scenes_coll = scenes_colls[0]
    assert len(cam_objs) == len(valid_pages), (fixture_name, len(cam_objs), len(valid_pages))
    assert set(scenes_coll.objects) == set(cam_objs), (fixture_name, list(scenes_coll.objects))

    by_base_name = {re.sub(r"\.\d{3}$", "", o.name): o for o in cam_objs}
    for idx, page in enumerate(valid_pages):
        want_name = page.name or f"Scene {idx + 1}"
        assert want_name in by_base_name, f"{fixture_name}: missing camera {want_name!r} in {list(by_base_name)}"
        cam_obj = by_base_name[want_name]
        cam_data = cam_obj.data
        assert cam_data.sensor_fit == "VERTICAL"

        eye_m = mathutils.Vector(tuple(c * 0.0254 for c in page.eye))
        target_m = mathutils.Vector(tuple(c * 0.0254 for c in page.target))
        want_dir = (target_m - eye_m).normalized()
        up_raw = mathutils.Vector(page.up) if page.up is not None else mathutils.Vector((0.0, 0.0, 1.0))
        want_right = want_dir.cross(up_raw).normalized()
        want_up = (-want_dir).cross(want_right).normalized()

        got_pos = cam_obj.matrix_world.translation
        got_dir = (cam_obj.matrix_world.to_3x3() @ mathutils.Vector((0.0, 0.0, -1.0))).normalized()
        got_up = (cam_obj.matrix_world.to_3x3() @ mathutils.Vector((0.0, 1.0, 0.0))).normalized()
        assert (got_pos - eye_m).length < 1e-5, f"{fixture_name}/{want_name}: pos {tuple(got_pos)} != {tuple(eye_m)}"
        assert (got_dir - want_dir).length < 1e-5, (
            f"{fixture_name}/{want_name}: dir {tuple(got_dir)} != {tuple(want_dir)}"
        )
        assert (got_up - want_up).length < 1e-5, f"{fixture_name}/{want_name}: up {tuple(got_up)} != {tuple(want_up)}"

        vf = cam_data.view_frame(scene=bpy.context.scene)
        v_span = max(v.y for v in vf) - min(v.y for v in vf)
        if page.parallel:
            assert cam_data.type == "ORTHO", (fixture_name, want_name, cam_data.type)
            want_h = page.ortho_height * 0.0254
            assert abs(cam_data.ortho_scale - want_h) < 1e-5, (cam_data.ortho_scale, want_h)
            assert abs(v_span - want_h) < 1e-5, (v_span, want_h)
        else:
            assert cam_data.type == "PERSP", (fixture_name, want_name, cam_data.type)
            assert abs(math.degrees(cam_data.angle) - page.fov) < 1e-4, (math.degrees(cam_data.angle), page.fov)
            z_depth = abs(vf[0].z)
            got_fov = math.degrees(2.0 * math.atan((v_span * 0.5) / z_depth))
            assert abs(got_fov - page.fov) < 1e-4, (got_fov, page.fov)

        if idx == 0:
            assert bpy.context.scene.camera == cam_obj, (
                f"{fixture_name}: active camera is {bpy.context.scene.camera}, expected {cam_obj}"
            )
    print(f"{fixture_name}: {len(valid_pages)} scene camera(s) match eye/target/up and frustum - OK")


def check_cameraless_page_is_skipped():
    """A SketchUp 2021+ scene that saves no camera (eye=None, openskp#458) is
    skipped cleanly without creating a broken camera or failing import, and
    an InstancedScene carrying .pages wires straight through import_skp() with
    zero SkpFile.parse() calls."""
    import openskp

    no_cam = openskp.model.Page(name="TagsOnly", eye=None, target=None)
    assert import_skp._build_camera_from_page(no_cam, 0) is None
    coincident = openskp.model.Page(name="Degenerate", eye=(10.0, 10.0, 10.0), target=(10.0, 10.0, 10.0))
    assert import_skp._build_camera_from_page(coincident, 0) is None

    path = os.path.join(FIXTURES_DIR, "two_scenes_2021.skp")
    expected_pages = openskp.SkpFile.open(path).parse().pages
    orig_build = openskp.SkpFile.build_instanced_scene
    orig_parse = openskp.SkpFile.parse
    parse_calls = 0

    def build_with_pages(self, *args, **kwargs):
        scene = orig_build(self, *args, **kwargs)
        scene.pages = expected_pages
        return scene

    def count_parse(self, *args, **kwargs):
        nonlocal parse_calls
        parse_calls += 1
        return orig_parse(self, *args, **kwargs)

    openskp.SkpFile.build_instanced_scene = build_with_pages
    openskp.SkpFile.parse = count_parse
    try:
        before = set(bpy.data.objects)
        stats = import_skp.import_skp(path)
        created_cams = [o for o in (set(bpy.data.objects) - before) if o.type == "CAMERA"]
    finally:
        openskp.SkpFile.build_instanced_scene = orig_build
        openskp.SkpFile.parse = orig_parse

    assert parse_calls == 0, f"expected 0 parse() calls, got {parse_calls}"
    assert stats["scenes_imported"] == 2 and len(created_cams) == 2, (stats["scenes_imported"], len(created_cams))
    print("cameraless/coincident Page skipped and InstancedScene.pages wired with 0 parse() calls - OK")


def _base_name(name):
    return re.sub(r"\.\d{3}$", "", name)


def _spline_world_endpoints(curve_obj, spline_idx=0):
    import mathutils

    sp = curve_obj.data.splines[spline_idx]
    mw = curve_obj.matrix_world
    p0 = mw @ mathutils.Vector(sp.points[0].co[:3])
    p1 = mw @ mathutils.Vector(sp.points[1].co[:3])
    return p0, p1


def _scan_fixture_no_private_paths(path):
    """Scans raw bytes (ASCII + UTF-16) and every ZIP entry (including
    meta/meta.dat) for host paths, usernames, temp dirs, and email addresses."""
    import zipfile

    blobs = [("RAW", open(path, "rb").read())]
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                blobs.append((name, zf.read(name)))
    email_re = re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    bad_tokens = (
        "Users", "Admin", "Devs", "lanes", ".gemini",
        "Temp\\", "Temp/", "scratch", "AppData", "C:\\", "G:\\",
    )
    for entry_name, data in blobs:
        for tok in bad_tokens:
            for enc in ("ascii", "utf-16-le", "utf-16-be"):
                needle = tok.encode(enc)
                assert needle.lower() not in data.lower(), (path, entry_name, tok, enc)
        assert not email_re.search(data), (path, entry_name, email_re.search(data).group(0))


def check_annotations_import():
    """Verifies SDK-written annotations_2017.skp and annotations_2021.skp (#27):
    - Raw + ZIP-entry path/email scan on annotations_2017.skp & annotations_2021.skp,
    - Positioned leader text ('Apex label' at (24, 14, 10)" anchored to (18, 6, 0)"),
    - Positionless screen text ('Title block note', including openskp#623
      leader_type == 0 / screen_position with (0,0,0) points) skipped & counted,
    - Offset rule: zero offset with openskp 1.6.0 (drawn along a-b), explicit
      offset_vector used when the library provides it,
    - Axis-constrained 5BD1 VFF dimensions and unresolved endpoints skipped & counted,
    - '<>' placeholder replacement in custom dimension text,
    - Error isolation across skp.parse(), _extract_model_annotations, and per-item
      creation without cancelling the geometry import."""
    import struct
    import tempfile
    import zipfile
    import mathutils
    import openskp

    path = os.path.join(FIXTURES_DIR, "annotations_2017.skp")
    path_2021 = os.path.join(FIXTURES_DIR, "annotations_2021.skp")
    _scan_fixture_no_private_paths(path)
    _scan_fixture_no_private_paths(path_2021)

    has_offset_vec = "offset_vector" in getattr(openskp.model.Dimension, "__dataclass_fields__", {})

    # 1. Hierarchy mode (keep_hierarchy=True, import_annotations=True)
    colls_before = set(bpy.data.collections)
    objs_before = set(bpy.data.objects)
    stats = import_skp.import_skp(path, keep_hierarchy=True, import_annotations=True)
    new_colls = set(bpy.data.collections) - colls_before
    new_objs = set(bpy.data.objects) - objs_before
    bpy.context.view_layer.update()

    assert stats["texts_imported"] == 1, stats
    assert stats["texts_skipped_no_position"] == 1, stats
    assert stats["dimensions_imported"] == 2, stats
    assert any("without a 3D position" in w for w in stats["warnings"]), stats["warnings"]

    root_coll = next(
        c for c in new_colls
        if _base_name(c.name) == "annotations_2017" and c in set(bpy.context.scene.collection.children)
    )
    tag_colls = {_base_name(c.name): c for c in root_coll.children}
    assert "Layer0" in tag_colls and "Framing" in tag_colls, list(tag_colls)

    layer0_annots = [c for c in tag_colls["Layer0"].children if _base_name(c.name) == "Annotations"]
    framing_annots = [c for c in tag_colls["Framing"].children if _base_name(c.name) == "Annotations"]
    assert len(layer0_annots) == 1 and len(framing_annots) == 1, (layer0_annots, framing_annots)

    l0_objs = list(layer0_annots[0].objects)
    fr_objs = list(framing_annots[0].objects)

    # Positioned leader text ("Apex label") on Layer0; screen text skipped
    l0_texts = {o.data.body: o for o in l0_objs if o.type == "FONT" and o.get("sketchup_annotation") != "dimension_text"}
    assert set(l0_texts) == {"Apex label"}, set(l0_texts)

    apex = l0_texts["Apex label"]
    assert apex["sketchup_annotation"] == "leader_text"
    assert apex.parent is None
    want_apex_pos = mathutils.Vector((24.0 * 0.0254, 14.0 * 0.0254, 10.0 * 0.0254))
    assert (apex.matrix_world.translation - want_apex_pos).length < 1e-5, (
        tuple(apex.matrix_world.translation), tuple(want_apex_pos)
    )
    want_anchor = mathutils.Vector((18.0 * 0.0254, 6.0 * 0.0254, 0.0))
    got_anchor = mathutils.Vector(tuple(apex["sketchup_anchor_m"]))
    assert (got_anchor - want_anchor).length < 1e-5, (tuple(got_anchor), tuple(want_anchor))

    # Root 36" (0.9144 m) linear dimension on Layer0: on segment with openskp 1.6.0
    l0_dim_lines = [o for o in l0_objs if o.type == "CURVE" and o.get("sketchup_annotation") == "dimension_line"]
    l0_dim_texts = [o for o in l0_objs if o.type == "FONT" and o.get("sketchup_annotation") == "dimension_text"]
    assert len(l0_dim_lines) == 1 and len(l0_dim_texts) == 1, (l0_dim_lines, l0_dim_texts)
    root_line, root_val = l0_dim_lines[0], l0_dim_texts[0]
    assert root_line.parent is None
    assert root_val.parent == root_line
    assert root_val.data.body == "0.9144 m", root_val.data.body
    assert abs(root_line["sketchup_distance_m"] - 0.9144) < 1e-5
    assert abs(root_line["sketchup_distance_inches"] - 36.0) < 1e-4
    d1, d2 = _spline_world_endpoints(root_line, 0)
    assert abs((d2 - d1).length - 0.9144) < 1e-5, ((d2 - d1).length,)
    want_root_mid = mathutils.Vector((18.0 * 0.0254, 0.0, 0.0))
    if not has_offset_vec:
        assert len(root_line.data.splines) == 1
        assert (root_line.matrix_world.translation - want_root_mid).length < 1e-5, tuple(root_line.matrix_world.translation)
        assert (root_val.matrix_world.translation - want_root_mid).length < 1e-5, tuple(root_val.matrix_world.translation)

    # Component 24" (0.6096 m) linear dimension inside "Beam" placed at (40, 10, 0)" on Framing
    fr_dim_lines = [o for o in fr_objs if o.type == "CURVE" and o.get("sketchup_annotation") == "dimension_line"]
    fr_dim_texts = [o for o in fr_objs if o.type == "FONT" and o.get("sketchup_annotation") == "dimension_text"]
    assert len(fr_dim_lines) == 1 and len(fr_dim_texts) == 1, (fr_dim_lines, fr_dim_texts)
    beam_line, beam_val = fr_dim_lines[0], fr_dim_texts[0]
    assert beam_val.parent == beam_line
    assert beam_val.data.body == "0.6096 m", beam_val.data.body
    assert abs(beam_line["sketchup_distance_m"] - 0.6096) < 1e-5
    assert abs(beam_line["sketchup_distance_inches"] - 24.0) < 1e-4
    bd1, bd2 = _spline_world_endpoints(beam_line, 0)
    assert abs((bd2 - bd1).length - 0.6096) < 1e-5, ((bd2 - bd1).length,)
    want_beam_mid = mathutils.Vector((52.0 * 0.0254, 10.0 * 0.0254, 0.0))
    if not has_offset_vec:
        assert (beam_line.matrix_world.translation - want_beam_mid).length < 1e-5, (
            tuple(beam_line.matrix_world.translation), tuple(want_beam_mid)
        )
        assert (beam_val.matrix_world.translation - want_beam_mid).length < 1e-5

    # Follows component hierarchy: parented to the "Beam 1" placement Empty and moves with it
    beam_empty = next(o for o in new_objs if o.type == "EMPTY" and _base_name(o.name) == "Beam 1")
    assert beam_line.parent == beam_empty, (beam_line.parent, beam_empty)
    orig_beam_mid = beam_line.matrix_world.translation.copy()
    delta = mathutils.Vector((1.25, -0.5, 0.75))
    beam_empty.location = beam_empty.location + delta
    bpy.context.view_layer.update()
    assert (beam_line.matrix_world.translation - (orig_beam_mid + delta)).length < 1e-5
    assert (beam_val.matrix_world.translation - (orig_beam_mid + delta)).length < 1e-5

    # 2. Flat mode (keep_hierarchy=False, import_annotations=True)
    objs_before = set(bpy.data.objects)
    stats_flat = import_skp.import_skp(path, keep_hierarchy=False, import_annotations=True)
    flat_objs = set(bpy.data.objects) - objs_before
    bpy.context.view_layer.update()
    assert stats_flat["texts_imported"] == 1 and stats_flat["dimensions_imported"] == 2
    assert stats_flat["texts_skipped_no_position"] == 1
    flat_beam_line = next(
        o for o in flat_objs
        if o.type == "CURVE" and o.get("sketchup_annotation") == "dimension_line"
        and abs(o["sketchup_distance_inches"] - 24.0) < 1e-4
    )
    assert flat_beam_line.parent is None
    if not has_offset_vec:
        assert (flat_beam_line.matrix_world.translation - want_beam_mid).length < 1e-5

    # 3. Real SDK-saved SketchUp 2021 fixture (annotations_2021.skp, zero parse() monkeypatching)
    has_623 = "layer" in getattr(openskp.model.Dimension, "__dataclass_fields__", {})
    objs_before = set(bpy.data.objects)
    stats_2021 = import_skp.import_skp(path_2021, keep_hierarchy=True, import_annotations=True)
    objs_2021 = set(bpy.data.objects) - objs_before
    bpy.context.view_layer.update()
    lines_2021 = [o for o in objs_2021 if o.type == "CURVE" and o.get("sketchup_annotation") == "dimension_line"]
    if has_623:
        assert stats_2021["dimensions_imported"] == 2 and stats_2021["texts_imported"] == 1, stats_2021
        assert stats_2021["texts_skipped_no_position"] == 1, stats_2021
        assert len(lines_2021) == 2
    else:
        assert stats_2021["dimensions_imported"] == 1, stats_2021
        assert stats_2021["vff_component_dimensions_skipped"] == 1, stats_2021
        assert stats_2021["vff_texts_skipped"] == 2, stats_2021
        assert any("2021+ component dimension" in w for w in stats_2021["warnings"]), stats_2021["warnings"]
        assert len(lines_2021) == 1
        assert abs(lines_2021[0]["sketchup_distance_inches"] - 36.0) < 1e-4
        assert (lines_2021[0].matrix_world.translation - want_root_mid).length < 1e-5

    # 3b. Axis-constrained 5BD1 VFF dimensions:
    # - axis=1 (plane X = (1,0,0), parallel to a=(0,0,0)..b=(36,0,0)): kept & imported!
    # - axis=2 (plane Y = (0,1,0), perpendicular to a-b): skipped & counted!
    with tempfile.TemporaryDirectory() as tmp:
        import io

        raw_2021 = open(path_2021, "rb").read()
        pk_off = raw_2021.index(b"PK\x03\x04")
        for axis_code, want_imp, want_skip in ((1, 1, 0), (2, 0, 1)):
            zbuf = io.BytesIO()
            with zipfile.ZipFile(path_2021, "r") as zin, zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zout:
                for item in zin.infolist():
                    data = zin.read(item.filename)
                    if item.filename == "model.dat":
                        needle = b"\xd1\x5b\x04\x00\x00\x00\x00\x00\x00\x00"
                        rpos = data.rfind(needle)
                        assert rpos > 0
                        data = data[:rpos + 6] + struct.pack("<I", axis_code) + data[rpos + 10:]
                    zout.writestr(item.filename, data)
            constrained_vff = os.path.join(tmp, f"constrained_axis{axis_code}_2021.skp")
            with open(constrained_vff, "wb") as f:
                f.write(raw_2021[:pk_off] + zbuf.getvalue())
            if not has_623:
                st_con = import_skp.import_skp(constrained_vff, keep_hierarchy=True, import_annotations=True)
                assert st_con["dimensions_imported"] == want_imp, (axis_code, st_con)
                assert st_con["constrained_dimensions_skipped"] == want_skip, (axis_code, st_con)
                if want_skip:
                    assert any("axis-constrained" in w for w in st_con["warnings"]), st_con["warnings"]

    # 4. capilla_quiroz_v17.skp: with openskp 1.6.0, all 26 dimensions sit directly on a-b (zero offset)
    cap_path = os.path.join(FIXTURES_DIR, "capilla_quiroz_v17.skp")
    objs_before = set(bpy.data.objects)
    stats_cap = import_skp.import_skp(cap_path, keep_hierarchy=True, import_annotations=True)
    cap_objs = set(bpy.data.objects) - objs_before
    bpy.context.view_layer.update()
    if stats_cap["texts_skipped_no_position"]:
        cap_leader_txts = [
            o for o in cap_objs if o.type == "FONT" and o.get("sketchup_annotation") != "dimension_text"
        ]
        assert not cap_leader_txts, cap_leader_txts
    assert stats_cap["dimensions_imported"] == 26, stats_cap
    if not has_offset_vec:
        for dline in (o for o in cap_objs if o.type == "CURVE" and o.get("sketchup_annotation") == "dimension_line"):
            s_m = mathutils.Vector(tuple(dline["sketchup_start_m"]))
            e_m = mathutils.Vector(tuple(dline["sketchup_end_m"]))
            assert len(dline.data.splines) == 1
            assert (dline.matrix_world.translation - (s_m + e_m) * 0.5).length < 1e-5

    # Explicit offset_vector (openskp #621/#623) is applied when present; scalar offset alone gives 0
    dim_scalar_only = import_skp._tag_dimension(
        openskp.model.Dimension(text="", hidden=False, a=(0.0, 20.0, 10.0), b=(-12.0, 20.0, 10.0), offset=-5.0),
        is_vff=False,
    )
    p1_e = mathutils.Vector((0.0, 20.0 * 0.0254, 10.0 * 0.0254))
    p2_e = mathutils.Vector((-12.0 * 0.0254, 20.0 * 0.0254, 10.0 * 0.0254))
    assert import_skp._dimension_offset_vec_m(dim_scalar_only, p1_e, p2_e, None).length == 0.0

    dim_with_ov = import_skp._tag_dimension(
        openskp.model.Dimension(text="Span <> TYP", hidden=False, a=(0.0, 20.0, 10.0), b=(-12.0, 20.0, 10.0), offset=5.0),
        is_vff=False,
        extra={"offset_vector": (0.0, 0.0, -5.0)},
    )
    off_ov = import_skp._dimension_offset_vec_m(dim_with_ov, p1_e, p2_e, None)
    assert abs(off_ov.x) < 1e-6 and abs(off_ov.y) < 1e-6 and abs(off_ov.z - (-5.0 * 0.0254)) < 1e-6, tuple(off_ov)

    # '<>' replacement in dimension text
    assert import_skp._format_dimension_label("<>", 0.9144) == "0.9144 m"
    assert import_skp._format_dimension_label("Span <> TYP", 0.9144) == "Span 0.9144 m TYP"
    assert import_skp._format_dimension_label("Custom", 0.9144) == "Custom"

    # openskp#623 TextEntity contract: placed texts from real files carry
    # screen_position=(0.0, 0.0) and leader_type=2, while screen texts carry
    # point=(0,0,0), label_point=(0,0,0), screen_position=(u,v), leader_type=0.
    from dataclasses import dataclass

    @dataclass
    class _Real623TextEntity:
        text: str = ""
        hidden: bool = False
        layer: str = ""
        point: object = None
        label_point: object = None
        screen_position: object = (0.0, 0.0)
        leader_type: int = 2
        arrow_type: int = 3
        display_leader: bool = True

    parsed_2017 = openskp.SkpFile.open(path).parse()
    real_texts_623 = [
        _Real623TextEntity(
            text=t.text,
            hidden=t.hidden,
            point=t.point if t.point is not None else (0.0, 0.0, 0.0),
            label_point=t.label_point if t.label_point is not None else (0.0, 0.0, 0.0),
            screen_position=(0.0, 0.0) if t.point is not None else (0.15, 0.85),
            leader_type=2 if t.point is not None else 0,
        )
        for t in parsed_2017.root.texts
    ]
    st_scr = {"texts_imported": 0, "texts_skipped_no_position": 0}
    for rt in real_texts_623:
        import_skp._create_blender_text_annotation(rt, None, lambda: layer0_annots[0], False, None, st_scr)
    assert st_scr["texts_imported"] == 1 and st_scr["texts_skipped_no_position"] == 1, st_scr

    # 5. Errors in per-annotation creation, skp.parse(), and _extract_model_annotations never cancel geometry import
    orig_fmt = import_skp._format_dimension_meters
    try:
        def boom_fmt(_dist):
            raise RuntimeError("simulated dimension failure")

        import_skp._format_dimension_meters = boom_fmt
        stats_err = import_skp.import_skp(path, keep_hierarchy=True, import_annotations=True)
        assert stats_err["placements"] == 2, stats_err
        assert stats_err["texts_imported"] == 1, stats_err
        assert stats_err["dimensions_imported"] == 0, stats_err
        assert stats_err["annotation_failures"] == 2, stats_err
        assert sum("failed to create 2 annotation(s)" in w for w in stats_err["warnings"]) == 1, stats_err["warnings"]
    finally:
        import_skp._format_dimension_meters = orig_fmt

    orig_parse = openskp.SkpFile.parse
    try:
        def boom_parse(_self):
            raise RuntimeError("simulated parse crash")

        openskp.SkpFile.parse = boom_parse
        stats_p = import_skp.import_skp(path, keep_hierarchy=True, import_annotations=True)
        assert stats_p["placements"] == 2 and stats_p["annotation_failures"] == 1, stats_p
        assert any("simulated parse crash" in w for w in stats_p["warnings"]), stats_p["warnings"]
    finally:
        openskp.SkpFile.parse = orig_parse

    orig_parse = openskp.SkpFile.parse
    try:
        class BrokenLayersModel:
            @property
            def layers(self):
                raise RuntimeError("simulated annot_model.layers crash")

        openskp.SkpFile.parse = lambda _self: BrokenLayersModel()
        stats_l = import_skp.import_skp(path, keep_hierarchy=True, import_annotations=True)
        assert stats_l["placements"] == 2 and stats_l["annotation_failures"] == 1, stats_l
        assert any("simulated annot_model.layers crash" in w for w in stats_l["warnings"]), stats_l["warnings"]
    finally:
        openskp.SkpFile.parse = orig_parse

    orig_ext = import_skp._extract_model_annotations
    try:
        def boom_ext(*_a, **_kw):
            raise RuntimeError("simulated extract crash")

        import_skp._extract_model_annotations = boom_ext
        stats_x = import_skp.import_skp(path, keep_hierarchy=True, import_annotations=True)
        assert stats_x["placements"] == 2 and stats_x["annotation_failures"] == 1, stats_x
        assert any("simulated extract crash" in w for w in stats_x["warnings"]), stats_x["warnings"]
    finally:
        import_skp._extract_model_annotations = orig_ext

    print("annotations_2017.skp & annotations_2021.skp: texts, dimensions, offset rule, and error isolation - OK")


def _scene_snapshot():
    """Deterministic snapshot of the current Blender scene for verifying that
    ``linked_duplicates=False`` produces the exact same scene as ``main``."""
    bpy.context.view_layer.update()

    def find_lc(lc, coll):
        if lc.collection == coll:
            return lc
        for ch in lc.children:
            got = find_lc(ch, coll)
            if got is not None:
                return got
        return None

    colls = []
    for c in sorted(bpy.data.collections, key=lambda x: x.name):
        lc = find_lc(bpy.context.view_layer.layer_collection, c)
        props = sorted((k, repr(v)) for k, v in c.items() if k != "_RNA_UI")
        colls.append((
            c.name,
            sorted(ch.name for ch in c.children),
            sorted(o.name for o in c.objects),
            bool(c.hide_viewport),
            bool(c.hide_render),
            bool(lc.exclude) if lc is not None else None,
            props,
        ))

    objs = []
    for o in sorted(bpy.data.objects, key=lambda x: x.name):
        mw = tuple(round(float(v), 6) for row in o.matrix_world for v in row)
        props = sorted((k, repr(v)) for k, v in o.items() if k != "_RNA_UI")
        objs.append((
            o.name,
            o.type,
            o.instance_type,
            o.instance_collection.name if o.instance_collection else None,
            o.parent.name if o.parent else None,
            o.data.name if o.data else None,
            mw,
            bool(o.hide_viewport),
            bool(o.hide_render),
            props,
        ))

    meshes = []
    for m in sorted(bpy.data.meshes, key=lambda x: x.name):
        props = sorted((k, repr(v)) for k, v in m.items() if k != "_RNA_UI")
        mats = [
            (mat.name, tuple(round(float(c), 5) for c in mat.diffuse_color)) if mat else None
            for mat in m.materials
        ]
        meshes.append((
            m.name,
            len(m.vertices),
            len(m.edges),
            len(m.polygons),
            mats,
            [uv.name for uv in m.uv_layers],
            props,
        ))

    return colls, objs, meshes


def _reset_empty_scene():
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
    for crv in list(bpy.data.curves):
        bpy.data.curves.remove(crv)
    for img in list(bpy.data.images):
        bpy.data.images.remove(img)


BASELINE_COMMIT = "28248002293a15eee0529ecdc7b88ad9c20c0691"


def check_linked_duplicates_off_matches_main():
    """With ``linked_duplicates`` off (the default), every fixture in both flat
    and ``keep_hierarchy=True`` modes must produce the exact same Blender scene
    and stats as the pinned pre-#35 baseline commit (#35)."""
    import importlib.util
    import subprocess
    import tempfile

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_src = subprocess.check_output(
        ["git", "-C", repo_root, "show", f"{BASELINE_COMMIT}:import_skp.py"],
        stderr=subprocess.PIPE,
    )

    with tempfile.TemporaryDirectory() as tmp:
        main_mod_path = os.path.join(tmp, "_main_import_skp_ref.py")
        with open(main_mod_path, "wb") as f:
            f.write(main_src)
        spec = importlib.util.spec_from_file_location("_main_import_skp_ref", main_mod_path)
        main_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_mod)

        fixture_paths = [
            os.path.join(FIXTURES_DIR, f)
            for f in sorted(os.listdir(FIXTURES_DIR))
            if f.endswith(".skp")
        ]
        ld_fixture = os.path.join(FIXTURES_DIR, "linked_duplicates", "inherited_paint_2021.skp")
        assert os.path.isfile(ld_fixture), ld_fixture
        fixture_paths.append(ld_fixture)

        comparisons = 0
        for fpath in fixture_paths:
            fname = os.path.basename(fpath)
            for kh in (False, True):
                _reset_empty_scene()
                st_main = main_mod.import_skp(fpath, keep_hierarchy=kh)
                snap_main = _scene_snapshot()

                _reset_empty_scene()
                st_curr = import_skp.import_skp(fpath, keep_hierarchy=kh)
                snap_curr = _scene_snapshot()

                assert st_curr == st_main, f"{fname} (kh={kh}): stats {st_curr} != baseline {st_main}"
                assert snap_curr == snap_main, f"{fname} (kh={kh}): scene differs from {BASELINE_COMMIT[:12]}"
                comparisons += 1

        assert comparisons == len(fixture_paths) * 2 and comparisons > 0, comparisons
    print(f"linked_duplicates=False parity: {comparisons} imports match {BASELINE_COMMIT[:12]} identically - OK")


def _evaluated_world_triangles():
    """List of world-space triangle corner triples ``((x,y,z), (x,y,z), (x,y,z))``
    across all evaluated visible mesh instances in the current scene."""
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    tris = []
    for inst in depsgraph.object_instances:
        eval_obj = inst.object
        if eval_obj.type != "MESH":
            continue
        vis_owner = inst.parent.original if inst.is_instance and inst.parent is not None else eval_obj.original
        if not vis_owner.visible_get():
            continue
        mesh = eval_obj.to_mesh()
        try:
            mesh.calc_loop_triangles()
            mw = inst.matrix_world.copy()
            verts = mesh.vertices
            for lt in mesh.loop_triangles:
                pts = tuple(
                    tuple(float(c) for c in (mw @ verts[vi].co))
                    for vi in lt.vertices
                )
                tris.append(pts)
        finally:
            eval_obj.to_mesh_clear()
    return tris


def _assert_same_world_triangles(tris_a, tris_b, tol_m=1e-5, label=""):
    """Verifies identical triangle count and <= ``tol_m`` (10 um) bidirectional
    Hausdorff match on both triangle centroids and corner vertices (immune to
    float32 decimal rounding boundaries)."""
    import bisect

    assert len(tris_a) == len(tris_b), f"{label}: triangle count {len(tris_a)} != {len(tris_b)}"
    if not tris_a:
        return

    def check_cloud(pts_src, pts_dst, kind):
        dst = sorted(pts_dst)
        xs = [p[0] for p in dst]
        worst = 0.0
        for px, py, pz in pts_src:
            lo = bisect.bisect_left(xs, px - tol_m * 2.0)
            hi = bisect.bisect_right(xs, px + tol_m * 2.0)
            best = float("inf")
            for qx, qy, qz in dst[lo:hi]:
                d2 = (px - qx) ** 2 + (py - qy) ** 2 + (pz - qz) ** 2
                if d2 < best:
                    best = d2
            d = best ** 0.5
            if d > worst:
                worst = d
        assert worst <= tol_m, f"{label} ({kind}): max world deviation {worst:.2e} m > {tol_m:.2e} m"

    verts_a = {p for tri in tris_a for p in tri}
    verts_b = {p for tri in tris_b for p in tri}
    cents_a = [
        ((t[0][0] + t[1][0] + t[2][0]) / 3.0, (t[0][1] + t[1][1] + t[2][1]) / 3.0, (t[0][2] + t[1][2] + t[2][2]) / 3.0)
        for t in tris_a
    ]
    cents_b = [
        ((t[0][0] + t[1][0] + t[2][0]) / 3.0, (t[0][1] + t[1][1] + t[2][1]) / 3.0, (t[0][2] + t[1][2] + t[2][2]) / 3.0)
        for t in tris_b
    ]
    check_cloud(verts_a, verts_b, "verts a->b")
    check_cloud(verts_b, verts_a, "verts b->a")
    check_cloud(cents_a, cents_b, "centroids a->b")
    check_cloud(cents_b, cents_a, "centroids b->a")


def check_linked_duplicates_import():
    """Verifies ``linked_duplicates=True`` (#35):
    - Single parse (0 ``SkpFile.parse()`` calls, 1 ``_core.full_parse()`` call),
    - No ``instance_type == 'COLLECTION'`` Empties and no ``(source geometry)``
      collection, while world-space triangles match ``linked_duplicates=False``,
    - Leaf components have no Empty and share one ``bpy.types.Mesh`` datablock
      (editing a vertex updates every placement),
    - Assembly components get an Empty named after the component and
      ``Select Grouped > Children`` picks all parts,
    - Groups keep today's ``keep_hierarchy`` tree with unshared ``Mesh`` data,
    - Mirrored & non-uniformly scaled placements, hidden instances & hidden tags,
      loose-edge curves, annotations, and instance-inherited paint (same paint
      shares one mesh; different inherited paint gets distinct meshes)."""
    import tempfile
    import mathutils
    import openskp
    from openskp import _core

    # 1. Single-parse verification + nested_components_2021.skp structure
    nested_path = os.path.join(FIXTURES_DIR, "nested_components_2021.skp")
    _reset_empty_scene()
    import_skp.import_skp(nested_path, keep_hierarchy=True, linked_duplicates=False)
    ref_nested_tris = _evaluated_world_triangles()
    assert len(ref_nested_tris) > 0

    _reset_empty_scene()
    orig_parse = openskp.SkpFile.parse
    orig_full_parse = _core.full_parse
    parse_calls = 0
    full_parse_calls = 0

    def count_parse(self, *a, **kw):
        nonlocal parse_calls
        parse_calls += 1
        return orig_parse(self, *a, **kw)

    def count_full_parse(*a, **kw):
        nonlocal full_parse_calls
        full_parse_calls += 1
        return orig_full_parse(*a, **kw)

    openskp.SkpFile.parse = count_parse
    _core.full_parse = count_full_parse
    try:
        st_ld = import_skp.import_skp(nested_path, keep_hierarchy=True, linked_duplicates=True)
    finally:
        openskp.SkpFile.parse = orig_parse
        _core.full_parse = orig_full_parse

    assert parse_calls == 0, f"expected 0 SkpFile.parse() calls, got {parse_calls}"
    assert full_parse_calls == 1, f"expected 1 _core.full_parse() call, got {full_parse_calls}"
    assert not [o for o in bpy.data.objects if o.instance_type == "COLLECTION"]
    assert not [c for c in bpy.data.collections if c.name.endswith("(source geometry)")]
    got_nested_tris = _evaluated_world_triangles()
    _assert_same_world_triangles(got_nested_tris, ref_nested_tris, label="nested_components_2021.skp")

    # Leaf component Bolt: zero Empties, all Bolt MESH objects share ONE Mesh datablock
    bolt_empties = [o for o in bpy.data.objects if o.type == "EMPTY" and _base_name(o.name) == "Bolt"]
    bolt_meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.get("sketchup_definition") == "Bolt"]
    assert not bolt_empties, bolt_empties
    assert len(bolt_meshes) == 8, len(bolt_meshes)
    assert len({o.data for o in bolt_meshes}) == 1, {o.data.name for o in bolt_meshes}

    # Assembly components Frame and Bracket: each gets a PLAIN_AXES Empty, and
    # Select Grouped > Children on Frame selects its own mesh + Brackets + Bolts
    frame_empties = [o for o in bpy.data.objects if o.type == "EMPTY" and o.get("sketchup_definition") == "Frame"]
    bracket_empties = [o for o in bpy.data.objects if o.type == "EMPTY" and o.get("sketchup_definition") == "Bracket"]
    assert len(frame_empties) == 2 and len(bracket_empties) == 4, (len(frame_empties), len(bracket_empties))

    for o in bpy.data.objects:
        o.select_set(False)
    f0 = frame_empties[0]
    f0.select_set(True)
    bpy.context.view_layer.objects.active = f0
    bpy.ops.object.select_grouped(extend=True, type="CHILDREN_RECURSIVE")
    selected = set(bpy.context.selected_objects)
    # 1 Frame Empty + 1 Frame own mesh + 2 Bracket Empties + 2 Bracket own meshes + 4 Bolt meshes = 10
    assert len(selected) == 10, sorted((o.name, o.type) for o in selected)
    assert sum(1 for o in selected if o.get("sketchup_definition") == "Bolt") == 4

    # Editing 1 vertex on Bolt's shared Mesh moves that vertex on all 8 Bolt placements
    shared_bolt_data = bolt_meshes[0].data
    before_world_v0 = [o.matrix_world @ o.data.vertices[0].co for o in bolt_meshes]
    shared_bolt_data.vertices[0].co = shared_bolt_data.vertices[0].co + mathutils.Vector((0.25, 0.0, 0.0))
    bpy.context.view_layer.update()
    after_world_v0 = [o.matrix_world @ o.data.vertices[0].co for o in bolt_meshes]
    assert all((a - b).length > 0.2 for a, b in zip(after_world_v0, before_world_v0))

    # 2. World-triangle parity on capilla_quiroz_v17.skp (textures + curves) & gondola_v20.skp (81 placements + curves)
    for fx in ("capilla_quiroz_v17.skp", "gondola_v20.skp"):
        fx_path = os.path.join(FIXTURES_DIR, fx)
        _reset_empty_scene()
        import_skp.import_skp(fx_path, keep_hierarchy=True, linked_duplicates=False)
        tris_ref = _evaluated_world_triangles()

        _reset_empty_scene()
        import_skp.import_skp(fx_path, keep_hierarchy=True, linked_duplicates=True)
        assert not [o for o in bpy.data.objects if o.instance_type == "COLLECTION"], fx
        assert not [c for c in bpy.data.collections if c.name.endswith("(source geometry)")], fx
        tris_ld = _evaluated_world_triangles()
        _assert_same_world_triangles(tris_ld, tris_ref, label=fx)

    # 3. Groups keep today's hierarchy & get independent unshared Mesh datablocks
    # In nesting_2021.skp, Outer (is_group=True) > Shelf (component, is_group=False) > Inner (is_group=True):
    nest_path = os.path.join(FIXTURES_DIR, "nesting_2021.skp")
    _reset_empty_scene()
    import_skp.import_skp(nest_path, keep_hierarchy=False, linked_duplicates=False)
    nest_ref_tris = _evaluated_world_triangles()

    _reset_empty_scene()
    st_nest_kh = import_skp.import_skp(nest_path, keep_hierarchy=True, linked_duplicates=True)
    assert st_nest_kh["groups"] == 2, st_nest_kh
    _assert_same_world_triangles(_evaluated_world_triangles(), nest_ref_tris, label="nesting_2021 (kh=True)")
    grp_empties = [o for o in bpy.data.objects if o.type == "EMPTY"]
    grp_meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    assert len(grp_empties) == 2 and len(grp_meshes) == 3, (grp_empties, grp_meshes)
    assert len({o.data for o in grp_meshes}) == 3
    outer_e = next(o for o in grp_empties if o.parent is None)
    shelf_e = next(o for o in grp_empties if o.parent == outer_e)
    inner_m = next(o for o in grp_meshes if _base_name(o.name) == "Inner")
    assert _base_name(outer_e.name) == "Outer" and _base_name(shelf_e.name) == "Shelf"
    assert inner_m.parent == shelf_e  # leaf group has no Empty of its own

    # With keep_hierarchy=False on nesting_2021.skp: Outer (group) has NO Empty,
    # while Shelf (component containing group Inner) DOES get an Empty with Shelf+Inner parented under it!
    _reset_empty_scene()
    st_nest_flat = import_skp.import_skp(nest_path, keep_hierarchy=False, linked_duplicates=True)
    assert st_nest_flat["groups"] == 1, st_nest_flat
    (shelf_only_e,) = [o for o in bpy.data.objects if o.type == "EMPTY"]
    assert _base_name(shelf_only_e.name) == "Shelf" and shelf_only_e.parent is None
    flat_meshes_by_name = {_base_name(o.name): o for o in bpy.data.objects if o.type == "MESH"}
    assert flat_meshes_by_name["Outer"].parent is None
    assert flat_meshes_by_name["Shelf"].parent == shelf_only_e
    assert flat_meshes_by_name["Inner"].parent == shelf_only_e
    _assert_same_world_triangles(_evaluated_world_triangles(), nest_ref_tris, label="nesting_2021 (kh=False)")

    # Pure nested groups (OuterGrp > InnerGrp, plus two placements of one group def -> unshared meshes):
    with tempfile.TemporaryDirectory() as tmp_grp:
        bg = openskp.create()
        with bg.add_component_definition("InnerGrpDef") as ig_def:
            ig_def.add_face([(0, 0, 0), (6, 0, 0), (6, 6, 0), (0, 6, 0)])
        with bg.add_group("OuterGrp") as og:
            og.add_face([(0, 0, 0), (20, 0, 0), (20, 20, 0), (0, 20, 0)])
            og.add_group_instance(ig_def, name="InnerGrpA", translation=(2.0, 2.0, 4.0))
            og.add_group_instance(ig_def, name="InnerGrpB", translation=(10.0, 2.0, 4.0))
        pure_grp_skp = os.path.join(tmp_grp, "pure_groups.skp")
        with open(pure_grp_skp, "wb") as f:
            f.write(bg.to_bytes())

        _reset_empty_scene()
        st_pg_kh = import_skp.import_skp(pure_grp_skp, keep_hierarchy=True, linked_duplicates=True)
        assert st_pg_kh["groups"] == 1  # OuterGrp Empty only; leaf groups InnerGrpA/B have no Empty
        pg_meshes = [o for o in bpy.data.objects if o.type == "MESH"]
        assert len(pg_meshes) == 3 and len({o.data for o in pg_meshes}) == 3  # unshared group meshes!

        _reset_empty_scene()
        st_pg_flat = import_skp.import_skp(pure_grp_skp, keep_hierarchy=False, linked_duplicates=True)
        assert st_pg_flat["groups"] == 0
        assert not [o for o in bpy.data.objects if o.type == "EMPTY"]
        pg_flat_meshes = [o for o in bpy.data.objects if o.type == "MESH"]
        assert len(pg_flat_meshes) == 3 and len({o.data for o in pg_flat_meshes}) == 3

    # 4. Mirrored, non-uniformly scaled, hidden instances/tags, and instance-inherited paint
    # Uses SDK-saved 2021 VFF fixture (so D107 instance paint, D207 tags, and 8E3C hidden tags
    # are exercised with openskp 1.6.0 as well as newer releases).
    synth_skp = os.path.join(FIXTURES_DIR, "linked_duplicates", "inherited_paint_2021.skp")
    _scan_fixture_no_private_paths(synth_skp)

    # Reference world geometry from linked_duplicates=False (HiddenInst stays visible, matching main)
    _reset_empty_scene()
    import_skp.import_skp(synth_skp, keep_hierarchy=True, linked_duplicates=False)
    synth_ref_tris = _evaluated_world_triangles()
    assert len(synth_ref_tris) == 20, len(synth_ref_tris)  # 5 visible placements x 4 triangles

    _reset_empty_scene()
    st_inh = import_skp.import_skp(synth_skp, keep_hierarchy=True, linked_duplicates=True)
    bpy.context.view_layer.update()
    assert st_inh["unique_meshes"] == 3 and st_inh["placements"] == 6, st_inh
    assert st_inh["unique_layers"] == 3 and st_inh["hidden_layers"] == 1, st_inh

    by_name = {_base_name(o.name): o for o in bpy.data.objects if o.type == "MESH"}
    red_a = by_name["RedA"]
    red_mir = by_name["RedMirroredX"]
    blue_sc = by_name["BlueScaled"]
    unp_mir = by_name["UnpaintedMirroredZ"]
    hid_inst = by_name["HiddenInst"]
    hid_tag_obj = by_name["OnHiddenTag"]

    # Same inherited paint -> shared Mesh datablock; different paint -> distinct Mesh datablocks
    assert red_a.data is red_mir.data and red_a.data is hid_inst.data
    assert blue_sc.data is hid_tag_obj.data
    assert red_a.data is not blue_sc.data
    assert red_a.data is not unp_mir.data
    assert blue_sc.data is not unp_mir.data

    # Mirrored determinants (< 0), non-uniform scale, visible HiddenInst, and hidden tag preserved
    assert red_mir.matrix_world.determinant() < 0.0
    assert unp_mir.matrix_world.determinant() < 0.0
    assert abs(blue_sc.matrix_world.determinant() - 1.5) < 1e-5
    assert not hid_inst.hide_viewport and not hid_inst.hide_render and hid_inst.visible_get()
    hid_coll = bpy.data.collections["HiddenTag"]
    assert hid_coll.hide_viewport and hid_coll.hide_render
    assert not hid_tag_obj.visible_get()
    _assert_same_world_triangles(_evaluated_world_triangles(), synth_ref_tris, label="inherited_paint_2021")

    # 5. Annotations with linked_duplicates=True:
    # (a) component dimension parented to leaf MESH "Beam 1"
    _reset_empty_scene()
    annot_path = os.path.join(FIXTURES_DIR, "annotations_2017.skp")
    st_ann = import_skp.import_skp(annot_path, keep_hierarchy=True, import_annotations=True, linked_duplicates=True)
    bpy.context.view_layer.update()
    assert st_ann["texts_imported"] == 1 and st_ann["dimensions_imported"] == 2
    beam_mesh = next(o for o in bpy.data.objects if o.type == "MESH" and _base_name(o.name) == "Beam 1")
    beam_dim = next(
        o for o in bpy.data.objects
        if o.type == "CURVE" and o.get("sketchup_annotation") == "dimension_line"
        and abs(o["sketchup_distance_inches"] - 24.0) < 1e-4
    )
    assert beam_dim.parent == beam_mesh
    orig_mid = beam_dim.matrix_world.translation.copy()
    beam_mesh.location = beam_mesh.location + mathutils.Vector((0.5, -0.25, 0.1))
    bpy.context.view_layer.update()
    assert (beam_dim.matrix_world.translation - (orig_mid + mathutils.Vector((0.5, -0.25, 0.1)))).length < 1e-5

    # (b) With Keep Hierarchy off and Import Annotations on, a root group containing a text
    # gets NO extra Empty, and moving the root group moves its text with it.
    with tempfile.TemporaryDirectory() as tmp_rg:
        brg = openskp.create()
        with brg.add_group("RootAnnotatedGrp", translation=(10.0, 20.0, 0.0)) as rg:
            rg.add_face([(0, 0, 0), (12, 0, 0), (12, 12, 0), (0, 12, 0)])
            rg.add_text("Root group note", point=(2.0, 3.0, 4.0), leader=(3.0, 3.0, 4.0))
        rg_skp = os.path.join(tmp_rg, "root_grp_text.skp")
        with open(rg_skp, "wb") as f:
            f.write(brg.to_bytes())

        _reset_empty_scene()
        st_rg = import_skp.import_skp(
            rg_skp,
            keep_hierarchy=False,
            import_annotations=True,
            linked_duplicates=True,
        )
        bpy.context.view_layer.update()
        assert st_rg["groups"] == 0, st_rg
        assert st_rg["texts_imported"] == 1, st_rg
        assert not [o for o in bpy.data.objects if o.type == "EMPTY"], [o.name for o in bpy.data.objects]
        (rg_mesh,) = [o for o in bpy.data.objects if o.type == "MESH"]
        (rg_txt,) = [o for o in bpy.data.objects if o.type == "FONT"]
        assert rg_txt.parent == rg_mesh
        want_txt_pos = mathutils.Vector(((10.0 + 5.0) * 0.0254, (20.0 + 6.0) * 0.0254, (4.0 + 4.0) * 0.0254))
        assert (rg_txt.matrix_world.translation - want_txt_pos).length < 1e-5, (
            tuple(rg_txt.matrix_world.translation), tuple(want_txt_pos)
        )
        move_delta = mathutils.Vector((1.2, -0.8, 0.4))
        rg_mesh.location = rg_mesh.location + move_delta
        bpy.context.view_layer.update()
        assert (rg_txt.matrix_world.translation - (want_txt_pos + move_delta)).length < 1e-5

    # Also test operator property wiring directly
    already_registered = hasattr(bpy.types, "IMPORT_SCENE_OT_openskp")
    addon_pkg = _load_module("__init__")
    if not already_registered:
        addon_pkg.register()
    try:
        _reset_empty_scene()
        res = bpy.ops.import_scene.openskp(
            filepath=nested_path,
            keep_hierarchy=True,
            linked_duplicates=True,
            import_annotations=False,
        )
        assert res == {"FINISHED"}, res
        assert not [o for o in bpy.data.objects if o.instance_type == "COLLECTION"]
    finally:
        if not already_registered:
            addon_pkg.unregister()

    print("linked_duplicates=True import: sharing, Empties, Select Grouped > Children, groups, mirror/scale, hidden & paint - OK")


def main():
    check_missing_bundled_library_is_a_clear_error()
    checked = 0
    for name in os.listdir(FIXTURES_DIR):
        if name.endswith(".skp"):
            check(name)
            checked += 1
    assert checked == len(EXPECTED), (
        f"expected to check {len(EXPECTED)} fixtures, found {checked} .skp files in {FIXTURES_DIR}"
    )
    check_colorized_texture_is_tinted()
    check_alpha_materials_use_the_right_render_method()
    check_vertices_are_welded()
    check_solid_view_shows_painted_side()
    check_concave_surfaces_keep_one_orientation()
    check_cameraless_page_is_skipped()
    check_annotations_import()
    check_library_warnings_are_collected()
    check_import_reports_no_warnings_for_a_clean_file()
    check_partial_read_warning_is_reported()
    check_a_skipped_texture_is_reported_by_the_import()
    check_original_textures_are_loaded()
    check_hierarchy("capilla_quiroz_v17.skp", expected_groups=0)
    check_nesting_tree()
    check_linked_duplicates_off_matches_main()
    check_linked_duplicates_import()
    print("test_import: all fixtures OK (including source-geometry discoverability and linked duplicates)")


if __name__ == "__main__":
    main()

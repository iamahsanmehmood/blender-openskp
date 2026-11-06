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
        check_material_names_and_sides(fixture_name, new_objects)
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
        r, g, b, a = front_gltf.get("pbrMetallicRoughness", {}).get("baseColorFactor", [0.8, 0.8, 0.8, 1.0])
        expected = tuple(round(c, 3) for c in (*import_skp.srgb_to_linear((r, g, b)), a))
        assert tuple(round(c, 3) for c in mat.diffuse_color) == expected, (
            f"{label}/{mat.name}: Solid shading should show the front paint"
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
    print("test_import: all fixtures OK (including source-geometry discoverability)")


if __name__ == "__main__":
    main()

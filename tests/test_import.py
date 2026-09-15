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
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _load_addon import _load_module  # noqa: E402

import_skp = _load_module("import_skp")

import bpy  # noqa: E402

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

# (unique_meshes, placements, triangles, unique_curve_meshes, curve_placements, curve_runs) per fixture.
EXPECTED = {
    "SU_File.skp": (1, 1, 104, 0, 0, 0),
    "capilla_quiroz_v17.skp": (3, 4, 871, 2, 2, 20),
}


def check(fixture_name):
    path = os.path.join(FIXTURES_DIR, fixture_name)
    stats = import_skp.import_skp(path)
    got = (
        stats["unique_meshes"], stats["placements"], stats["triangles"],
        stats["unique_curve_meshes"], stats["curve_placements"], stats["curve_runs"],
    )
    print(f"{fixture_name}: {got}")
    expected = EXPECTED.get(fixture_name)
    if expected is not None:
        assert got == expected, f"{fixture_name}: expected {expected}, got {got}"
    if stats["unique_meshes"]:
        check_source_geometry_is_discoverable_but_excluded(fixture_name)
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

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import import_skp  # noqa: E402

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
    return got


def main():
    checked = 0
    for name in os.listdir(FIXTURES_DIR):
        if name.endswith(".skp"):
            check(name)
            checked += 1
    assert checked == len(EXPECTED), (
        f"expected to check {len(EXPECTED)} fixtures, found {checked} .skp files in {FIXTURES_DIR}"
    )
    print("test_import: all fixtures OK")


if __name__ == "__main__":
    main()

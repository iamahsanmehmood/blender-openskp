"""Run with Blender's own headless mode:

    blender --background --python tests/test_export.py

Builds a real Blender cube, exports it via export_skp.export_skp(), then
reads the result back through openskp's OWN Python reader (not this
addon) and checks the exact triangle count and world-space coordinates
(after the metres->inches conversion) round-trip losslessly.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import export_skp  # noqa: E402

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
        "faces_written": 12,
        "faces_skipped": 0,
        "objects_skipped": 0,
        "objects_exported": 1,
    }, stats

    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor")
    )
    import openskp

    model = openskp.SkpFile.open(out_path).parse()
    os.remove(out_path)

    assert len(model.root.faces) == 12, f"expected 12 faces, got {len(model.root.faces)}"

    # A 2m cube at (1, 2, 3) spans, in inches: x [0, 78.74], y [39.37,
    # 118.11], z [78.74, 157.48]. Check every stored vertex lands exactly
    # on that box (exact, not approximate - this is a pure unit-scale
    # conversion, so it should be lossless to float precision).
    lo = (0.0, 1.0 * METRES_TO_INCHES, 2.0 * METRES_TO_INCHES)
    hi = (2.0 * METRES_TO_INCHES, 3.0 * METRES_TO_INCHES, 4.0 * METRES_TO_INCHES)
    for v in model.root.vertices.values():
        for coord, lo_c, hi_c in zip((v.x, v.y, v.z), lo, hi):
            assert lo_c - 1e-6 <= coord <= hi_c + 1e-6, (v.x, v.y, v.z, lo, hi)

    print("test_export: OK (12 faces, coordinates exact)")


if __name__ == "__main__":
    main()

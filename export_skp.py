"""Blender exporter for native SketchUp (.skp) files, built on OpenSKP.

Exports every selected mesh object's geometry (or every visible mesh
object if nothing is selected) as a flat set of faces at a new .skp
file's root - global (world) transform resolved, so parented/linked
objects land in the right position. Blender's mesh polygons are not
guaranteed planar (nothing in Blender enforces that on an edited mesh),
so this exports Blender's own triangulated view (mesh.calc_loop_triangles,
which IS guaranteed planar per-triangle) rather than raw n-gons - simpler
and more defensive than passing arbitrary n-gons through and hoping.
No hole reconstruction (Blender's own mesh polygons have no native
"outer boundary + holes" concept to read one back from, unlike a real
B-rep), no material/layer carry-over - geometry only, matching the scope
of the FreeCAD addon's own exporter.
"""
from __future__ import annotations

import os
import sys

_VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

# 1 inch = 0.0254 m exactly - SketchUp's native unit is inches, Blender's is
# metres (both are Z-up, so no axis swap is needed here - only the import
# side's glTF-Y-up InstancedScene API needed one).
METRES_TO_INCHES = 1.0 / 0.0254


def _export_object(builder, obj, stats):
    mesh = obj.to_mesh()
    try:
        mesh.calc_loop_triangles()
        mw = obj.matrix_world
        for tri in mesh.loop_triangles:
            pts = []
            for vi in tri.vertices:
                world_co = mw @ mesh.vertices[vi].co
                pts.append(
                    (
                        world_co.x * METRES_TO_INCHES,
                        world_co.y * METRES_TO_INCHES,
                        world_co.z * METRES_TO_INCHES,
                    )
                )
            try:
                builder.add_face(pts)
                stats["faces_written"] += 1
            except Exception:
                # A degenerate triangle (zero area, from a pinched/duplicate
                # vertex) isn't something openskp's writer can represent -
                # skip it rather than aborting the whole export.
                stats["faces_skipped"] += 1
    finally:
        obj.to_mesh_clear()


def export_skp(filepath, context):
    """Exports selected (or all visible) mesh objects to a new .skp file
    at filepath, returning a stats dict."""
    from openskp import create

    objs = [o for o in context.selected_objects if o.type == "MESH"]
    if not objs:
        objs = [o for o in context.scene.objects if o.type == "MESH" and o.visible_get()]

    builder = create()
    stats = {"faces_written": 0, "faces_skipped": 0, "objects_skipped": 0, "objects_exported": 0}

    for obj in objs:
        if obj.data is None or len(obj.data.polygons) == 0:
            stats["objects_skipped"] += 1
            continue
        _export_object(builder, obj, stats)
        stats["objects_exported"] += 1

    with open(filepath, "wb") as f:
        f.write(builder.to_bytes())

    print(
        f"openskp export: {stats['objects_exported']} objects, "
        f"{stats['faces_written']} faces written, {stats['faces_skipped']} skipped"
    )
    return stats

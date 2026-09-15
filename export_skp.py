"""Blender exporter for native SketchUp (.skp) files, built on OpenSKP.

Exports every selected mesh object's geometry (or every visible mesh
object if nothing is selected) as a flat set of faces at a new .skp
file's root - global (world) transform resolved, so parented/linked
objects land in the right position.

Each Blender polygon (n-gon) is exported as ONE SketchUp face when it's
planar - which a quad/n-gon from clean, CAD-style modelling almost always
is - rather than always triangulating (mesh.calc_loop_triangles()), which
was the original approach here. That first version exported every quad
face (all six sides of a plain cube, for instance) as two triangles
sharing a diagonal, so the exported model had a spurious "middle line"
down every face that SketchUp itself has no reason to draw - caught by
testing a real export, not a theoretical concern. Triangulation is now
the FALLBACK, used only for a face that genuinely isn't planar (nothing
in Blender enforces that on an edited mesh, so it can happen), and even
then only for that one face, not the whole mesh.

Planarity is checked in the mesh's own LOCAL space, before the object's
world matrix is applied - correct because any affine transform (rotation/
uniform or non-uniform scale/shear, which is everything a Blender object
matrix can express) maps a plane to a plane, so a check done once locally
holds regardless of how the object is placed in the scene.

No hole reconstruction (Blender's own mesh polygons have no native
"outer boundary + holes" concept to read one back from, unlike a real
B-rep), no material carry-over - geometry plus layers, for now.

Layers (SketchUp "tags"): each exported object's layer is its own
Collection membership - the first Collection it's linked into other
than the scene's master Collection or a "(source geometry)" one (the
same convention this addon's own import side already establishes:
Blender's closest equivalent to a SketchUp tag is a named Collection,
and organizing objects into named Collections to control the exported
tag is the same pattern glTF/FBX/USD exporters already use for
layers/collections). An object in no meaningful Collection of its own
exports to SketchUp's default layer, same as before this existed.

Deliberately NOT wired through the master/source objects an import
leaves behind (`<file> (source geometry)`): those sit in one Collection
per unique DEFINITION, not per layer - the same definition can appear
on several different tags across different placements (see
import_skp.py's own module docstring on this), so there is no single
correct layer to read off a shared master object. Exporting placement
Empties (each with its own well-defined, resolved layer) instead of
flattening back to their master mesh is future scope, not this pass.
"""
from __future__ import annotations

# 1 inch = 0.0254 m exactly - SketchUp's native unit is inches, Blender's is
# metres (both are Z-up, so no axis swap is needed here - only the import
# side's glTF-Y-up InstancedScene API needed one).
METRES_TO_INCHES = 1.0 / 0.0254

# Relative-to-polygon-size tolerance for the planarity check - a vertex's
# distance from the polygon's own best-fit plane, divided by the polygon's
# longest edge, has to clear this before the face is triangulated instead
# of exported whole. Loose enough to absorb ordinary float noise from
# modelling operations, tight enough to still catch a real bent quad.
_PLANARITY_TOLERANCE = 1e-4


def _world_point(mw, co):
    p = mw @ co
    return (p.x * METRES_TO_INCHES, p.y * METRES_TO_INCHES, p.z * METRES_TO_INCHES)


def _is_planar(mesh, polygon):
    verts = [mesh.vertices[vi].co for vi in polygon.vertices]
    if len(verts) <= 3:
        return True  # a triangle is always planar
    normal = polygon.normal
    center = polygon.center
    longest_edge = max(
        (verts[i] - verts[i - 1]).length for i in range(len(verts))
    )
    if longest_edge <= 0.0:
        return True
    for v in verts:
        dist = abs((v - center).dot(normal))
        if dist / longest_edge > _PLANARITY_TOLERANCE:
            return False
    return True


def _object_layer_name(obj, scene_collection):
    """The Collection this object's export should be tagged with in
    SketchUp - the first Collection it's linked into other than the
    scene's own implicit root ("Scene Collection", `scene.collection`
    itself - an object can be linked directly into it, bypassing every
    named collection) or "Collection" (every new .blend's own
    auto-created default collection, confirmed directly: it is a real,
    separately-named CHILD of `scene.collection`, not `scene.collection`
    itself, so an unmodified default scene's objects all resolve to it -
    excluded by name specifically so a from-scratch export without any
    deliberate organizing doesn't pick up a spurious "Collection" tag
    nothing chose), or an addon-generated "(source geometry)" one (see
    module docstring). None for SketchUp's own default layer when no
    other Collection applies.

    Compares by name, not `is` - Blender's Python API can hand back a
    freshly-wrapped RNA proxy for the same underlying Collection on each
    property access, so `coll is scene_collection` silently never matches
    even where the names are identical (confirmed directly)."""
    for coll in obj.users_collection:
        if coll.name in (scene_collection.name, "Collection"):
            continue
        if coll.name.endswith("(source geometry)"):
            continue
        return coll.name
    return None


def _export_object(builder, obj, stats, layer):
    mesh = obj.to_mesh()
    try:
        mw = obj.matrix_world
        needs_triangulation = [
            poly.index for poly in mesh.polygons if not _is_planar(mesh, poly)
        ]

        for poly in mesh.polygons:
            if poly.index in needs_triangulation:
                continue
            pts = [_world_point(mw, mesh.vertices[vi].co) for vi in poly.vertices]
            try:
                builder.add_face(pts, layer=layer)
                stats["faces_written"] += 1
            except Exception:
                # A degenerate polygon (zero area, from pinched/duplicate
                # vertices) isn't something openskp's writer can represent -
                # skip it rather than aborting the whole export.
                stats["faces_skipped"] += 1

        if needs_triangulation:
            mesh.calc_loop_triangles()
            wanted = set(needs_triangulation)
            for tri in mesh.loop_triangles:
                if tri.polygon_index not in wanted:
                    continue
                pts = [_world_point(mw, mesh.vertices[vi].co) for vi in tri.vertices]
                try:
                    builder.add_face(pts, layer=layer)
                    stats["faces_written"] += 1
                except Exception:
                    stats["faces_skipped"] += 1
    finally:
        obj.to_mesh_clear()


def export_skp(filepath, context):
    """Exports selected (or all visible) mesh objects to a new .skp file
    at filepath, returning a stats dict."""
    from .vendor.openskp import create

    objs = [o for o in context.selected_objects if o.type == "MESH"]
    if not objs:
        objs = [o for o in context.scene.objects if o.type == "MESH" and o.visible_get()]

    builder = create()

    # openskp's writer requires every layer to be registered before the
    # first add_face call, so this resolves each object's layer name (see
    # _object_layer_name) up front and registers each distinct one exactly
    # once - a plain dedup, not a second export pass.
    scene_collection = context.scene.collection
    object_layer_names = {obj.name: _object_layer_name(obj, scene_collection) for obj in objs}
    layer_handle_by_name = {}
    for name in object_layer_names.values():
        if name is not None and name not in layer_handle_by_name:
            layer_handle_by_name[name] = builder.add_layer(name)

    stats = {
        "faces_written": 0, "faces_skipped": 0, "objects_skipped": 0, "objects_exported": 0,
        "layers_written": len(layer_handle_by_name),
    }

    for obj in objs:
        if obj.data is None or len(obj.data.polygons) == 0:
            stats["objects_skipped"] += 1
            continue
        layer_name = object_layer_names[obj.name]
        layer = layer_handle_by_name.get(layer_name) if layer_name is not None else None
        _export_object(builder, obj, stats, layer)
        stats["objects_exported"] += 1

    with open(filepath, "wb") as f:
        f.write(builder.to_bytes())

    print(
        f"openskp export: {stats['objects_exported']} objects, "
        f"{stats['faces_written']} faces written, {stats['faces_skipped']} skipped, "
        f"{stats['layers_written']} layers"
    )
    return stats

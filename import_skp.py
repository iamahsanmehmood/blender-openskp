"""Blender importer for native SketchUp (.skp) files, built on OpenSKP.

Uses openskp's ``build_instanced_scene()`` rather than its flattened
``build_scene()`` - the former keeps each unique component/group
DEFINITION's geometry as one local-space mesh plus a separate list of
placements, instead of baking every placement into its own copy of the
vertex data. That maps directly onto Blender's own collection-instancing
model: one Collection per unique definition, referenced by an Empty
(``instance_type='COLLECTION'``) per placement. A component placed 1,000
times costs one mesh, not 1,000 - unlike a flat triangle-soup import,
editing the source mesh updates every placement at once, matching how
SketchUp's own components behave.

Coordinates: openskp's InstancedScene is already metres, glTF Y-up
(positions/normals) with a 16-element column-major matrix per node -
Blender is also metres but Z-up, so every matrix gets one fixed
axis-conversion matrix multiplied in (see _YUP_TO_ZUP below), applied once
per node rather than per vertex.

Loose edges (construction lines/structural framing - a light-gauge-steel
member is routinely drawn this way, not as a solid) come from the SAME
instanced scene via ``InstancedScene.curve_resources``, imported as
separate edges-only Mesh Objects (no faces, so Blender renders them as
plain lines) placed the same way and cached the same way as the mesh
resources above - a definition can have both a mesh_resource_id and a
curve_resource_id at once. Found missing by testing this importer against
a real structural-framing file (93 of 145 definitions were entirely or
partly loose-edge, silently invisible before openskp gained
InstancedCurveResource support).
"""
from __future__ import annotations

import os
import sys
import time

import bpy
import mathutils

_VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

# glTF Y-up -> Blender Z-up: (x, y, z) -> (x, -z, y). Same convention
# openskp's own scene.py/instanced_scene.py already document for their
# own Y-up output, just expressed as a Blender Matrix instead of a
# per-point axis swap.
_YUP_TO_ZUP = mathutils.Matrix(
    (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
)


def _gltf_matrix_to_blender(m):
    """16-element column-major glTF matrix -> mathutils.Matrix (row-major
    constructor, so this transposes while unpacking)."""
    return mathutils.Matrix(
        (
            (m[0], m[4], m[8], m[12]),
            (m[1], m[5], m[9], m[13]),
            (m[2], m[6], m[10], m[14]),
            (m[3], m[7], m[11], m[15]),
        )
    )


def _build_mesh_object(name, resource):
    """Builds one Mesh Object (local space, at the origin) from an
    InstancedMeshResource's primitives - already-triangulated, so this is
    a direct from_pydata() with no triangulation of its own to do."""
    verts = []
    tris = []
    for prim in resource.primitives:
        base = len(verts)
        pos = prim.positions
        for i in range(0, len(pos), 3):
            verts.append((pos[i], pos[i + 1], pos[i + 2]))
        idx = prim.indices
        for i in range(0, len(idx), 3):
            tris.append((base + idx[i], base + idx[i + 1], base + idx[i + 2]))

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], tris)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    return obj


def _get_or_build_collection(resource_id, resources_by_id, collection_cache, stats):
    """Returns the Collection holding resource_id's mesh object, building
    it exactly once per unique id and caching it for every later
    placement - the whole point of using build_instanced_scene() over
    build_scene()."""
    if resource_id in collection_cache:
        return collection_cache[resource_id]

    resource = resources_by_id[resource_id]
    coll = bpy.data.collections.new(resource.definition_name or resource_id)
    obj = _build_mesh_object(resource.definition_name or resource_id, resource)
    coll.objects.link(obj)
    collection_cache[resource_id] = coll
    stats["unique_meshes"] += 1
    stats["triangles"] += sum(len(p.indices) // 3 for p in resource.primitives)
    return coll


def _build_curve_object(name, resource):
    """Builds one edges-only Mesh Object (local space, at the origin) from
    an InstancedCurveResource's runs - no faces, so Blender renders these
    as plain construction lines, matching what the source file actually
    stores for loose edges (a light-gauge-steel/structural-framing member
    drawn as a line, not a solid).

    Renders each run's own chords (LocalCurve.points_m) rather than
    reconstructing a true circular arc from LocalCurve.arc when present -
    a well-tessellated arc's chords are visually indistinguishable at any
    reasonable zoom, and a genuine smooth-curve (bpy.data.curves) import
    for the analytic-arc case is future scope, not required for loose
    edges to stop being silently invisible.
    """
    verts = []
    edges = []
    for curve in resource.curves:
        base = len(verts)
        pts = curve.points_m
        verts.extend(pts)
        for i in range(len(pts) - 1):
            edges.append((base + i, base + i + 1))
        if curve.closed and len(pts) > 2:
            edges.append((base + len(pts) - 1, base))

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, edges, [])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    return obj


def _get_or_build_curve_collection(resource_id, curve_resources_by_id, curve_collection_cache, stats):
    """Returns the Collection holding resource_id's loose-edge object,
    building it exactly once per unique id - same caching strategy as
    _get_or_build_collection, independent cache/id space since a node can
    carry a mesh_resource_id, a curve_resource_id, or both (a definition
    can have faces AND loose edges at once)."""
    if resource_id in curve_collection_cache:
        return curve_collection_cache[resource_id]

    resource = curve_resources_by_id[resource_id]
    name = (resource.definition_name or resource_id) + " (lines)"
    coll = bpy.data.collections.new(name)
    obj = _build_curve_object(name, resource)
    coll.objects.link(obj)
    curve_collection_cache[resource_id] = coll
    stats["unique_curve_meshes"] += 1
    stats["curve_runs"] += len(resource.curves)
    return coll


def _place_node(
    node,
    parent_matrix,
    resources_by_id,
    collection_cache,
    curve_resources_by_id,
    curve_collection_cache,
    target_collection,
    stats,
):
    """Walks one InstancedNode, placing an Empty (collection-instance) for
    its own mesh_resource_id and/or curve_resource_id (a definition can
    have both - faces AND loose edges at once) at the composed world
    matrix, then recurses into children with that same world matrix as
    their new parent matrix - matrices are relative-to-parent in openskp's
    own model, composed here into world space for a flat, editable result
    rather than mirroring the source's exact nesting as Blender parent/
    child objects."""
    local_matrix = _gltf_matrix_to_blender(node.matrix)
    world_matrix = parent_matrix @ local_matrix

    if node.mesh_resource_id is not None:
        coll = _get_or_build_collection(node.mesh_resource_id, resources_by_id, collection_cache, stats)
        empty = bpy.data.objects.new(node.name or node.mesh_resource_id, None)
        empty.instance_type = "COLLECTION"
        empty.instance_collection = coll
        empty.matrix_world = _YUP_TO_ZUP @ world_matrix
        target_collection.objects.link(empty)
        stats["placements"] += 1

    if node.curve_resource_id is not None:
        curve_coll = _get_or_build_curve_collection(
            node.curve_resource_id, curve_resources_by_id, curve_collection_cache, stats
        )
        curve_empty = bpy.data.objects.new(
            (node.name or node.curve_resource_id) + " (lines)", None
        )
        curve_empty.instance_type = "COLLECTION"
        curve_empty.instance_collection = curve_coll
        curve_empty.matrix_world = _YUP_TO_ZUP @ world_matrix
        target_collection.objects.link(curve_empty)
        stats["curve_placements"] += 1

    for child in node.children:
        _place_node(
            child,
            world_matrix,
            resources_by_id,
            collection_cache,
            curve_resources_by_id,
            curve_collection_cache,
            target_collection,
            stats,
        )


def import_skp(filepath, context=None):
    """Imports a .skp file into the current (or given) Blender context's
    scene, returning a stats dict."""
    import openskp

    context = context or bpy.context

    print(f"openskp: parsing {os.path.basename(filepath)}...")
    t0 = time.time()
    skp = openskp.SkpFile.open(filepath)
    scene = skp.build_instanced_scene()
    print(f"openskp: parsed in {time.time() - t0:.1f}s, building objects...")

    resources_by_id = {r.id: r for r in scene.mesh_resources}
    collection_cache = {}
    curve_resources_by_id = {r.id: r for r in getattr(scene, "curve_resources", None) or []}
    curve_collection_cache = {}
    stats = {
        "unique_meshes": 0,
        "placements": 0,
        "triangles": 0,
        "unique_curve_meshes": 0,
        "curve_placements": 0,
        "curve_runs": 0,
    }

    root_name = os.path.splitext(os.path.basename(filepath))[0]
    root_collection = bpy.data.collections.new(root_name)
    context.scene.collection.children.link(root_collection)

    t0 = time.time()
    _place_node(
        scene.scene_hierarchy,
        mathutils.Matrix.Identity(4),
        resources_by_id,
        collection_cache,
        curve_resources_by_id,
        curve_collection_cache,
        root_collection,
        stats,
    )
    print(
        f"openskp: {stats['unique_meshes']} unique meshes "
        f"({stats['triangles']} triangles), {stats['placements']} instances placed; "
        f"{stats['unique_curve_meshes']} unique loose-edge groups "
        f"({stats['curve_runs']} runs), {stats['curve_placements']} placed "
        f"in {time.time() - t0:.1f}s"
    )
    return stats

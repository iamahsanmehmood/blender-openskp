"""Blender exporter for native SketchUp (.skp) files, built on OpenSKP.

Exports selected objects (or every visible object when nothing exportable is
selected) to a new ``.skp`` file:

- **Collection instances as SketchUp components (#28)**: each instanced
  ``Collection`` (including child collections, ``instance_offset``, and parented
  objects inside the collection) and each ``keep_hierarchy`` imported
  group/component ``Empty`` tree is written once in bottom-up order via
  ``add_component_definition``, and each placement ``Empty`` is written via
  ``add_instance`` carrying its 4x4 transform (position, rotation, non-uniform
  scale and mirroring) and SketchUp tag (Collection membership).
- **Plain mesh objects as loose root geometry**: standalone ``MESH`` objects
  (including meshes parented under ordinary Blender ``Empty``s, armatures, or
  other meshes, and visible collections that are also instanced) keep the flat
  root-face export, preserving planar n-gons (triangulating only genuinely
  non-planar polygons), per-face solid material colors/opacity, and per-object
  layer/tag assignment.
"""
from __future__ import annotations

import re

import bpy
import mathutils

# 1 inch = 0.0254 m exactly - SketchUp's native unit is inches, Blender's is metres.
METRES_TO_INCHES = 1.0 / 0.0254

# Relative-to-polygon-size tolerance for the planarity check - a vertex's
# distance from the polygon's own best-fit plane, divided by the polygon's
# longest edge, has to clear this before the face is triangulated instead
# of exported whole.
_PLANARITY_TOLERANCE = 1e-4

# Matches import_skp.py's glTF Y-up -> Blender Z-up conversion. Source meshes
# inside "<model> (source geometry)" have vertices in glTF Y-up metres while
# their placement Empties post-multiply _YUP_TO_ZUP into matrix_world; applying
# _YUP_TO_ZUP to source mesh vertices and _ZUP_TO_YUP on the right of placement
# matrices recovers SketchUp's local Z-up definition coordinates and transforms.
_YUP_TO_ZUP = mathutils.Matrix(
    (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
)
_ZUP_TO_YUP = _YUP_TO_ZUP.transposed()

_UNPAINTED_SLOT_RE = re.compile(r"^openskp_material_\d+(?:\.\d{3})?$")
_BLENDER_DUP_SUFFIX_RE = re.compile(r"^(.*)\.(\d{3})$")


def _is_source_geom_name(name: str) -> bool:
    return re.sub(r"\.\d{3}$", "", name).endswith("(source geometry)")


def _strip_blender_dup_suffix(name: str, existing_names) -> str:
    """Strips a trailing ``.001``..``.999`` suffix only when the base name
    already exists in ``existing_names`` (meaning Blender auto-numbered a
    duplicate datablock), keeping real user names such as ``Cabinet.600``."""
    m = _BLENDER_DUP_SUFFIX_RE.match(name)
    if m and m.group(1) and m.group(1) in existing_names:
        return m.group(1)
    return name


def _world_point(mw, co):
    p = mw @ co
    return (p.x * METRES_TO_INCHES, p.y * METRES_TO_INCHES, p.z * METRES_TO_INCHES)


def _polygon_area_sq(pts) -> float:
    if len(pts) < 3:
        return 0.0
    x0, y0, z0 = pts[0]
    cx = cy = cz = 0.0
    for i in range(1, len(pts) - 1):
        ux, uy, uz = pts[i][0] - x0, pts[i][1] - y0, pts[i][2] - z0
        vx, vy, vz = pts[i + 1][0] - x0, pts[i + 1][1] - y0, pts[i + 1][2] - z0
        cx += uy * vz - uz * vy
        cy += uz * vx - ux * vz
        cz += ux * vy - uy * vx
    return 0.25 * (cx * cx + cy * cy + cz * cz)


def _is_planar(mesh, polygon):
    verts = [mesh.vertices[vi].co for vi in polygon.vertices]
    if len(verts) <= 3:
        return True
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


def _collect_import_collections():
    """Classifies collections created by ``import_skp`` so source-geometry
    collections convert back from glTF Y-up to SketchUp Z-up and imported tag
    collections strip any Blender ``.001`` duplicate suffix."""
    imported_roots = set()
    source_roots = []
    for coll in bpy.data.collections:
        for child in coll.children:
            if _is_source_geom_name(child.name):
                imported_roots.add(coll.name)
                source_roots.append(child)

    source_collections = set()

    def walk_source(c):
        if c.name in source_collections:
            return
        source_collections.add(c.name)
        for ch in c.children:
            walk_source(ch)

    for sr in source_roots:
        walk_source(sr)

    imported_tag_collections = set()
    for root_name in imported_roots:
        root_coll = bpy.data.collections.get(root_name)
        if root_coll is None:
            continue
        for child in root_coll.children:
            if _is_source_geom_name(child.name):
                continue
            base = _strip_blender_dup_suffix(child.name, bpy.data.collections)
            if base in ("Scenes", "Annotations"):
                continue
            imported_tag_collections.add(child.name)

    return imported_roots, source_collections, imported_tag_collections


def _object_layer_name(
    obj,
    scene_collection,
    imported_roots=(),
    source_collections=(),
    imported_tag_collections=(),
    instanced_colls=(),
    own_container_colls=(),
):
    """The Collection this object's export should be tagged with in SketchUp -
    the first Collection it is linked into other than the scene's own default
    ones ("Scene Collection", "Collection"), an imported model root, an
    instanced component definition collection, or a "(source geometry)"
    collection. Returns ``None`` for SketchUp's default untagged layer
    ("Layer0")."""
    for coll in obj.users_collection:
        cname = coll.name
        if cname in (scene_collection.name, "Collection"):
            continue
        if (
            cname in imported_roots
            or cname in source_collections
            or cname in instanced_colls
            or cname in own_container_colls
        ):
            continue
        if _is_source_geom_name(cname):
            continue
        if cname in imported_tag_collections:
            tag = _strip_blender_dup_suffix(cname, bpy.data.collections)
        else:
            tag = cname
        if tag in ("Layer0", "Annotations"):
            return None
        return tag
    return None


def _linear_to_srgb(c):
    return c * 12.92 if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def _material_rgba_opacity(mat):
    """A Material's solid display color as openskp's writer wants it:
    (r, g, b) ints 0-255, plus a separate 0.0-1.0 opacity (``None`` when
    fully opaque). ``diffuse_color`` is scene-linear and SketchUp colours are
    sRGB, so it is converted on the way out."""
    r, g, b, a = mat.diffuse_color
    rgba = tuple(round(_linear_to_srgb(min(max(c, 0.0), 1.0)) * 255) for c in (r, g, b))
    opacity = None if a >= 0.999 else float(a)
    return rgba, opacity


def _material_spec(mat):
    """Returns ``(dedup_key, skp_name, rgba, opacity)`` for a Blender Material,
    or ``None`` when unpainted (including ``import_skp``'s unnamed fallback
    slots ``openskp_material_N`` so unpainted SketchUp faces stay unpainted)."""
    if mat is None:
        return None
    if "openskp_material_index" in mat:
        base = _strip_blender_dup_suffix(mat.name, bpy.data.materials)
        if "openskp_back_material_index" in mat and " | " in base:
            front_name, back_name = base.split(" | ", 1)
            front_unpainted = bool(_UNPAINTED_SLOT_RE.match(front_name))
            back_unpainted = bool(_UNPAINTED_SLOT_RE.match(back_name))
            if front_unpainted and back_unpainted:
                return None
            skp_name = back_name if front_unpainted else front_name
        elif _UNPAINTED_SLOT_RE.match(base):
            return None
        else:
            skp_name = base
        rgba, opacity = _material_rgba_opacity(mat)
        return (f"skp:{skp_name}:{rgba}:{opacity}", skp_name, rgba, opacity)
    rgba, opacity = _material_rgba_opacity(mat)
    return (mat.name, mat.name, rgba, opacity)


def _build_local_mesh_polys(mesh):
    """Extracts local-space polygons ``([Vector, ...], material_index)`` once
    for a ``bpy.types.Mesh``, triangulating only genuinely non-planar faces."""
    polys = []
    verts = mesh.vertices
    needs_triangulation = {
        poly.index for poly in mesh.polygons if not _is_planar(mesh, poly)
    }

    for poly in mesh.polygons:
        if poly.index in needs_triangulation:
            continue
        cos = [verts[vi].co.copy() for vi in poly.vertices]
        polys.append((cos, poly.material_index))

    if needs_triangulation:
        mesh.calc_loop_triangles()
        for tri in mesh.loop_triangles:
            if tri.polygon_index not in needs_triangulation:
                continue
            cos = [verts[vi].co.copy() for vi in tri.vertices]
            polys.append((cos, tri.material_index))

    return polys, 0


def _get_local_mesh_polys(obj, depsgraph=None, local_mesh_cache=None):
    """Returns ``(local_polys, local_skipped)`` for ``obj``, caching per
    ``Mesh`` datablock when ``obj`` carries no modifiers (#28)."""
    if obj.data is None:
        return (), 0
    if not obj.modifiers:
        key = obj.data.as_pointer()
        if local_mesh_cache is not None and key in local_mesh_cache:
            return local_mesh_cache[key]
        mesh = obj.to_mesh()
        try:
            entry = _build_local_mesh_polys(mesh)
        finally:
            obj.to_mesh_clear()
        if local_mesh_cache is not None:
            local_mesh_cache[key] = entry
        return entry

    eval_obj = obj.evaluated_get(depsgraph) if depsgraph is not None else obj
    mesh = eval_obj.to_mesh()
    try:
        return _build_local_mesh_polys(mesh)
    finally:
        eval_obj.to_mesh_clear()


def _extract_mesh_faces(obj, transform, stats, depsgraph=None, local_mesh_cache=None):
    """Extracts ``(pts_inches, mat_spec)`` tuples from ``obj`` transformed by
    4x4 metre matrix ``transform``."""
    local_polys, local_skipped = _get_local_mesh_polys(obj, depsgraph, local_mesh_cache)
    stats["faces_skipped"] += local_skipped
    if not local_polys:
        return []

    specs = [_material_spec(slot.material) for slot in obj.material_slots]
    if not specs and obj.data and obj.data.materials:
        specs = [_material_spec(m) for m in obj.data.materials]
    num_specs = len(specs)

    faces = []
    for cos, mat_index in local_polys:
        pts = [_world_point(transform, co) for co in cos]
        if _polygon_area_sq(pts) == 0.0:
            stats["faces_skipped"] += 1
            continue
        spec = specs[mat_index] if 0 <= mat_index < num_specs else None
        faces.append((pts, spec))
    return faces


def _try_invert(m4):
    try:
        return m4.inverted()
    except ValueError:
        return None


def _is_identity_matrix(m4, rot_tol=1e-4, trans_tol_m=1e-4) -> bool:
    ident = mathutils.Matrix.Identity(4)
    for r in range(4):
        for c in range(4):
            limit = trans_tol_m if c == 3 and r < 3 else rot_tol
            if abs(m4[r][c] - ident[r][c]) > limit:
                return False
    return True


def _matrix_to_skp_args(m4):
    """Converts a 4x4 SketchUp Z-up metre matrix into ``(matrix3x3, translation_in)``
    as expected by ``SkpBuilder.add_instance`` (row-major 3x3 + inch translation)."""
    m3 = (
        float(m4[0][0]), float(m4[0][1]), float(m4[0][2]),
        float(m4[1][0]), float(m4[1][1]), float(m4[1][2]),
        float(m4[2][0]), float(m4[2][1]), float(m4[2][2]),
    )
    ident3 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    if all(abs(a - b) <= 1e-6 for a, b in zip(m3, ident3)):
        m3 = None
    tr = (
        float(m4[0][3] * METRES_TO_INCHES),
        float(m4[1][3] * METRES_TO_INCHES),
        float(m4[2][3] * METRES_TO_INCHES),
    )
    return m3, tr


def _instance_name_for(empty_obj, def_base_name: str, is_imported: bool = False) -> str:
    """Returns the SketchUp instance name for ``empty_obj``, or ``""`` when
    the Empty only carries its definition's name (with optional ``.001``
    Blender duplicate suffix) - SketchUp stores unnamed instances as ``""``
    rather than baking the definition's name into the instance."""
    raw_name = empty_obj.name
    if not raw_name or raw_name == def_base_name:
        return ""
    m = _BLENDER_DUP_SUFFIX_RE.match(raw_name)
    if m and m.group(1) == def_base_name:
        return ""
    if is_imported:
        base = _strip_blender_dup_suffix(raw_name, bpy.data.objects)
        if not base or base == def_base_name or re.match(r"^(?:mesh|Component)_\d+$", base):
            return ""
        return base
    if m and m.group(1) and f"{m.group(1)}.001" in bpy.data.objects and m.group(1) in bpy.data.objects:
        return m.group(1)
    return raw_name


def _faces_approx_equal(faces_a, faces_b, pos_tol_in=2e-3) -> bool:
    if len(faces_a) != len(faces_b):
        return False
    for (pts_a, spec_a, lay_a), (pts_b, spec_b, lay_b) in zip(faces_a, faces_b):
        if len(pts_a) != len(pts_b):
            return False
        if (spec_a[0] if spec_a else None) != (spec_b[0] if spec_b else None):
            return False
        if (lay_a or "") != (lay_b or ""):
            return False
        for pa, pb in zip(pts_a, pts_b):
            if (
                abs(pa[0] - pb[0]) > pos_tol_in
                or abs(pa[1] - pb[1]) > pos_tol_in
                or abs(pa[2] - pb[2]) > pos_tol_in
            ):
                return False
    return True


def _children_approx_equal(ch_a, ch_b, rot_tol=2e-4, pos_tol_in=2e-3) -> bool:
    if len(ch_a) != len(ch_b):
        return False
    ident3 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    for (dk_a, in_a, m3_a, tr_a, lay_a), (dk_b, in_b, m3_b, tr_b, lay_b) in zip(ch_a, ch_b):
        if dk_a != dk_b or (in_a or "") != (in_b or "") or (lay_a or "") != (lay_b or ""):
            return False
        ra = m3_a or ident3
        rb = m3_b or ident3
        if any(abs(a - b) > rot_tol for a, b in zip(ra, rb)):
            return False
        if (
            abs(tr_a[0] - tr_b[0]) > pos_tol_in
            or abs(tr_a[1] - tr_b[1]) > pos_tol_in
            or abs(tr_a[2] - tr_b[2]) > pos_tol_in
        ):
            return False
    return True


def _is_collection_empty(obj) -> bool:
    return (
        obj.type == "EMPTY"
        and obj.instance_type == "COLLECTION"
        and obj.instance_collection is not None
    )


def _is_group_empty(obj) -> bool:
    return (
        obj.type == "EMPTY"
        and not _is_collection_empty(obj)
        and "sketchup_definition" in obj
    )


def _iter_collection_objects(coll):
    """Yields unique viewport-enabled objects in ``coll`` and its
    viewport-enabled child collections (matching Blender's collection-instance
    evaluation)."""
    seen = set()

    def walk(c):
        if c.hide_viewport:
            return
        for o in c.objects:
            if o.hide_viewport:
                continue
            ptr = o.as_pointer()
            if ptr not in seen:
                seen.add(ptr)
                yield o
        for ch in c.children:
            yield from walk(ch)

    yield from walk(coll)


def _parent_relative_blender_matrix(ancestor_obj, desc_obj):
    """Computes ``ancestor_obj.matrix_world^-1 @ desc_obj.matrix_world`` from
    ``matrix_parent_inverse @ matrix_basis`` along the parent chain whenever
    the chain uses unconstrained object parenting, avoiding float32 rotation
    drift from world-matrix inversion (#28)."""
    steps = []
    cur = desc_obj
    while cur is not None and cur != ancestor_obj:
        if cur.parent is None or cur.parent_type != "OBJECT" or len(cur.constraints) > 0:
            inv = _try_invert(ancestor_obj.matrix_world)
            if inv is not None:
                return inv @ desc_obj.matrix_world
            return ancestor_obj.matrix_world.inverted_safe() @ desc_obj.matrix_world
        steps.append(cur.matrix_parent_inverse @ cur.matrix_basis)
        cur = cur.parent
    if cur != ancestor_obj:
        inv = _try_invert(ancestor_obj.matrix_world)
        if inv is not None:
            return inv @ desc_obj.matrix_world
        return ancestor_obj.matrix_world.inverted_safe() @ desc_obj.matrix_world
    rel = mathutils.Matrix.Identity(4)
    for step_m in reversed(steps):
        rel = rel @ step_m
    return rel


def export_skp(filepath, context):
    """Exports selected (or all visible) objects to a new .skp file at
    ``filepath``, preserving collection instances and imported hierarchy groups
    as SketchUp component definitions and instances (#28) while exporting plain
    mesh objects as loose root geometry."""
    from openskp import create

    depsgraph = context.evaluated_depsgraph_get()
    local_mesh_cache = {}
    imported_roots, source_collections, imported_tag_collections = _collect_import_collections()
    scene_collection = context.scene.collection

    selected_candidates = [
        o for o in context.selected_objects
        if o.type == "MESH" or _is_collection_empty(o) or _is_group_empty(o)
    ]
    selected_ptrs = {o.as_pointer() for o in context.selected_objects}
    explicit_selection = bool(selected_candidates)
    if selected_candidates:
        raw_candidates = selected_candidates
    else:
        raw_candidates = [
            o for o in context.scene.objects
            if (o.type == "MESH" or _is_collection_empty(o) or _is_group_empty(o))
            and o.visible_get()
        ]

    # Expand hierarchy descendants under any included imported group Empty,
    # walking through all intermediate object types (Mesh -> Mesh, Armature ->
    # Mesh, ordinary Empty -> Mesh) while skipping hidden children unless they
    # were explicitly selected themselves.
    expanded = list(raw_candidates)
    seen_ptrs = {o.as_pointer() for o in expanded}

    def is_child_included(ch) -> bool:
        if explicit_selection and ch.as_pointer() in selected_ptrs:
            return True
        return ch.visible_get() and not ch.hide_viewport

    def add_group_descendants(parent_obj):
        for ch in parent_obj.children:
            included = is_child_included(ch)
            if _is_group_empty(ch):
                if not included:
                    continue
                if ch.as_pointer() not in seen_ptrs:
                    seen_ptrs.add(ch.as_pointer())
                    expanded.append(ch)
                add_group_descendants(ch)
                continue
            if included and (ch.type == "MESH" or _is_collection_empty(ch)):
                if ch.as_pointer() not in seen_ptrs:
                    seen_ptrs.add(ch.as_pointer())
                    expanded.append(ch)
            add_group_descendants(ch)

    for obj in list(raw_candidates):
        if _is_group_empty(obj):
            add_group_descendants(obj)

    # Identify all Collections referenced via collection-instance Empties
    # (including nested collection-instance Empties inside child collections).
    instanced_colls = set()
    coll_skp_def_hint = {}

    def mark_instanced_collection(coll, def_hint=None):
        if coll is None:
            return
        is_lines_coll = re.sub(r"\.\d{3}$", "", coll.name).endswith(" (lines)")
        if def_hint and not is_lines_coll and coll.name not in coll_skp_def_hint:
            coll_skp_def_hint[coll.name] = def_hint
        if coll.name in instanced_colls:
            return
        instanced_colls.add(coll.name)
        for o in _iter_collection_objects(coll):
            if _is_collection_empty(o):
                mark_instanced_collection(o.instance_collection, o.get("sketchup_definition"))

    for obj in expanded:
        if _is_collection_empty(obj):
            mark_instanced_collection(obj.instance_collection, obj.get("sketchup_definition"))

    has_any_instances = bool(instanced_colls)

    def is_definition_master_mesh(obj) -> bool:
        if obj.type != "MESH" or not has_any_instances:
            return False
        # A mesh that is visible in the View Layer itself is a real scene object
        # as well as part of a collection definition - never drop the original.
        if obj.visible_get():
            return False
        if not obj.users_collection:
            return False
        return all(
            c.name in source_collections or c.name in instanced_colls
            for c in obj.users_collection
        )

    active_objs = [o for o in expanded if not is_definition_master_mesh(o)]
    active_ptrs = {o.as_pointer() for o in active_objs}

    def has_active_group_ancestor(obj) -> bool:
        p = obj.parent
        while p is not None:
            if p.as_pointer() in active_ptrs and _is_group_empty(p):
                return True
            p = p.parent
        return False

    top_items = [o for o in active_objs if not has_active_group_ancestor(o)]

    stats = {
        "faces_written": 0,
        "faces_skipped": 0,
        "objects_skipped": 0,
        "objects_exported": 0,
        "materials_written": 0,
        "layers_written": 0,
        "definitions_written": 0,
        "instances_written": 0,
    }

    def layer_of(obj, own_container_colls=()):
        return _object_layer_name(
            obj,
            scene_collection,
            imported_roots,
            source_collections,
            imported_tag_collections,
            instanced_colls,
            own_container_colls,
        )

    def uses_skp_yup_frame(obj) -> bool:
        if _is_collection_empty(obj):
            return obj.instance_collection.name in source_collections
        return "sketchup_definition" in obj

    def coll_coord_matrix(coll):
        return _YUP_TO_ZUP if coll.name in source_collections else mathutils.Matrix.Identity(4)

    def empty_skp_world_matrix(empty_obj):
        if uses_skp_yup_frame(empty_obj):
            return empty_obj.matrix_world @ _ZUP_TO_YUP
        return empty_obj.matrix_world.copy()

    def collection_base_name(coll) -> str:
        if coll.name in coll_skp_def_hint and coll_skp_def_hint[coll.name]:
            return coll_skp_def_hint[coll.name]
        if coll.name in source_collections:
            return _strip_blender_dup_suffix(coll.name, bpy.data.collections)
        return coll.name

    # Ordered bottom-up component definitions:
    # def_key -> {"base_name": str, "name": str, "faces": [(pts, spec, lname)],
    #             "children": [(child_key, inst_name, m3, tr, layer_name)]}
    definitions = {}
    def_order = []
    used_def_names = set()
    coll_def_cache = {}
    coll_faces_cache = {}
    visiting_colls = set()

    def alloc_def_name(base_name: str) -> str:
        clean = base_name or "Component"
        if clean not in used_def_names:
            used_def_names.add(clean)
            return clean
        idx = 1
        while f"{clean}#{idx}" in used_def_names:
            idx += 1
        candidate = f"{clean}#{idx}"
        used_def_names.add(candidate)
        return candidate

    def flatten_collection_faces(coll, world_prefix_bl, fallback_layer, out_faces, visited=frozenset()):
        """Bakes a collection's hierarchy into ``out_faces`` under ``world_prefix_bl``
        (used when an instance carries a singular/zero-axis scale matrix that
        SketchUp cannot store on a component instance)."""
        if coll is None or coll.name in visited:
            return 0
        next_visited = visited | {coll.name}
        offset_mat = mathutils.Matrix.Translation(-coll.instance_offset)
        own_containers = (coll.name,)
        added = 0
        for o in _iter_collection_objects(coll):
            m_obj = world_prefix_bl @ offset_mat @ o.matrix_world
            if o.type == "MESH" and o.data is not None:
                o_layer = layer_of(o, own_containers) or fallback_layer
                for pts, spec in _extract_mesh_faces(o, m_obj, stats, depsgraph, local_mesh_cache):
                    out_faces.append((pts, spec, o_layer))
                    added += 1
            elif _is_collection_empty(o):
                o_layer = layer_of(o, own_containers) or fallback_layer
                added += flatten_collection_faces(
                    o.instance_collection, m_obj, o_layer, out_faces, next_visited
                )
        return added

    def extract_collection_own_faces(coll):
        if coll.name in coll_faces_cache:
            return coll_faces_cache[coll.name]
        cmat = coll_coord_matrix(coll)
        offset_mat = mathutils.Matrix.Translation(-coll.instance_offset)
        own_containers = (coll.name,)
        faces = []
        for o in _iter_collection_objects(coll):
            if o.type == "MESH" and o.data is not None:
                m_local = cmat @ offset_mat @ o.matrix_world
                f_layer = layer_of(o, own_containers)
                for pts, spec in _extract_mesh_faces(o, m_local, stats, depsgraph, local_mesh_cache):
                    faces.append((pts, spec, f_layer))
        coll_faces_cache[coll.name] = faces
        return faces

    def resolve_collection_def(coll):
        if coll is None:
            return None
        if coll.name in coll_def_cache:
            return coll_def_cache[coll.name]
        if coll.name in visiting_colls:
            return None
        visiting_colls.add(coll.name)
        try:
            cmat = coll_coord_matrix(coll)
            offset_mat = mathutils.Matrix.Translation(-coll.instance_offset)
            own_containers = (coll.name,)
            faces = list(extract_collection_own_faces(coll))
            children = []
            for o in _iter_collection_objects(coll):
                if _is_collection_empty(o):
                    child_coll = o.instance_collection
                    m_bl = cmat @ offset_mat @ o.matrix_world
                    child_cmat_inv = coll_coord_matrix(child_coll).inverted()
                    m_local = m_bl @ child_cmat_inv
                    o_layer = layer_of(o, own_containers)
                    if _try_invert(m_local) is None:
                        flatten_collection_faces(
                            child_coll, m_bl, o_layer, faces, frozenset({coll.name})
                        )
                        continue
                    child_key = resolve_collection_def(child_coll)
                    if child_key is None:
                        continue
                    child_base = definitions[child_key]["base_name"]
                    m3, tr = _matrix_to_skp_args(m_local)
                    is_imp = child_coll.name in source_collections or ("sketchup_definition" in o)
                    children.append((
                        child_key,
                        _instance_name_for(o, child_base, is_imported=is_imp),
                        m3,
                        tr,
                        o_layer,
                    ))
            if not faces and not children:
                coll_def_cache[coll.name] = None
                return None
            base_name = collection_base_name(coll)
            def_key = ("coll", coll.name)
            definitions[def_key] = {
                "base_name": base_name,
                "name": alloc_def_name(base_name),
                "faces": faces,
                "children": children,
            }
            def_order.append(def_key)
            coll_def_cache[coll.name] = def_key
            return def_key
        finally:
            visiting_colls.discard(coll.name)

    group_defs_by_bucket = {}

    def relative_skp_matrix(grp_obj, desc_obj):
        rel_bl = _parent_relative_blender_matrix(grp_obj, desc_obj)
        left = _YUP_TO_ZUP if uses_skp_yup_frame(grp_obj) else mathutils.Matrix.Identity(4)
        right = _ZUP_TO_YUP if uses_skp_yup_frame(desc_obj) else mathutils.Matrix.Identity(4)
        return left @ rel_bl @ right

    def relative_mesh_matrix(grp_obj, desc_obj):
        rel_bl = _parent_relative_blender_matrix(grp_obj, desc_obj)
        left = _YUP_TO_ZUP if uses_skp_yup_frame(grp_obj) else mathutils.Matrix.Identity(4)
        return left @ rel_bl

    def is_own_geometry_child(grp_obj, ch, def_base_name: str) -> bool:
        if ch.parent != grp_obj or not _is_collection_empty(ch):
            return False
        if ch.instance_collection.name not in source_collections:
            return False
        ch_def = ch.get("sketchup_definition")
        coll_base = collection_base_name(ch.instance_collection)
        if ch_def is not None and ch_def != def_base_name:
            return False
        if ch_def is None and coll_base != def_base_name:
            return False
        if layer_of(ch) != layer_of(grp_obj):
            return False
        return _is_identity_matrix(relative_skp_matrix(grp_obj, ch))

    def flatten_group_subtree(grp_obj, prefix_to_grp_bl, fallback_layer, out_faces):
        """Flattens ``grp_obj``'s active subtree into ``out_faces`` using
        ``prefix_to_grp_bl`` (e.g. ``grp_obj.matrix_world`` at root when
        ``grp_obj`` carries a singular zero-axis scale)."""
        added = 0

        def walk_sub(parent_obj):
            nonlocal added
            for ch in parent_obj.children:
                if _is_group_empty(ch):
                    if ch.as_pointer() not in active_ptrs:
                        continue
                    m_ch_bl = prefix_to_grp_bl @ _parent_relative_blender_matrix(grp_obj, ch)
                    added += flatten_group_subtree(
                        ch, m_ch_bl, layer_of(ch) or fallback_layer, out_faces
                    )
                    continue
                if ch.as_pointer() in active_ptrs:
                    m_ch_bl = prefix_to_grp_bl @ _parent_relative_blender_matrix(grp_obj, ch)
                    ch_layer = layer_of(ch) or fallback_layer
                    if ch.type == "MESH" and ch.data is not None:
                        for pts, spec in _extract_mesh_faces(ch, m_ch_bl, stats, depsgraph, local_mesh_cache):
                            out_faces.append((pts, spec, ch_layer))
                            added += 1
                    elif _is_collection_empty(ch):
                        added += flatten_collection_faces(
                            ch.instance_collection, m_ch_bl, ch_layer, out_faces
                        )
                walk_sub(ch)

        walk_sub(grp_obj)
        return added

    def resolve_group_def(grp_obj):
        m_grp = empty_skp_world_matrix(grp_obj)
        if _try_invert(m_grp) is None:
            return None
        def_base_name = grp_obj.get("sketchup_definition") or grp_obj.name or "Group"

        faces = []
        children = []
        claimed_own_mesh = False

        def walk_group_children(parent_obj):
            nonlocal claimed_own_mesh
            for ch in parent_obj.children:
                if _is_group_empty(ch):
                    if ch.as_pointer() not in active_ptrs:
                        continue
                    m_rel = relative_skp_matrix(grp_obj, ch)
                    ch_layer = layer_of(ch)
                    if _try_invert(m_rel) is None:
                        m_rel_bl = relative_mesh_matrix(grp_obj, ch)
                        flatten_group_subtree(ch, m_rel_bl, ch_layer, faces)
                        continue
                    child_key = resolve_group_def(ch)
                    if child_key is None:
                        continue
                    m3, tr = _matrix_to_skp_args(m_rel)
                    child_base = definitions[child_key]["base_name"]
                    children.append((
                        child_key,
                        _instance_name_for(ch, child_base, is_imported=True),
                        m3,
                        tr,
                        ch_layer,
                    ))
                    continue

                if ch.as_pointer() in active_ptrs:
                    if not claimed_own_mesh and is_own_geometry_child(grp_obj, ch, def_base_name):
                        own_faces = extract_collection_own_faces(ch.instance_collection)
                        if own_faces:
                            faces.extend(own_faces)
                            claimed_own_mesh = True
                        walk_group_children(ch)
                        continue
                    if is_own_geometry_child(grp_obj, ch, def_base_name):
                        # Curve-only "(lines)" own child: 0 faces, skip instance creation.
                        walk_group_children(ch)
                        continue

                    if _is_collection_empty(ch):
                        child_coll = ch.instance_collection
                        m_rel = relative_skp_matrix(grp_obj, ch)
                        ch_layer = layer_of(ch)
                        if _try_invert(m_rel) is None:
                            m_rel_bl = relative_mesh_matrix(grp_obj, ch)
                            flatten_collection_faces(child_coll, m_rel_bl, ch_layer, faces)
                        else:
                            child_key = resolve_collection_def(child_coll)
                            if child_key is not None:
                                m3, tr = _matrix_to_skp_args(m_rel)
                                child_base = definitions[child_key]["base_name"]
                                is_imp = child_coll.name in source_collections or ("sketchup_definition" in ch)
                                children.append((
                                    child_key,
                                    _instance_name_for(ch, child_base, is_imported=is_imp),
                                    m3,
                                    tr,
                                    ch_layer,
                                ))
                    elif ch.type == "MESH" and ch.data is not None:
                        m_rel = relative_mesh_matrix(grp_obj, ch)
                        ch_layer = layer_of(ch)
                        for pts, spec in _extract_mesh_faces(ch, m_rel, stats, depsgraph, local_mesh_cache):
                            faces.append((pts, spec, ch_layer))

                walk_group_children(ch)

        walk_group_children(grp_obj)

        if not faces and not children:
            return None

        bucket_key = (def_base_name, len(faces), len(children))
        for existing_key in group_defs_by_bucket.get(bucket_key, ()):
            edef = definitions[existing_key]
            if _faces_approx_equal(faces, edef["faces"]) and _children_approx_equal(children, edef["children"]):
                return existing_key

        def_key = ("group", len(definitions))
        definitions[def_key] = {
            "base_name": def_base_name,
            "name": alloc_def_name(def_base_name),
            "faces": faces,
            "children": children,
        }
        def_order.append(def_key)
        group_defs_by_bucket.setdefault(bucket_key, []).append(def_key)
        return def_key

    # Count how many top-level Empties reference each ROOT_MODEL mesh collection
    # so a single unmoved ROOT_MODEL Empty from import_skp exports as loose root faces.
    root_model_top_empties = [
        o for o in top_items
        if _is_collection_empty(o)
        and o.instance_collection.name in source_collections
        and collection_base_name(o.instance_collection) == "ROOT_MODEL"
        and bool(extract_collection_own_faces(o.instance_collection))
    ]

    root_faces = []       # [(pts, mat_spec, layer_name)]
    root_instances = []   # [(def_key, inst_name, m3, tr, layer_name)]

    for obj in top_items:
        if obj.type == "MESH":
            if obj.data is None or (not obj.modifiers and len(obj.data.polygons) == 0):
                stats["objects_skipped"] += 1
                continue
            obj_faces = _extract_mesh_faces(obj, obj.matrix_world, stats, depsgraph, local_mesh_cache)
            if not obj_faces and len(obj.data.polygons) == 0:
                stats["objects_skipped"] += 1
                continue
            obj_layer = layer_of(obj)
            for pts, spec in obj_faces:
                root_faces.append((pts, spec, obj_layer))
            stats["objects_exported"] += 1
        elif _is_collection_empty(obj):
            coll = obj.instance_collection
            m_skp = empty_skp_world_matrix(obj)
            obj_layer = layer_of(obj)
            if (
                len(root_model_top_empties) == 1
                and obj is root_model_top_empties[0]
                and obj_layer is None
                and _is_identity_matrix(m_skp)
            ):
                own_faces = extract_collection_own_faces(coll)
                if not own_faces:
                    stats["objects_skipped"] += 1
                    continue
                for pts, spec, f_layer in own_faces:
                    root_faces.append((pts, spec, f_layer))
                stats["objects_exported"] += 1
                continue

            if _try_invert(m_skp) is None:
                added = flatten_collection_faces(coll, obj.matrix_world, obj_layer, root_faces)
                if added > 0:
                    stats["objects_exported"] += 1
                else:
                    stats["objects_skipped"] += 1
                continue

            def_key = resolve_collection_def(coll)
            if def_key is None:
                stats["objects_skipped"] += 1
                continue
            m3, tr = _matrix_to_skp_args(m_skp)
            def_base = definitions[def_key]["base_name"]
            is_imp = coll.name in source_collections or ("sketchup_definition" in obj)
            root_instances.append((
                def_key,
                _instance_name_for(obj, def_base, is_imported=is_imp),
                m3,
                tr,
                obj_layer,
            ))
            stats["objects_exported"] += 1
        elif _is_group_empty(obj):
            m_skp = empty_skp_world_matrix(obj)
            obj_layer = layer_of(obj)
            if _try_invert(m_skp) is None:
                added = flatten_group_subtree(obj, obj.matrix_world, obj_layer, root_faces)
                if added > 0:
                    stats["objects_exported"] += 1
                else:
                    stats["objects_skipped"] += 1
                continue

            def_key = resolve_group_def(obj)
            if def_key is None:
                if explicit_selection:
                    stats["objects_skipped"] += 1
                continue
            m3, tr = _matrix_to_skp_args(m_skp)
            def_base = definitions[def_key]["base_name"]
            root_instances.append((
                def_key,
                _instance_name_for(obj, def_base, is_imported=True),
                m3,
                tr,
                obj_layer,
            ))
            stats["objects_exported"] += 1

    # Prune any definition not reachable from root_instances (e.g. if only
    # ROOT_MODEL was present and exported as loose root faces).
    reachable_defs = set()

    def mark_reachable(dk):
        if dk in reachable_defs:
            return
        reachable_defs.add(dk)
        for child_key, *_ in definitions[dk]["children"]:
            mark_reachable(child_key)

    for dk, *_ in root_instances:
        mark_reachable(dk)

    ordered_defs = [dk for dk in def_order if dk in reachable_defs]

    # openskp's writer requires strict ordering:
    # 1. Materials  2. Layers  3. Component definitions (bottom-up)  4. Root geometry & instances
    builder = create()
    material_handle_by_key = {}
    used_mat_names = set()

    def ensure_material(spec):
        if spec is None:
            return None
        mkey, mname, rgba, opacity = spec
        if mkey not in material_handle_by_key:
            unique_mname = mname or "Material"
            if unique_mname in used_mat_names:
                idx = 1
                while f"{unique_mname}_{idx}" in used_mat_names:
                    idx += 1
                unique_mname = f"{unique_mname}_{idx}"
            used_mat_names.add(unique_mname)
            material_handle_by_key[mkey] = builder.add_material(unique_mname, rgba, opacity=opacity)
        return material_handle_by_key[mkey]

    for dk in ordered_defs:
        for _pts, spec, _lname in definitions[dk]["faces"]:
            ensure_material(spec)
    for _pts, spec, _layer_name in root_faces:
        ensure_material(spec)

    layer_handle_by_name = {}

    def ensure_layer(lname):
        if not lname or lname == "Layer0":
            return None
        if lname not in layer_handle_by_name:
            layer_handle_by_name[lname] = builder.add_layer(lname)
        return layer_handle_by_name[lname]

    for dk in ordered_defs:
        for _pts, _spec, lname in definitions[dk]["faces"]:
            ensure_layer(lname)
        for _ck, _iname, _m3, _tr, lname in definitions[dk]["children"]:
            ensure_layer(lname)
    for _pts, _spec, lname in root_faces:
        ensure_layer(lname)
    for _dk, _iname, _m3, _tr, lname in root_instances:
        ensure_layer(lname)

    def_handles = {}
    for dk in ordered_defs:
        dinfo = definitions[dk]
        with builder.add_component_definition(dinfo["name"]) as comp:
            for pts, spec, lname in dinfo["faces"]:
                try:
                    comp.add_face(
                        pts,
                        material=material_handle_by_key.get(spec[0]) if spec else None,
                        layer=layer_handle_by_name.get(lname) if lname else None,
                    )
                    stats["faces_written"] += 1
                except Exception:
                    stats["faces_skipped"] += 1
            for child_key, inst_name, m3, tr, lname in dinfo["children"]:
                comp.add_instance(
                    def_handles[child_key],
                    name=inst_name,
                    translation=tr,
                    matrix3x3=m3,
                    layer=layer_handle_by_name.get(lname) if lname else None,
                )
                stats["instances_written"] += 1
        def_handles[dk] = comp
        stats["definitions_written"] += 1

    for pts, spec, lname in root_faces:
        try:
            builder.add_face(
                pts,
                layer=layer_handle_by_name.get(lname) if lname else None,
                material=material_handle_by_key.get(spec[0]) if spec else None,
            )
            stats["faces_written"] += 1
        except Exception:
            stats["faces_skipped"] += 1

    for dk, inst_name, m3, tr, lname in root_instances:
        builder.add_instance(
            def_handles[dk],
            name=inst_name,
            translation=tr,
            matrix3x3=m3,
            layer=layer_handle_by_name.get(lname) if lname else None,
        )
        stats["instances_written"] += 1

    stats["materials_written"] = len(material_handle_by_key)
    stats["layers_written"] = len(layer_handle_by_name)

    with open(filepath, "wb") as f:
        f.write(builder.to_bytes())

    print(
        f"openskp export: {stats['objects_exported']} objects, "
        f"{stats['definitions_written']} definitions, {stats['instances_written']} instances, "
        f"{stats['faces_written']} faces written, {stats['faces_skipped']} skipped, "
        f"{stats['materials_written']} materials, {stats['layers_written']} layers"
    )
    return stats

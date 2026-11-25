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
  root-face export, applying active mesh modifiers, preserving planar n-gons
  (triangulating only genuinely non-planar polygons), per-face solid material
  colors/opacity, and per-object layer/tag assignment.
"""
from __future__ import annotations

import math
import re

import bpy
import mathutils

# 1 inch = 0.0254 m exactly - SketchUp's native unit is inches, Blender's is metres.
METRES_TO_INCHES = 1.0 / 0.0254
INCHES_TO_METRES = 0.0254

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

_UNPAINTED_SLOT_RE = re.compile(r"^openskp_material_\d+(?:\.\d{3,})?$")
_BLENDER_DUP_SUFFIX_RE = re.compile(r"^(.*)\.(\d{3,})$")


def _is_source_geom_name(name: str) -> bool:
    return re.sub(r"\.\d{3,}$", "", name).endswith("(source geometry)")


def _strip_blender_dup_suffix(name: str, existing_names) -> str:
    """Strips a trailing ``.001``..``.999``..``.1000+`` suffix only when the
    base name already exists in ``existing_names`` (meaning Blender
    auto-numbered a duplicate datablock), keeping real user names such as
    ``Cabinet.600``."""
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


def _collect_import_collections(all_collection_names):
    """Classifies collections created by ``import_skp`` so source-geometry
    collections convert back from glTF Y-up to SketchUp Z-up, imported tag
    collections strip any Blender ``.001`` duplicate suffix, and only imported
    ``Scenes`` / ``Annotations`` metadata collections (never user collections
    named ``Annotations``) are ignored as tags."""
    imported_roots = set()
    source_roots = []
    for coll in bpy.data.collections:
        if coll.get("openskp_imported_root", False):
            imported_roots.add(coll.name)
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
    imported_meta_collections = set()

    def mark_meta_subtree(c):
        if c.name in imported_meta_collections:
            return
        imported_meta_collections.add(c.name)
        for ch in c.children:
            mark_meta_subtree(ch)

    for root_name in imported_roots:
        root_coll = bpy.data.collections.get(root_name)
        if root_coll is None:
            continue
        for child in root_coll.children:
            if _is_source_geom_name(child.name):
                continue
            base = _strip_blender_dup_suffix(child.name, all_collection_names)
            if base in ("Scenes", "Annotations"):
                mark_meta_subtree(child)
                continue
            imported_tag_collections.add(child.name)
            for sub in child.children:
                sub_base = _strip_blender_dup_suffix(sub.name, all_collection_names)
                if sub_base == "Annotations":
                    mark_meta_subtree(sub)

    return imported_roots, source_collections, imported_tag_collections, imported_meta_collections


def _object_layer_name(
    coll_names_for_obj,
    scene_collection_name: str,
    imported_roots=(),
    source_collections=(),
    imported_tag_collections=(),
    imported_meta_collections=(),
    own_container_colls=(),
    all_collection_names=(),
):
    """The Collection an object's export should be tagged with in SketchUp -
    the first Collection it is linked into other than the scene's own default
    ones ("Scene Collection", "Collection"), an imported model root, an
    imported metadata collection ("Scenes"/"Annotations"), its own enclosing
    definition collection, or a "(source geometry)" collection. Returns ``None``
    for SketchUp's default untagged layer ("Layer0")."""
    for cname in coll_names_for_obj:
        if cname in (scene_collection_name, "Collection"):
            continue
        if (
            cname in imported_roots
            or cname in source_collections
            or cname in imported_meta_collections
            or cname in own_container_colls
        ):
            continue
        if _is_source_geom_name(cname):
            continue
        if cname in imported_tag_collections:
            tag = _strip_blender_dup_suffix(cname, all_collection_names)
        else:
            tag = cname
        if tag == "Layer0":
            return None
        return tag
    return None


def _linear_to_srgb(c):
    return c * 12.92 if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def _rgba_opacity_from_linear(lin_rgba):
    r, g, b, a = lin_rgba
    rgba = tuple(round(_linear_to_srgb(min(max(c, 0.0), 1.0)) * 255) for c in (r, g, b))
    opacity = None if a >= 0.999 else float(a)
    return rgba, opacity


def _material_rgba_opacity(mat):
    """A Material's solid display color as openskp's writer wants it:
    (r, g, b) ints 0-255, plus a separate 0.0-1.0 opacity (``None`` when
    fully opaque). ``diffuse_color`` is scene-linear and SketchUp colours are
    sRGB, so it is converted on the way out."""
    return _rgba_opacity_from_linear(mat.diffuse_color)


def _recolored_import_slot_rgba_opacity(mat):
    """Checks whether an unnamed ``openskp_material_N`` fallback material created
    by ``import_skp`` was recoloured by the user (via ``diffuse_color`` or via
    Principled BSDF ``Base Color``/``Alpha``). Returns ``(rgba, opacity)`` when
    recoloured, or ``None`` when still holding ``import_skp``'s fallback colour."""
    dc = tuple(float(x) for x in mat.diffuse_color)
    init = mat.get("openskp_initial_diffuse")
    init_tup = tuple(float(x) for x in init) if init is not None and len(init) == 4 else None

    bsdf_col = None
    if getattr(mat, "use_nodes", False) and getattr(mat, "node_tree", None) is not None:
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None and "Base Color" in bsdf.inputs and not bsdf.inputs["Base Color"].is_linked:
            bc = bsdf.inputs["Base Color"].default_value
            ba = bsdf.inputs["Alpha"].default_value if "Alpha" in bsdf.inputs else 1.0
            bsdf_col = (float(bc[0]), float(bc[1]), float(bc[2]), float(ba))

    if init_tup is not None:
        if any(abs(a - b) > 1e-4 for a, b in zip(dc, init_tup)):
            return _rgba_opacity_from_linear(dc)
        if bsdf_col is not None and any(abs(a - b) > 1e-4 for a, b in zip(bsdf_col, init_tup)):
            return _rgba_opacity_from_linear(bsdf_col)
        return None

    if bsdf_col is not None and any(abs(a - b) > 1e-4 for a, b in zip(dc, bsdf_col)):
        return _rgba_opacity_from_linear(dc)
    return None


def _material_spec(mat, all_material_names=None):
    """Returns ``(dedup_key, skp_name, rgba, opacity)`` for a Blender Material,
    or ``None`` when unpainted (including untouched ``import_skp`` unnamed
    fallback slots ``openskp_material_N``, while exporting them if recoloured)."""
    if mat is None:
        return None
    if all_material_names is None:
        all_material_names = bpy.data.materials
    if "openskp_material_index" in mat:
        base = _strip_blender_dup_suffix(mat.name, all_material_names)
        if "openskp_back_material_index" in mat and " | " in base:
            front_name, back_name = base.split(" | ", 1)
            front_unpainted = bool(_UNPAINTED_SLOT_RE.match(front_name))
            back_unpainted = bool(_UNPAINTED_SLOT_RE.match(back_name))
            if front_unpainted and back_unpainted:
                recol = _recolored_import_slot_rgba_opacity(mat)
                if recol is None:
                    return None
                rgba, opacity = recol
                return (f"skp:{mat.name}:{rgba}:{opacity}", mat.name, rgba, opacity)
            skp_name = back_name if front_unpainted else front_name
        elif _UNPAINTED_SLOT_RE.match(base):
            recol = _recolored_import_slot_rgba_opacity(mat)
            if recol is None:
                return None
            rgba, opacity = recol
            return (f"skp:{mat.name}:{rgba}:{opacity}", mat.name, rgba, opacity)
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


def _get_mesh_polys_and_materials(obj, depsgraph=None, local_mesh_cache=None):
    """Returns ``(local_polys, local_skipped, materials_list)`` for ``obj``,
    caching per ``Mesh`` datablock when ``obj`` carries no modifiers and reading
    evaluated ``mesh.materials`` when modifiers (such as Geometry Nodes
    ``Set Material``) are present (#28)."""
    if obj.data is None:
        return (), 0, ()
    if not obj.modifiers:
        key = obj.data.as_pointer()
        if local_mesh_cache is not None and key in local_mesh_cache:
            local_polys, local_skipped = local_mesh_cache[key]
        else:
            mesh = obj.to_mesh()
            try:
                local_polys, local_skipped = _build_local_mesh_polys(mesh)
            finally:
                obj.to_mesh_clear()
            if local_mesh_cache is not None:
                local_mesh_cache[key] = (local_polys, local_skipped)
        mats = [slot.material for slot in obj.material_slots]
        if not mats and obj.data.materials:
            mats = list(obj.data.materials)
        return local_polys, local_skipped, mats

    eval_obj = obj.evaluated_get(depsgraph) if depsgraph is not None else obj
    mesh = eval_obj.to_mesh()
    try:
        local_polys, local_skipped = _build_local_mesh_polys(mesh)
        eval_slots = eval_obj.material_slots
        mats = []
        n_mats = max(len(mesh.materials), len(eval_slots))
        for idx in range(n_mats):
            if idx < len(eval_slots) and eval_slots[idx].link == "OBJECT":
                mats.append(eval_slots[idx].material)
            elif idx < len(mesh.materials):
                mats.append(mesh.materials[idx])
            elif idx < len(eval_slots):
                mats.append(eval_slots[idx].material)
        return local_polys, local_skipped, mats
    finally:
        eval_obj.to_mesh_clear()


def _extract_mesh_faces(
    obj,
    transform,
    stats,
    depsgraph=None,
    local_mesh_cache=None,
    mat_spec_fn=None,
):
    """Extracts ``(pts_inches, mat_spec)`` tuples from ``obj`` transformed by
    4x4 metre matrix ``transform``."""
    local_polys, local_skipped, mats = _get_mesh_polys_and_materials(
        obj, depsgraph, local_mesh_cache
    )
    stats["faces_skipped"] += local_skipped
    if not local_polys:
        return []

    resolve_mat = mat_spec_fn if mat_spec_fn is not None else _material_spec
    specs = [resolve_mat(m) for m in mats]
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


def _faces_max_radius_in(faces) -> float:
    max_r_sq = 0.0
    for pts, _spec, _lay in faces:
        for px, py, pz in pts:
            r_sq = px * px + py * py + pz * pz
            if r_sq > max_r_sq:
                max_r_sq = r_sq
    return math.sqrt(max_r_sq)


def _m3_scale_bound(m3) -> float:
    if m3 is None:
        return 1.0
    r0 = m3[0] * m3[0] + m3[1] * m3[1] + m3[2] * m3[2]
    r1 = m3[3] * m3[3] + m3[4] * m3[4] + m3[5] * m3[5]
    r2 = m3[6] * m3[6] + m3[7] * m3[7] + m3[8] * m3[8]
    return math.sqrt(max(r0, r1, r2))


def _compute_def_max_radius_in(faces, children, definitions) -> float:
    max_r = _faces_max_radius_in(faces)
    for child_key, _iname, m3, tr, _lay in children:
        child_r = definitions[child_key]["max_radius_in"]
        tr_len = math.sqrt(tr[0] * tr[0] + tr[1] * tr[1] + tr[2] * tr[2])
        bound = tr_len + _m3_scale_bound(m3) * child_r
        if bound > max_r:
            max_r = bound
    return max_r


def _is_identity_matrix(m4, radius_in: float = 0.0, rot_tol=1e-4, trans_tol_m=1e-4) -> bool:
    radius_m = radius_in * INCHES_TO_METRES
    eff_rot_tol = min(rot_tol, trans_tol_m / max(1.0, radius_m))
    ident = mathutils.Matrix.Identity(4)
    for r in range(4):
        for c in range(4):
            limit = trans_tol_m if c == 3 and r < 3 else eff_rot_tol
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


def _instance_name_for(
    empty_obj,
    def_base_name: str,
    is_imported: bool = False,
    all_object_names=None,
) -> str:
    """Returns the SketchUp instance name for ``empty_obj``, or ``""`` when
    the Empty only carries its definition's name (with optional ``.001`` /
    ``.1000+`` Blender duplicate suffix) - SketchUp stores unnamed instances as
    ``""`` rather than baking the definition's name into the instance."""
    raw_name = empty_obj.name
    if not raw_name or raw_name == def_base_name:
        return ""
    m = _BLENDER_DUP_SUFFIX_RE.match(raw_name)
    if m and m.group(1) == def_base_name:
        return ""
    if all_object_names is None:
        all_object_names = bpy.data.objects
    if is_imported:
        base = _strip_blender_dup_suffix(raw_name, all_object_names)
        if not base or base == def_base_name or re.match(r"^(?:mesh|Component)_\d+$", base):
            return ""
        return base
    if m and m.group(1) and f"{m.group(1)}.001" in all_object_names and m.group(1) in all_object_names:
        return m.group(1)
    return raw_name


def _faces_approx_equal(faces_a, faces_b, pos_tol_in=2e-3) -> bool:
    if faces_a is faces_b:
        return True
    if len(faces_a) != len(faces_b):
        return False
    for fa, fb in zip(faces_a, faces_b):
        if fa is fb:
            continue
        pts_a, spec_a, lay_a = fa
        pts_b, spec_b, lay_b = fb
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


def _children_approx_equal(
    ch_a,
    ch_b,
    definitions,
    rot_tol=2e-4,
    pos_tol_in=2e-3,
) -> bool:
    """Compares two child-instance lists for definition deduplication, scaling
    the allowed 3x3 matrix tolerance with each child definition's radius so a
    100 m beam rotated by 1.9e-4 rad is never merged with an unrotated one, and
    ignoring child instance names so Blender's auto-numbered child Empties do
    not split identical definitions (#28)."""
    if len(ch_a) != len(ch_b):
        return False
    ident3 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    for (dk_a, _in_a, m3_a, tr_a, lay_a), (dk_b, _in_b, m3_b, tr_b, lay_b) in zip(ch_a, ch_b):
        if dk_a != dk_b or (lay_a or "") != (lay_b or ""):
            return False
        child_r = definitions[dk_a]["max_radius_in"] if dk_a in definitions else 1.0
        eff_rot_tol = min(rot_tol, pos_tol_in / max(1.0, child_r))
        ra = m3_a or ident3
        rb = m3_b or ident3
        if any(abs(a - b) > eff_rot_tol for a, b in zip(ra, rb)):
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


def _world_relative_fallback(ancestor_obj, desc_obj):
    inv = _try_invert(ancestor_obj.matrix_world)
    if inv is not None:
        return inv @ desc_obj.matrix_world
    return ancestor_obj.matrix_world.inverted_safe() @ desc_obj.matrix_world


def _parent_relative_blender_matrix(ancestor_obj, desc_obj):
    """Computes ``ancestor_obj.matrix_world^-1 @ desc_obj.matrix_world`` from
    ``matrix_parent_inverse @ matrix_basis`` along the parent chain whenever
    that chain agrees with ``desc_obj.matrix_world``, falling back to
    ``matrix_world`` whenever constraints, Follow Path curves, or drivers move
    the object (#28)."""
    steps = []
    cur = desc_obj
    while cur is not None and cur != ancestor_obj:
        if cur.parent is None or cur.parent_type != "OBJECT" or len(cur.constraints) > 0:
            return _world_relative_fallback(ancestor_obj, desc_obj)
        steps.append(cur.matrix_parent_inverse @ cur.matrix_basis)
        cur = cur.parent
    if cur != ancestor_obj:
        return _world_relative_fallback(ancestor_obj, desc_obj)
    rel = mathutils.Matrix.Identity(4)
    for step_m in reversed(steps):
        rel = rel @ step_m
    mw_desc = desc_obj.matrix_world
    pred_world = ancestor_obj.matrix_world @ rel
    for r in range(4):
        for c in range(4):
            ref_mag = max(1.0, abs(mw_desc[r][c]))
            if abs(pred_world[r][c] - mw_desc[r][c]) > 1e-4 * ref_mag:
                return _world_relative_fallback(ancestor_obj, desc_obj)
    return rel


def export_skp(filepath, context):
    """Exports selected (or all visible) objects to a new .skp file at
    ``filepath``, preserving collection instances and imported hierarchy groups
    as SketchUp component definitions and instances (#28) while exporting plain
    mesh objects as loose root geometry."""
    from openskp import create

    depsgraph = context.evaluated_depsgraph_get()
    local_mesh_cache = {}
    all_object_names = {o.name for o in bpy.data.objects}
    all_collection_names = {c.name for c in bpy.data.collections}
    all_material_names = {m.name for m in bpy.data.materials}

    # Build O(1) parent -> children and object -> collection-names maps once per
    # export (Object.children and Object.users_collection scan the whole blend
    # file linearly on each access, making large scenes quadratic).
    children_by_ptr = {}
    for o in context.scene.objects:
        p = o.parent
        if p is not None:
            children_by_ptr.setdefault(p.as_pointer(), []).append(o)

    def children_of(parent_obj):
        return children_by_ptr.get(parent_obj.as_pointer(), ())

    user_colls_by_ptr = {}

    def record_coll_tree(c):
        cname = c.name
        for o in c.objects:
            lst = user_colls_by_ptr.setdefault(o.as_pointer(), [])
            if cname not in lst:
                lst.append(cname)

    record_coll_tree(context.scene.collection)
    for c in bpy.data.collections:
        record_coll_tree(c)

    (
        imported_roots,
        source_collections,
        imported_tag_collections,
        imported_meta_collections,
    ) = _collect_import_collections(all_collection_names)
    scene_collection_name = context.scene.collection.name

    mat_spec_cache = {}

    def cached_material_spec(mat):
        if mat is None:
            return None
        mptr = mat.as_pointer()
        if mptr not in mat_spec_cache:
            mat_spec_cache[mptr] = _material_spec(mat, all_material_names)
        return mat_spec_cache[mptr]

    coll_objects_cache = {}

    def coll_objects(coll):
        if coll.name not in coll_objects_cache:
            coll_objects_cache[coll.name] = tuple(_iter_collection_objects(coll))
        return coll_objects_cache[coll.name]

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
    # walking through ALL intermediate object types — including hidden or
    # excluded nested group Empties — so active descendants below an inactive
    # intermediate group are never dropped.
    expanded = list(raw_candidates)
    seen_ptrs = {o.as_pointer() for o in expanded}

    def is_child_included(ch) -> bool:
        if explicit_selection and ch.as_pointer() in selected_ptrs:
            return True
        return ch.visible_get() and not ch.hide_viewport

    def add_group_descendants(parent_obj):
        for ch in children_of(parent_obj):
            if is_child_included(ch) and (
                ch.type == "MESH" or _is_collection_empty(ch) or _is_group_empty(ch)
            ):
                ptr = ch.as_pointer()
                if ptr not in seen_ptrs:
                    seen_ptrs.add(ptr)
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
        is_lines_coll = re.sub(r"\.\d{3,}$", "", coll.name).endswith(" (lines)")
        if def_hint and not is_lines_coll and coll.name not in coll_skp_def_hint:
            coll_skp_def_hint[coll.name] = def_hint
        if coll.name in instanced_colls:
            return
        instanced_colls.add(coll.name)
        for o in coll_objects(coll):
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
        ucolls = user_colls_by_ptr.get(obj.as_pointer())
        if not ucolls:
            return False
        return all(
            cname in source_collections or cname in instanced_colls
            for cname in ucolls
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

    obj_layer_cache = {}

    def layer_of(obj, own_container_colls=()):
        cache_key = (obj.as_pointer(), own_container_colls)
        if cache_key not in obj_layer_cache:
            ucolls = user_colls_by_ptr.get(obj.as_pointer(), ())
            obj_layer_cache[cache_key] = _object_layer_name(
                ucolls,
                scene_collection_name,
                imported_roots,
                source_collections,
                imported_tag_collections,
                imported_meta_collections,
                own_container_colls,
                all_collection_names,
            )
        return obj_layer_cache[cache_key]

    def uses_skp_yup_frame(obj) -> bool:
        if obj.get("openskp_zup_frame", False):
            return False
        if _is_collection_empty(obj):
            return obj.instance_collection.name in source_collections
        return "sketchup_definition" in obj

    def is_linked_component_mesh(obj) -> bool:
        return (
            obj.type == "MESH"
            and obj.data is not None
            and not obj.get("sketchup_own_geometry", False)
            and (
                bool(obj.get("sketchup_is_component", False))
                or bool(obj.get("sketchup_is_group", False))
                or ("sketchup_component_definition" in obj.data)
            )
        )

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
            return _strip_blender_dup_suffix(coll.name, all_collection_names)
        return coll.name

    # Ordered bottom-up component definitions:
    # def_key -> {"base_name": str, "name": str, "faces": [(pts, spec, lname)],
    #             "children": [(child_key, inst_name, m3, tr, layer_name)],
    #             "max_radius_in": float}
    definitions = {}
    def_order = []
    used_def_names = set()
    coll_def_cache = {}
    coll_faces_cache = {}
    coll_own_radius_cache = {}
    visiting_colls = set()
    linked_mesh_def_cache = {}
    mesh_defs_by_bucket = {}
    own_mesh_faces_cache = {}
    own_mesh_radius_cache = {}

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

    def linked_mesh_base_name(obj) -> str:
        hint = obj.get("sketchup_definition") or obj.data.get("sketchup_component_definition")
        if hint:
            return str(hint)
        if obj.get("sketchup_is_group", False):
            return _strip_blender_dup_suffix(obj.name, all_object_names) or "Group"
        return _strip_blender_dup_suffix(obj.data.name, all_object_names) or "Component"

    def _object_unmodified_specs(obj):
        mats = [slot.material for slot in obj.material_slots]
        if not mats and obj.data.materials:
            mats = list(obj.data.materials)
        specs = [cached_material_spec(m) for m in mats]
        specs_key = tuple(s[0] if s is not None else None for s in specs)
        return specs, specs_key

    def _local_polys_to_inch_faces(local_polys, specs, face_layer=None):
        num_specs = len(specs)
        faces = []
        skipped = 0
        for cos, mat_index in local_polys:
            pts = [
                (co.x * METRES_TO_INCHES, co.y * METRES_TO_INCHES, co.z * METRES_TO_INCHES)
                for co in cos
            ]
            if _polygon_area_sq(pts) == 0.0:
                skipped += 1
                continue
            spec = specs[mat_index] if 0 <= mat_index < num_specs else None
            faces.append((pts, spec, face_layer))
        return faces, skipped

    def extract_own_mesh_faces_cached(ch, own_face_layer):
        specs, specs_key = _object_unmodified_specs(ch)
        cache_key = (ch.data.as_pointer(), specs_key, own_face_layer)
        if cache_key in own_mesh_faces_cache:
            return own_mesh_faces_cache[cache_key], own_mesh_radius_cache[cache_key]
        local_polys, local_skipped, _ = _get_mesh_polys_and_materials(
            ch, depsgraph, local_mesh_cache
        )
        faces_list, area_skipped = _local_polys_to_inch_faces(local_polys, specs, own_face_layer)
        stats["faces_skipped"] += local_skipped + area_skipped
        faces_tup = tuple(faces_list)
        radius_in = _faces_max_radius_in(faces_tup)
        own_mesh_faces_cache[cache_key] = faces_tup
        own_mesh_radius_cache[cache_key] = radius_in
        return faces_tup, radius_in

    def resolve_linked_mesh_def(obj):
        is_group_mesh = bool(obj.get("sketchup_is_group", False))
        has_mods = bool(obj.modifiers)

        if not has_mods:
            specs, specs_key = _object_unmodified_specs(obj)
            ptr_key = (obj.data.as_pointer(), specs_key)
            if ptr_key in linked_mesh_def_cache:
                return linked_mesh_def_cache[ptr_key]

            local_polys, local_skipped, _ = _get_mesh_polys_and_materials(
                obj, depsgraph, local_mesh_cache
            )
            if not local_polys:
                stats["faces_skipped"] += local_skipped
                linked_mesh_def_cache[ptr_key] = None
                return None

            faces, area_skipped = _local_polys_to_inch_faces(local_polys, specs, None)
            if not faces:
                stats["faces_skipped"] += local_skipped + area_skipped
                linked_mesh_def_cache[ptr_key] = None
                return None

            base_name = linked_mesh_base_name(obj)
            if is_group_mesh:
                bucket_key = ("group_mesh", base_name, len(faces), specs_key)
                for existing_key in mesh_defs_by_bucket.get(bucket_key, ()):
                    if _faces_approx_equal(faces, definitions[existing_key]["faces"]):
                        linked_mesh_def_cache[ptr_key] = existing_key
                        return existing_key

            stats["faces_skipped"] += local_skipped + area_skipped
            def_key = ("mesh", len(definitions))
            definitions[def_key] = {
                "base_name": base_name,
                "name": alloc_def_name(base_name),
                "faces": faces,
                "children": [],
                "max_radius_in": _compute_def_max_radius_in(faces, [], definitions),
            }
            def_order.append(def_key)
            linked_mesh_def_cache[ptr_key] = def_key
            if is_group_mesh:
                mesh_defs_by_bucket.setdefault(
                    ("group_mesh", base_name, len(faces), specs_key), []
                ).append(def_key)
            return def_key

        local_polys, local_skipped, mats = _get_mesh_polys_and_materials(
            obj, depsgraph, local_mesh_cache
        )
        if not local_polys:
            stats["faces_skipped"] += local_skipped
            return None
        specs = [cached_material_spec(m) for m in mats]
        specs_key = tuple(s[0] if s is not None else None for s in specs)
        faces, area_skipped = _local_polys_to_inch_faces(local_polys, specs, None)
        if not faces:
            stats["faces_skipped"] += local_skipped + area_skipped
            return None

        base_name = linked_mesh_base_name(obj)
        if not is_group_mesh:
            unmod_key = linked_mesh_def_cache.get((obj.data.as_pointer(), specs_key))
            if unmod_key is not None and _faces_approx_equal(faces, definitions[unmod_key]["faces"]):
                return unmod_key

        kind = "group_mesh" if is_group_mesh else "mod_comp_mesh"
        bucket_key = (kind, base_name, len(faces), specs_key)
        for existing_key in mesh_defs_by_bucket.get(bucket_key, ()):
            if _faces_approx_equal(faces, definitions[existing_key]["faces"]):
                return existing_key

        stats["faces_skipped"] += local_skipped + area_skipped
        def_key = ("mesh", len(definitions))
        definitions[def_key] = {
            "base_name": base_name,
            "name": alloc_def_name(base_name),
            "faces": faces,
            "children": [],
            "max_radius_in": _compute_def_max_radius_in(faces, [], definitions),
        }
        def_order.append(def_key)
        mesh_defs_by_bucket.setdefault(bucket_key, []).append(def_key)
        return def_key

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
        for o in coll_objects(coll):
            m_obj = world_prefix_bl @ offset_mat @ o.matrix_world
            if o.type == "MESH" and o.data is not None:
                o_layer = layer_of(o, own_containers) or fallback_layer
                for pts, spec in _extract_mesh_faces(
                    o, m_obj, stats, depsgraph, local_mesh_cache, cached_material_spec
                ):
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
        for o in coll_objects(coll):
            if o.type == "MESH" and o.data is not None:
                m_local = cmat @ offset_mat @ o.matrix_world
                f_layer = layer_of(o, own_containers)
                for pts, spec in _extract_mesh_faces(
                    o, m_local, stats, depsgraph, local_mesh_cache, cached_material_spec
                ):
                    faces.append((pts, spec, f_layer))
        coll_faces_cache[coll.name] = faces
        coll_own_radius_cache[coll.name] = _faces_max_radius_in(faces)
        return faces

    def collection_own_radius_in(coll) -> float:
        if coll.name not in coll_own_radius_cache:
            extract_collection_own_faces(coll)
        return coll_own_radius_cache[coll.name]

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
            for o in coll_objects(coll):
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
                        _instance_name_for(o, child_base, is_imported=is_imp, all_object_names=all_object_names),
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
                "max_radius_in": _compute_def_max_radius_in(faces, children, definitions),
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
        own_r = collection_own_radius_in(ch.instance_collection)
        return _is_identity_matrix(relative_skp_matrix(grp_obj, ch), radius_in=own_r)

    def flatten_group_subtree(grp_obj, prefix_to_grp_bl, fallback_layer, out_faces):
        """Flattens ``grp_obj``'s active subtree into ``out_faces`` using
        ``prefix_to_grp_bl``, walking through inactive intermediate group
        Empties so active descendants below them are preserved."""
        added = 0

        def walk_sub(parent_obj):
            nonlocal added
            for ch in children_of(parent_obj):
                if _is_group_empty(ch):
                    if ch.as_pointer() not in active_ptrs:
                        walk_sub(ch)
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
                        for pts, spec in _extract_mesh_faces(
                            ch, m_ch_bl, stats, depsgraph, local_mesh_cache, cached_material_spec
                        ):
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
        grp_layer = layer_of(grp_obj)

        faces = []
        children = []
        claimed_own_mesh = False

        def walk_group_children(parent_obj):
            nonlocal claimed_own_mesh
            for ch in children_of(parent_obj):
                if _is_group_empty(ch):
                    if ch.as_pointer() not in active_ptrs:
                        walk_group_children(ch)
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
                        _instance_name_for(ch, child_base, is_imported=True, all_object_names=all_object_names),
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
                                    _instance_name_for(
                                        ch, child_base, is_imported=is_imp, all_object_names=all_object_names
                                    ),
                                    m3,
                                    tr,
                                    ch_layer,
                                ))
                    elif ch.parent == grp_obj and ch.type == "MESH" and ch.get("sketchup_own_geometry", False):
                        if ch.data is not None:
                            m_rel = relative_mesh_matrix(grp_obj, ch)
                            ch_layer = layer_of(ch)
                            own_face_layer = None if ch_layer == grp_layer else ch_layer
                            if not ch.modifiers:
                                own_faces_tup, own_r = extract_own_mesh_faces_cached(ch, own_face_layer)
                                if _is_identity_matrix(m_rel, radius_in=own_r):
                                    faces.extend(own_faces_tup)
                                else:
                                    for pts, spec in _extract_mesh_faces(
                                        ch, m_rel, stats, depsgraph, local_mesh_cache, cached_material_spec
                                    ):
                                        faces.append((pts, spec, own_face_layer))
                            else:
                                for pts, spec in _extract_mesh_faces(
                                    ch, m_rel, stats, depsgraph, local_mesh_cache, cached_material_spec
                                ):
                                    faces.append((pts, spec, own_face_layer))
                    elif is_linked_component_mesh(ch):
                        m_rel = relative_skp_matrix(grp_obj, ch)
                        ch_layer = layer_of(ch)
                        if _try_invert(m_rel) is None:
                            m_rel_bl = relative_mesh_matrix(grp_obj, ch)
                            for pts, spec in _extract_mesh_faces(
                                ch, m_rel_bl, stats, depsgraph, local_mesh_cache, cached_material_spec
                            ):
                                faces.append((pts, spec, ch_layer))
                        else:
                            child_key = resolve_linked_mesh_def(ch)
                            if child_key is not None:
                                m3, tr = _matrix_to_skp_args(m_rel)
                                child_base = definitions[child_key]["base_name"]
                                children.append((
                                    child_key,
                                    _instance_name_for(
                                        ch, child_base, is_imported=True, all_object_names=all_object_names
                                    ),
                                    m3,
                                    tr,
                                    ch_layer,
                                ))
                    elif ch.type == "MESH" and ch.data is not None:
                        m_rel = relative_mesh_matrix(grp_obj, ch)
                        ch_layer = layer_of(ch)
                        for pts, spec in _extract_mesh_faces(
                            ch, m_rel, stats, depsgraph, local_mesh_cache, cached_material_spec
                        ):
                            faces.append((pts, spec, ch_layer))

                walk_group_children(ch)

        walk_group_children(grp_obj)

        if not faces and not children:
            return None

        bucket_key = (def_base_name, len(faces), len(children))
        for existing_key in group_defs_by_bucket.get(bucket_key, ()):
            edef = definitions[existing_key]
            if _faces_approx_equal(faces, edef["faces"]) and _children_approx_equal(
                children, edef["children"], definitions
            ):
                return existing_key

        def_key = ("group", len(definitions))
        definitions[def_key] = {
            "base_name": def_base_name,
            "name": alloc_def_name(def_base_name),
            "faces": faces,
            "children": children,
            "max_radius_in": _compute_def_max_radius_in(faces, children, definitions),
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
        if is_linked_component_mesh(obj):
            m_skp = empty_skp_world_matrix(obj)
            obj_layer = layer_of(obj)
            if _try_invert(m_skp) is None:
                obj_faces = _extract_mesh_faces(
                    obj, obj.matrix_world, stats, depsgraph, local_mesh_cache, cached_material_spec
                )
                if obj_faces:
                    for pts, spec in obj_faces:
                        root_faces.append((pts, spec, obj_layer))
                    stats["objects_exported"] += 1
                else:
                    stats["objects_skipped"] += 1
                continue

            def_key = resolve_linked_mesh_def(obj)
            if def_key is None:
                stats["objects_skipped"] += 1
                continue
            m3, tr = _matrix_to_skp_args(m_skp)
            def_base = definitions[def_key]["base_name"]
            root_instances.append((
                def_key,
                _instance_name_for(obj, def_base, is_imported=True, all_object_names=all_object_names),
                m3,
                tr,
                obj_layer,
            ))
            stats["objects_exported"] += 1
        elif obj.type == "MESH":
            if obj.data is None or (not obj.modifiers and len(obj.data.polygons) == 0):
                stats["objects_skipped"] += 1
                continue
            obj_faces = _extract_mesh_faces(
                obj, obj.matrix_world, stats, depsgraph, local_mesh_cache, cached_material_spec
            )
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
                and _is_identity_matrix(m_skp, radius_in=collection_own_radius_in(coll))
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
                _instance_name_for(obj, def_base, is_imported=is_imp, all_object_names=all_object_names),
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
                _instance_name_for(obj, def_base, is_imported=True, all_object_names=all_object_names),
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

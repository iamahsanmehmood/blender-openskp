"""Blender exporter for native SketchUp (.skp) files, built on OpenSKP.

Exports selected objects (or every visible object when nothing is selected) to
a new ``.skp`` file:

- **Collection instances as SketchUp components (#28)**: each instanced
  ``Collection`` (and each ``keep_hierarchy`` parented group/component Empty
  tree) is written once in bottom-up order via ``add_component_definition``,
  and each placement ``Empty`` is written via ``add_instance`` carrying its
  4x4 transform (position, rotation, non-uniform scale and mirroring) and
  SketchUp tag (Collection membership).
- **Plain mesh objects as loose root geometry**: standalone ``MESH`` objects
  outside instanced definitions keep the flat root-face export, preserving
  planar n-gons (triangulating only genuinely non-planar polygons), per-face
  solid material colors/opacity, and per-object layer/tag assignment.
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

_UNPAINTED_SLOT_RE = re.compile(r"^openskp_material_\d+$")


def _strip_num_suffix(name: str) -> str:
    return re.sub(r"\.\d{3}$", "", name)


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
    collections strip any Blender ``.001`` auto-suffix."""
    imported_roots = set()
    source_roots = []
    for coll in bpy.data.collections:
        for child in coll.children:
            if _strip_num_suffix(child.name).endswith("(source geometry)"):
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
            base = _strip_num_suffix(child.name)
            if base.endswith("(source geometry)") or base == "Scenes":
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
        if cname in imported_roots or cname in source_collections or cname in instanced_colls:
            continue
        if _strip_num_suffix(cname).endswith("(source geometry)"):
            continue
        tag = _strip_num_suffix(cname) if cname in imported_tag_collections else cname
        if tag == "Layer0":
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
        base = _strip_num_suffix(mat.name)
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


def _extract_mesh_faces(obj, transform, stats):
    """Extracts ``(pts_inches, mat_spec)`` tuples from ``obj`` transformed by
    4x4 metre matrix ``transform``."""
    faces = []
    mesh = obj.to_mesh()
    try:
        specs = [_material_spec(m) for m in (obj.data.materials if obj.data else ())]

        def spec_for(index):
            return specs[index] if 0 <= index < len(specs) else None

        needs_triangulation = [
            poly.index for poly in mesh.polygons if not _is_planar(mesh, poly)
        ]

        for poly in mesh.polygons:
            if poly.index in needs_triangulation:
                continue
            local_pts = [tuple(mesh.vertices[vi].co) for vi in poly.vertices]
            if _polygon_area_sq(local_pts) <= 1e-24:
                stats["faces_skipped"] += 1
                continue
            pts = [_world_point(transform, mesh.vertices[vi].co) for vi in poly.vertices]
            if _polygon_area_sq(pts) <= 1e-16:
                stats["faces_skipped"] += 1
                continue
            faces.append((pts, spec_for(poly.material_index)))

        if needs_triangulation:
            mesh.calc_loop_triangles()
            wanted = set(needs_triangulation)
            for tri in mesh.loop_triangles:
                if tri.polygon_index not in wanted:
                    continue
                local_pts = [tuple(mesh.vertices[vi].co) for vi in tri.vertices]
                if _polygon_area_sq(local_pts) <= 1e-24:
                    stats["faces_skipped"] += 1
                    continue
                pts = [_world_point(transform, mesh.vertices[vi].co) for vi in tri.vertices]
                if _polygon_area_sq(pts) <= 1e-16:
                    stats["faces_skipped"] += 1
                    continue
                faces.append((pts, spec_for(tri.material_index)))
    finally:
        obj.to_mesh_clear()
    return faces


def _is_identity_matrix(m4, tol=1e-5) -> bool:
    ident = mathutils.Matrix.Identity(4)
    for r in range(4):
        for c in range(4):
            limit = tol * 0.0254 if c == 3 and r < 3 else tol
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
    if all(abs(a - b) <= 1e-9 for a, b in zip(m3, ident3)):
        m3 = None
    tr = (
        float(m4[0][3] * METRES_TO_INCHES),
        float(m4[1][3] * METRES_TO_INCHES),
        float(m4[2][3] * METRES_TO_INCHES),
    )
    return m3, tr


def _instance_name_for(empty_obj, def_base_name: str) -> str:
    """Returns the SketchUp instance name for ``empty_obj``, or ``""`` when
    the Empty only carries its definition's name (with optional ``.001``
    Blender uniqueness suffix) - SketchUp stores unnamed instances as ``""``
    rather than baking the definition's name into the instance."""
    base = _strip_num_suffix(empty_obj.name)
    if not base or base == def_base_name:
        return ""
    return base


def _faces_signature(faces):
    return tuple(
        (
            tuple((round(x, 5), round(y, 5), round(z, 5)) for x, y, z in pts),
            spec[0] if spec is not None else None,
        )
        for pts, spec in faces
    )


def _children_signature(children):
    return tuple(
        (
            child_def_key,
            inst_name or "",
            tuple(round(v, 6) for v in (m3 or (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))),
            tuple(round(v, 5) for v in tr),
            layer_name or "",
        )
        for child_def_key, inst_name, m3, tr, layer_name in children
    )


def _is_collection_empty(obj) -> bool:
    return (
        obj.type == "EMPTY"
        and obj.instance_type == "COLLECTION"
        and obj.instance_collection is not None
    )


def export_skp(filepath, context):
    """Exports selected (or all visible) objects to a new .skp file at
    ``filepath``, preserving collection instances and hierarchy groups as
    SketchUp component definitions and instances (#28) while exporting plain
    mesh objects as loose root geometry."""
    from openskp import create

    imported_roots, source_collections, imported_tag_collections = _collect_import_collections()
    scene_collection = context.scene.collection

    selected = [o for o in context.selected_objects if o.type in ("MESH", "EMPTY")]
    explicit_selection = bool(selected)
    if selected:
        raw_candidates = selected
    else:
        raw_candidates = [
            o for o in context.scene.objects
            if o.type in ("MESH", "EMPTY") and o.visible_get()
        ]

    # Expand hierarchy descendants under any included PLAIN_AXES group Empty so
    # selecting a parent component Empty exports its whole subtree, while
    # ensuring children whose parent Empty is also included are never exported a
    # second time at the root.
    expanded = list(raw_candidates)
    seen_names = {o.name for o in expanded}

    def add_descendants(parent_obj):
        for ch in parent_obj.children:
            if ch.type not in ("MESH", "EMPTY"):
                continue
            if not explicit_selection and not ch.visible_get():
                continue
            if ch.name not in seen_names:
                seen_names.add(ch.name)
                expanded.append(ch)
            if ch.type == "EMPTY" and not _is_collection_empty(ch):
                add_descendants(ch)

    for obj in list(raw_candidates):
        if obj.type == "EMPTY" and not _is_collection_empty(obj):
            add_descendants(obj)

    # Identify all Collections referenced via collection-instance Empties
    # (including nested collection-instance Empties inside those Collections).
    instanced_colls = set()

    def mark_instanced_collection(coll):
        if coll is None or coll.name in instanced_colls:
            return
        instanced_colls.add(coll.name)
        for o in coll.objects:
            if _is_collection_empty(o):
                mark_instanced_collection(o.instance_collection)

    for obj in expanded:
        if _is_collection_empty(obj):
            mark_instanced_collection(obj.instance_collection)

    has_any_instances = bool(instanced_colls)

    def is_definition_master_mesh(obj) -> bool:
        if obj.type != "MESH" or not has_any_instances:
            return False
        if not obj.users_collection:
            return False
        return all(
            c.name in source_collections or c.name in instanced_colls
            for c in obj.users_collection
        )

    active_objs = [o for o in expanded if not is_definition_master_mesh(o)]
    active_set = {o.name for o in active_objs}

    def has_active_empty_ancestor(obj) -> bool:
        p = obj.parent
        while p is not None:
            if p.name in active_set and p.type == "EMPTY" and not _is_collection_empty(p):
                return True
            p = p.parent
        return False

    top_items = [o for o in active_objs if not has_active_empty_ancestor(o)]

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

    def layer_of(obj):
        return _object_layer_name(
            obj,
            scene_collection,
            imported_roots,
            source_collections,
            imported_tag_collections,
            instanced_colls,
        )

    def coll_coord_matrix(coll):
        return _YUP_TO_ZUP if coll.name in source_collections else mathutils.Matrix.Identity(4)

    def empty_skp_world_matrix(empty_obj):
        if _is_collection_empty(empty_obj):
            if empty_obj.instance_collection.name in source_collections:
                return empty_obj.matrix_world @ _ZUP_TO_YUP
            return empty_obj.matrix_world.copy()
        if "sketchup_definition" in empty_obj:
            return empty_obj.matrix_world @ _ZUP_TO_YUP
        return empty_obj.matrix_world.copy()

    # Ordered bottom-up component definitions:
    # def_key -> {"name": str, "faces": [(pts, spec)], "children": [(child_key, inst_name, m3, tr, layer_name)]}
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

    def extract_collection_own_faces(coll):
        if coll.name in coll_faces_cache:
            return coll_faces_cache[coll.name]
        cmat = coll_coord_matrix(coll)
        faces = []
        for o in coll.objects:
            if o.type == "MESH" and o.data is not None and len(o.data.polygons) > 0:
                faces.extend(_extract_mesh_faces(o, cmat @ o.matrix_local, stats))
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
            faces = list(extract_collection_own_faces(coll))
            children = []
            for o in coll.objects:
                if _is_collection_empty(o):
                    child_coll = o.instance_collection
                    child_key = resolve_collection_def(child_coll)
                    if child_key is None:
                        continue
                    child_base = definitions[child_key]["base_name"]
                    child_cmat_inv = coll_coord_matrix(child_coll).inverted()
                    m_local = cmat @ o.matrix_local @ child_cmat_inv
                    if abs(m_local.to_3x3().determinant()) <= 1e-12:
                        continue
                    m3, tr = _matrix_to_skp_args(m_local)
                    children.append((
                        child_key,
                        _instance_name_for(o, child_base),
                        m3,
                        tr,
                        layer_of(o),
                    ))
            if not faces and not children:
                coll_def_cache[coll.name] = None
                return None
            base_name = _strip_num_suffix(coll.name)
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

    group_sig_cache = {}

    def is_own_geometry_child(grp_obj, m_grp_inv, ch, def_base_name: str) -> bool:
        if not _is_collection_empty(ch):
            return False
        coll_base = _strip_num_suffix(ch.instance_collection.name)
        ch_def = ch.get("sketchup_definition")
        if ch_def is not None and ch_def != def_base_name:
            return False
        if ch_def is None and coll_base != def_base_name:
            return False
        if layer_of(ch) != layer_of(grp_obj):
            return False
        m_rel = m_grp_inv @ empty_skp_world_matrix(ch)
        return _is_identity_matrix(m_rel)

    def resolve_group_def(grp_obj):
        m_grp = empty_skp_world_matrix(grp_obj)
        if abs(m_grp.to_3x3().determinant()) <= 1e-12:
            return None
        m_grp_inv = m_grp.inverted()
        def_base_name = grp_obj.get("sketchup_definition") or _strip_num_suffix(grp_obj.name) or "Group"

        direct_children = [ch for ch in grp_obj.children if ch.name in active_set]
        faces = []
        children = []
        claimed_own_mesh = False

        for ch in direct_children:
            if not claimed_own_mesh and is_own_geometry_child(grp_obj, m_grp_inv, ch, def_base_name):
                own_faces = extract_collection_own_faces(ch.instance_collection)
                if own_faces:
                    faces.extend(own_faces)
                    claimed_own_mesh = True
                    continue
                # A curve-only "(lines)" own child has 0 faces - drop it cleanly.
                continue

            if _is_collection_empty(ch):
                child_key = resolve_collection_def(ch.instance_collection)
                if child_key is None:
                    continue
                m_rel = m_grp_inv @ empty_skp_world_matrix(ch)
                if abs(m_rel.to_3x3().determinant()) <= 1e-12:
                    continue
                m3, tr = _matrix_to_skp_args(m_rel)
                child_base = definitions[child_key]["base_name"]
                children.append((
                    child_key,
                    _instance_name_for(ch, child_base),
                    m3,
                    tr,
                    layer_of(ch),
                ))
            elif ch.type == "EMPTY":
                child_key = resolve_group_def(ch)
                if child_key is None:
                    continue
                m_rel = m_grp_inv @ empty_skp_world_matrix(ch)
                if abs(m_rel.to_3x3().determinant()) <= 1e-12:
                    continue
                m3, tr = _matrix_to_skp_args(m_rel)
                child_base = definitions[child_key]["base_name"]
                children.append((
                    child_key,
                    _instance_name_for(ch, child_base),
                    m3,
                    tr,
                    layer_of(ch),
                ))
            elif ch.type == "MESH" and ch.data is not None and len(ch.data.polygons) > 0:
                m_rel = m_grp_inv @ ch.matrix_world
                faces.extend(_extract_mesh_faces(ch, m_rel, stats))

        if not faces and not children:
            return None

        sig = (def_base_name, _faces_signature(faces), _children_signature(children))
        if sig in group_sig_cache:
            return group_sig_cache[sig]

        def_key = ("group", len(definitions))
        definitions[def_key] = {
            "base_name": def_base_name,
            "name": alloc_def_name(def_base_name),
            "faces": faces,
            "children": children,
        }
        def_order.append(def_key)
        group_sig_cache[sig] = def_key
        return def_key

    # Count how many top-level Empties reference each ROOT_MODEL collection so a
    # single unmoved ROOT_MODEL Empty from import_skp exports as loose root faces.
    root_model_top_empties = [
        o for o in top_items
        if _is_collection_empty(o)
        and o.instance_collection.name in source_collections
        and _strip_num_suffix(o.instance_collection.name) == "ROOT_MODEL"
    ]

    root_faces = []       # [(pts, mat_spec, layer_name)]
    root_instances = []   # [(def_key, inst_name, m3, tr, layer_name)]

    for obj in top_items:
        if obj.type == "MESH":
            if obj.data is None or len(obj.data.polygons) == 0:
                stats["objects_skipped"] += 1
                continue
            obj_faces = _extract_mesh_faces(obj, obj.matrix_world, stats)
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
                for pts, spec in own_faces:
                    root_faces.append((pts, spec, None))
                stats["objects_exported"] += 1
                continue

            if abs(m_skp.to_3x3().determinant()) <= 1e-12:
                stats["objects_skipped"] += 1
                continue
            def_key = resolve_collection_def(coll)
            if def_key is None:
                stats["objects_skipped"] += 1
                continue
            m3, tr = _matrix_to_skp_args(m_skp)
            def_base = definitions[def_key]["base_name"]
            root_instances.append((
                def_key,
                _instance_name_for(obj, def_base),
                m3,
                tr,
                obj_layer,
            ))
            stats["objects_exported"] += 1
        elif obj.type == "EMPTY":
            def_key = resolve_group_def(obj)
            if def_key is None:
                if explicit_selection:
                    stats["objects_skipped"] += 1
                continue
            m_skp = empty_skp_world_matrix(obj)
            m3, tr = _matrix_to_skp_args(m_skp)
            def_base = definitions[def_key]["base_name"]
            root_instances.append((
                def_key,
                _instance_name_for(obj, def_base),
                m3,
                tr,
                layer_of(obj),
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
        for _pts, spec in definitions[dk]["faces"]:
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
            for pts, spec in dinfo["faces"]:
                try:
                    comp.add_face(pts, material=material_handle_by_key.get(spec[0]) if spec else None)
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

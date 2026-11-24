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

Editing the source geometry: every unique definition's master mesh/curve
object lives in its own Collection, all gathered under one
"<filename> (source geometry)" Collection which is excluded from the View
Layer (Collection.children + a LayerCollection.exclude = True, found via
_find_layer_collection) - kept out of the viewport/render at its
local-space origin, where it would otherwise show up duplicated on top of
the real, placed instances, while staying visible and selectable in the
Outliner. This is the same pattern Blender's own manual recommends for a
source/library collection feeding instances. To edit one: find it under
that collection in the Outliner, enable its checkbox (un-excluding it),
select the object, Tab into Edit Mode as normal - the edit affects every
placement at once, matching how SketchUp's own components behave.
Selecting a visible placement Empty and pressing Tab does nothing on its
own (an Empty has no mesh data of its own to edit) - this is what a real
user reported as "cannot access Edit Mode" (github#3), traced to there
being no way to even select the right object in the first place (the
source collections were entirely unlinked before this), not a limitation
of Edit Mode itself.

Layers (SketchUp calls them "tags"): every placement Empty is linked into
a Collection named for its own resolved InstancedNode.layer, one per
distinct layer in the file, all direct children of the root Collection
(separate from sources_collection above - a layer is a property of a
PLACEMENT, not of the definition it references, since the same
definition can appear on different layers in different places, so
per-layer grouping only makes sense at the placement level). A layer
hidden in the source file (openskp's InstancedScene.layer_hidden, keyed
by layer name) hides its Collection the same way - Collection.
hide_viewport/hide_render, the ordinary Outliner eye/camera toggles - not
the exclude-from-View-Layer mechanism sources_collection uses, since a
hidden layer is still meant to be a normal, working part of the scene,
just not shown right now.

Materials: each mesh resource's per-triangle material_index (resolved by
openskp's build_instanced_scene() into InstancedScene.gltf_materials, a
glTF-style pbrMetallicRoughness list - the same resolution already used
for openskp's other glTF-shaped exports, just never consumed by this
addon before now) becomes a real Blender Material per unique color/alpha
combination (Principled BSDF Base Color + Alpha, cached in
material_cache so repeated colors across definitions share one Material
datablock), assigned as a mesh material slot with per-polygon
material_index set to match - so a component with several differently
colored faces keeps that per-face variation in Blender, not one flat
color for the whole object. Materials are named after their SketchUp
material when openskp provides the name, their colour is converted from
SketchUp's sRGB to Blender's scene-linear values (srgb_to_linear), a
single-sided material hides its back side (_hide_back_side), and a
colorized texture is re-tinted toward its colour (_add_colorize_tint).

Front and back paint: a SketchUp face whose two sides are painted
differently comes from openskp as two single-sided copies on the same
plane, one wound each way. Drawn as they are, the copies overlap, and which
one shows depends on the view (Solid shading ignores node trees and the
culling flag, so it shows whichever copy wins the depth test, usually the
wrong side's paint on the outside). _build_mesh_object pairs each copy with
its exact twin and keeps ONE polygon per pair, shaded by a material that
mixes the two paints on the Backfacing input
(_get_or_build_pair_material): the front paint where the polygon faces the
camera, the back paint where it faces away. A texture on the back side is
read from a second UV layer (UVMap_back), since the two sides can map the
image differently.

Textured materials: each mesh's own UV coordinates (LocalPrimitive.uvs -
already resolved by openskp per vertex, just never read by this addon
before now) are written into a real UV layer, and a material whose
gltf_materials entry carries a pbrMetallicRoughness.baseColorTexture
gets a real Image Texture node wired into Base Color (and Alpha, for an
image with its own alpha channel), built from InstancedScene.textures'
raw image bytes (PNG/JPEG only - the same restriction glTF itself has)
via a temp file + Image.pack() (embeds the bytes into the .blend itself,
so the temp file isn't left as a dangling external reference) - cached
by texture index in image_cache so the same source image shared by
several materials/objects loads exactly once. A textured material's
resolved solid baseColorFactor (SketchUp's own paint-bucket average
color) is still what Solid-shading and material_cache dedup key off of;
only the shader's actual Base Color input differs when a texture image
is available.
"""
from __future__ import annotations

import importlib
import inspect
import logging
import math
import os
import re
import tempfile
import time

import bpy
import mathutils

_INCH_TO_M = 0.0254

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


# The file extension Blender needs to recognise an image's format when it loads it, by the MIME
# type openskp reports. Anything not listed is loaded as a JPEG (what every non-PNG texture was
# before openskp could carry a scene texture in its own format).
_TEXTURE_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/tiff": ".tif",
    "image/gif": ".gif",
    "image/x-tga": ".tga",
    "image/bmp": ".bmp",
    "image/webp": ".webp",
    "image/vnd.adobe.photoshop": ".psd",
    "image/vnd.ms-dds": ".dds",
    "image/jp2": ".jp2",
}

# Texture formats the scene can carry that Blender cannot use as an image texture. Blender opens a
# GIF as a MOVIE source: the file stays open (it cannot be removed after packing) and the frame is
# not reliably there, so a material with one is built without it, and the import says so.
_UNLOADABLE_TEXTURES = {"image/gif": "GIF"}


def _get_or_build_texture_image(texture_index, textures, image_cache):
    """Returns the Blender Image for textures[texture_index], loading and
    packing it exactly once and caching it for every later material that
    references the same source image (a texture is routinely shared
    across many materials/objects in a real file).

    Blender has no "load an image straight from bytes" API - the bytes
    are written to a real temp file, loaded from there, then Image.pack()
    is called BEFORE the temp file is removed: pack() is what actually
    forces Blender to read the file's bytes into the .blend's own data,
    so the temp file has to still exist at that point (confirmed by
    reading Blender's own docs on Image.pack(), not assumed) - after
    pack() succeeds, the temp file is redundant and safe to delete."""
    if texture_index in image_cache:
        return image_cache[texture_index]
    if textures is None or not (0 <= texture_index < len(textures)):
        return None

    tex = textures[texture_index]
    if tex.mime_type in _UNLOADABLE_TEXTURES:
        return None
    ext = _TEXTURE_EXTENSIONS.get(tex.mime_type, ".jpg")
    fd, path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(tex.data)
        image = bpy.data.images.load(path)
        image.pack()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    # bpy.data.images.load() names the image after the temp file's own
    # basename ("tmp17pryts4.jpg") - rename to the source file's own
    # original filename when openskp recovered one, purely cosmetic
    # (Outliner/Shader Editor readability) but a real improvement over a
    # meaningless temp name.
    source_name = os.path.basename(tex.filename) if tex.filename else ""
    image.name = source_name or f"openskp_texture_{texture_index}"

    image_cache[texture_index] = image
    return image


def _srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def srgb_to_linear(rgb):
    """openskp's baseColorFactor is SketchUp's 0-255 sRGB colour divided by
    255; Blender colour sockets and Material.diffuse_color take scene-linear
    values, so used as-is every colour renders lighter than in SketchUp."""
    return tuple(_srgb_to_linear(c) for c in rgb)


def _get_or_build_material(material_index, gltf_materials, material_cache, textures, image_cache):
    """Returns the Blender Material for gltf_materials[material_index],
    building it exactly once per unique index and caching it for every
    later mesh that references the same one - openskp's own
    get_material_index() (instanced_scene.py) already deduplicates by
    (color, double_sided, texture, transparency) within one file, so the
    index alone is a valid, stable cache key here.

    When the material has a baseColorTexture, a real Image Texture node
    is wired into Base Color (and Alpha, for an image with its own alpha
    channel) - see _get_or_build_texture_image and this module's own
    "Textured materials" docstring section. Otherwise (or if the image
    fails to load) falls back to the resolved solid baseColorFactor,
    same as before textures existed.
    """
    if material_index in material_cache:
        return material_cache[material_index]

    gltf_mat = gltf_materials[material_index] if 0 <= material_index < len(gltf_materials) else {}
    r, g, b, a = _base_color(gltf_mat)

    # openskp 1.6+ carries the SketchUp material's own name; older wheels
    # (and unpainted faces drawn in their layer colour) have none.
    mat = bpy.data.materials.new(name=gltf_mat.get("name") or f"openskp_material_{material_index}")
    mat["openskp_material_index"] = material_index
    mat.use_nodes = True
    lr, lg, lb = srgb_to_linear((r, g, b))
    mat.diffuse_color = (lr, lg, lb, a)  # Solid viewport shading / thumbnails
    if not gltf_mat.get("name") and not _has_texture(gltf_mat):
        mat["openskp_initial_diffuse"] = [lr, lg, lb, a]

    _populate_surface(mat, mat.node_tree.nodes.get("Principled BSDF"), gltf_mat, textures, image_cache)

    if gltf_mat.get("doubleSided"):
        mat.use_backface_culling = False
    else:
        _hide_back_side(mat)

    material_cache[material_index] = mat
    return mat


def _base_color(gltf_mat):
    pbr = gltf_mat.get("pbrMetallicRoughness", {})
    return pbr.get("baseColorFactor", [0.8, 0.8, 0.8, 1.0])


def _has_texture(gltf_mat):
    return gltf_mat.get("pbrMetallicRoughness", {}).get("baseColorTexture") is not None


def _populate_surface(mat, bsdf, gltf_mat, textures, image_cache, uv_map=None):
    """Fills `bsdf` (a Principled BSDF in `mat`'s node tree) from one
    openskp material: its texture, tint and alpha, or its solid colour.
    `uv_map` names a UV layer for the texture to read instead of the active
    one (the back side of a pair, see _get_or_build_pair_material).

    Returns True when the surface takes its transparency from the texture's
    own alpha channel (a fence, a leaf, a cut-out sign): the material is then
    set to the Dithered render method (see _use_dithered_alpha)."""
    r, g, b, a = _base_color(gltf_mat)
    texture_alpha = False
    texture_ref = gltf_mat.get("pbrMetallicRoughness", {}).get("baseColorTexture")
    lr, lg, lb = srgb_to_linear((r, g, b))
    if a < 1.0:
        mat.surface_render_method = "BLENDED"

    image = None
    if texture_ref is not None and bsdf is not None:
        image = _get_or_build_texture_image(texture_ref.get("index"), textures, image_cache)

    if image is not None:
        tex_node = mat.node_tree.nodes.new("ShaderNodeTexImage")
        tex_node.image = image
        tex_node.location = (bsdf.location.x - 300, bsdf.location.y)
        colorize = (gltf_mat.get("extras") or {}).get("sketchup") or {}
        if colorize.get("colorized"):
            _add_colorize_tint(mat, tex_node, bsdf, image, (r, g, b), colorize.get("colorize_type", 0))
        else:
            mat.node_tree.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
        if uv_map is not None:
            uv_node = mat.node_tree.nodes.new("ShaderNodeUVMap")
            uv_node.uv_map = uv_map
            uv_node.location = (tex_node.location.x - 220, tex_node.location.y)
            mat.node_tree.links.new(uv_node.outputs["UV"], tex_node.inputs["Vector"])
        if "Alpha" in bsdf.inputs and image.depth in (32, 128):  # has its own alpha channel
            mat.node_tree.links.new(tex_node.outputs["Alpha"], bsdf.inputs["Alpha"])
            texture_alpha = True
            _use_dithered_alpha(mat)
        elif "Alpha" in bsdf.inputs:
            # A translucent textured material (SketchUp's library glass)
            # with an opaque image still has the material's own opacity.
            bsdf.inputs["Alpha"].default_value = a
    elif bsdf is not None:
        # Blender 4.x renamed the alpha-adjacent socket; "Base Color" is
        # stable across the versions this addon targets (4.2+).
        bsdf.inputs["Base Color"].default_value = (lr, lg, lb, 1.0)
        if "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = a
    return texture_alpha


def _use_dithered_alpha(mat):
    """EEVEE's Dithered render method (Material > Settings > Render Method)
    draws a cut-out texture without sorting it against other faces, so the
    holes of a fence or lattice show what is behind them from any angle.
    The Blended method sorts per object, and a mesh of many cut-out faces
    shows the wrong ones in front. Dithered is Blender's own default for a
    new material; it is set explicitly so a surface whose opacity is a flat
    value (SketchUp's glass, below) can use Blended, which is smooth where
    Dithered is grainy, without changing this one. Cycles ignores the
    setting and always renders alpha correctly."""
    mat.surface_render_method = "DITHERED"


_BACK_UV_LAYER = "UVMap_back"


def _is_painted(gltf_mat):
    """Whether ``gltf_mat`` comes from a SketchUp material (which carries its
    name or a texture) rather than an unpainted side's fallback colour."""
    return bool(gltf_mat.get("name")) or _has_texture(gltf_mat)


def _solid_display_material(front, back):
    """Which side's colour Solid viewport shading (``Material.diffuse_color``)
    shows for a front/back pair: the painted side when the other side is
    unpainted (such as a box wall painted only on the inside, #16), and the
    polygon's front side when both sides are painted or neither is."""
    if _is_painted(back) and not _is_painted(front):
        return back
    return front


def _get_or_build_pair_material(front_index, back_index, gltf_materials, material_cache, textures, image_cache):
    """The Blender Material for a polygon with one paint on its front and
    another on its back: both surfaces live in one node tree and a Mix
    Shader driven by Geometry > Backfacing picks between them, so the
    polygon is visible, and correct, from both sides in every render engine
    (and Solid shading shows the painted side when the other side is just
    the default, or the front paint when both sides are painted, with no
    overlapping copy to fight it). The back surface reads its texture from
    the UVMap_back layer. Cached per (front, back) pair of openskp material
    indices."""
    key = (front_index, back_index)
    if key in material_cache:
        return material_cache[key]

    def entry(index):
        return gltf_materials[index] if 0 <= index < len(gltf_materials) else {}

    front, back = entry(front_index), entry(back_index)

    def label(gltf_mat, index):
        return gltf_mat.get("name") or f"openskp_material_{index}"

    mat = bpy.data.materials.new(name=f"{label(front, front_index)} | {label(back, back_index)}")
    mat["openskp_material_index"] = front_index
    mat["openskp_back_material_index"] = back_index
    mat.use_nodes = True
    r, g, b, a = _base_color(_solid_display_material(front, back))
    lr, lg, lb = srgb_to_linear((r, g, b))
    mat.diffuse_color = (lr, lg, lb, a)
    if not _is_painted(front) and not _is_painted(back):
        mat["openskp_initial_diffuse"] = [lr, lg, lb, a]

    nt = mat.node_tree
    front_bsdf = nt.nodes.get("Principled BSDF")
    back_bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    back_bsdf.location = (front_bsdf.location.x, front_bsdf.location.y - 700)
    front_alpha = _populate_surface(mat, front_bsdf, front, textures, image_cache)
    back_alpha = _populate_surface(
        mat, back_bsdf, back, textures, image_cache, uv_map=_BACK_UV_LAYER if _has_texture(back) else None
    )
    if front_alpha or back_alpha:
        _use_dithered_alpha(mat)  # one render method serves both paints: cut-outs win over flat glass

    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    geometry = nt.nodes.new("ShaderNodeNewGeometry")
    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.label = "Front / back paint"
    geometry.location = (out.location.x - 200, out.location.y + 250)
    mix.location = (out.location.x, out.location.y + 150)
    out.location.x += 200
    nt.links.new(geometry.outputs["Backfacing"], mix.inputs["Fac"])
    nt.links.new(front_bsdf.outputs["BSDF"], mix.inputs[1])
    nt.links.new(back_bsdf.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    mat.use_backface_culling = False  # both sides are drawn, each with its own paint

    material_cache[key] = mat
    return mat


def _hide_back_side(mat):
    """A single-sided openskp material is one side of a face whose two
    sides are painted differently: openskp emits that face twice, on the
    same plane, once per side with opposite winding. Unless each copy is
    invisible from behind, both render on top of each other and the wrong
    side's paint shows through (a record wall's grey front covering its
    painted back, seen on a real file). Backface culling covers the
    viewport and EEVEE; the Backfacing mix covers Cycles, which ignores
    the culling flag."""
    mat.use_backface_culling = True
    nt = mat.node_tree
    out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
    if out is None or not out.inputs["Surface"].links:
        return
    surface = out.inputs["Surface"].links[0].from_socket
    geometry = nt.nodes.new("ShaderNodeNewGeometry")
    transparent = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.label = "Hide back side"
    geometry.location = (out.location.x - 200, out.location.y + 250)
    transparent.location = (out.location.x - 200, out.location.y - 150)
    mix.location = (out.location.x, out.location.y + 150)
    out.location.x += 200
    nt.links.new(geometry.outputs["Backfacing"], mix.inputs["Fac"])
    nt.links.new(surface, mix.inputs[1])
    nt.links.new(transparent.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])


def _image_average_rgb(image):
    """Mean linear RGB of an image's pixels, or None when it can't be read.
    ``Image.pixels`` holds a byte image's values as stored (sRGB), so the
    mean is converted the same way the Image Texture node converts them."""
    try:
        import numpy as np

        w, h = image.size
        if not w or not h or image.channels < 3:
            return None
        pixels = np.empty(w * h * image.channels, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        rgb = pixels.reshape(-1, image.channels)[:, :3]
        mean = (float(c) for c in rgb.mean(axis=0))
        return tuple(mean) if image.is_float else tuple(_srgb_to_linear(c) for c in mean)
    except Exception:  # noqa: BLE001 - tinting is cosmetic, never fail the import over it
        return None


def _input(node, name, socket_type):
    """``ShaderNodeMix`` has one "A"/"B"/"Result" socket per data type."""
    return next(s for s in node.inputs if s.name == name and s.type == socket_type)


def _output(node, name, socket_type):
    return next(s for s in node.outputs if s.name == name and s.type == socket_type)


def _add_colorize_tint(mat, tex_node, bsdf, image, color, colorize_type):
    """Re-tints a colorized material's shared image toward its own colour,
    as SketchUp does for its "[Name]1" copies: openskp gives the untinted
    source image plus the target colour (baseColorFactor, sRGB).

    Approximated by scaling the image so its average lands on the target
    colour - colorize_type 0 (shift) scales the colour image, 1 (tint)
    greyscales it first. Without this, a darkened wood veneer imports with
    the original, much lighter image."""
    nt = mat.node_tree
    target = srgb_to_linear(color)
    average = _image_average_rgb(image)
    source = tex_node.outputs["Color"]
    if colorize_type == 1:
        bw = nt.nodes.new("ShaderNodeRGBToBW")
        bw.location = (tex_node.location.x + 200, tex_node.location.y - 150)
        nt.links.new(source, bw.inputs["Color"])
        source = bw.outputs["Val"]
        if average is not None:
            luminance = 0.2126 * average[0] + 0.7152 * average[1] + 0.0722 * average[2]
            average = (luminance,) * 3
    scale = tuple(t / max(avg, 1e-4) for t, avg in zip(target, average)) if average else target

    tint = nt.nodes.new("ShaderNodeMix")
    tint.data_type = "RGBA"
    tint.blend_type = "MULTIPLY"
    tint.clamp_result = True
    tint.label = "SketchUp colorize"
    tint.location = (tex_node.location.x + 400, tex_node.location.y)
    _input(tint, "Factor", "VALUE").default_value = 1.0
    _input(tint, "B", "RGBA").default_value = (*scale, 1.0)
    nt.links.new(source, _input(tint, "A", "RGBA"))
    nt.links.new(_output(tint, "Result", "RGBA"), bsdf.inputs["Base Color"])
    tex_node.location.x -= 200


def _triangle_key(verts, tri):
    """(position set, winding) of a triangle, or None for a degenerate one.
    Two triangles with the same set and opposite winding are twins: the same
    face seen from its two sides. Positions compare exactly: openskp emits
    both copies from one triangulation, so their vertices are identical."""
    corners = [verts[v] for v in tri[:3]]
    if len(set(corners)) < 3:
        return None
    start = corners.index(min(corners))
    ordered = corners[start:] + corners[:start]
    return frozenset(corners), ordered[1] < ordered[2]


def _find_twin_triangles(verts, tris):
    """{triangle index: its twin's index} for every pair of triangles that
    are the same face with opposite winding and different materials. The
    materials must differ: a face painted the same on both sides is already
    one double-sided copy, and two same-paint triangles stacked in one place
    are two faces of the model, not two sides of one."""
    if len({tri[3] for tri in tris}) < 2:
        return {}  # twins need two different materials: nothing to search for
    earlier = {}  # shape -> [(triangle index, winding)] not yet paired
    twin_of = {}
    for t, tri in enumerate(tris):
        key = _triangle_key(verts, tri)
        if key is None:
            continue
        shape, winding = key
        candidates = earlier.setdefault(shape, [])
        for position, (other, other_winding) in enumerate(candidates):
            if other_winding != winding and tris[other][3] != tri[3]:
                twin_of[other] = t
                twin_of[t] = other
                del candidates[position]
                break
        else:
            candidates.append((t, winding))
    return twin_of


def _choose_front_triangles(verts, tris, twin_of):
    """{kept triangle: dropped twin}. Either copy can carry the pair in EEVEE
    and Cycles, because the Backfacing mix picks the paint by which side is
    seen; the choice decides the polygon normals, which of the two paints uses
    the main UV layer, and which paint Solid shading shows when both sides are
    painted (or neither is).

    Connected pairs are turned the same way round, so a surface keeps one
    orientation across its edges, and then each connected surface as a whole
    faces outward. Deciding triangle by triangle ("facing away from the mesh's
    centre") is only right for convex shapes: on a torus it turned the faces
    of the inner ring inside out (#33). A surface with no outside, such as a
    single flat face, keeps the first copy."""
    if not twin_of:
        return {}
    pairs = sorted((t, o) for t, o in twin_of.items() if t < o)

    # Neighbours: pairs sharing an edge that belongs to exactly two pairs. An
    # edge where more of them meet doesn't say which two form one surface.
    by_edge = {}
    for n, (first, _) in enumerate(pairs):
        a, b, c = (verts[v] for v in tris[first][:3])
        for start, end in ((a, b), (b, c), (c, a)):
            forward = start < end
            by_edge.setdefault((start, end) if forward else (end, start), []).append((n, forward))
    neighbours = [[] for _ in pairs]
    for users in by_edge.values():
        if len(users) == 2 and users[0][0] != users[1][0]:
            (n, n_forward), (m, m_forward) = users
            # Two triangles of one consistently turned surface run along
            # their shared edge in opposite directions.
            flip = n_forward == m_forward
            neighbours[n].append((m, flip))
            neighbours[m].append((n, flip))

    centre = [sum(v[axis] for v in verts) / len(verts) for axis in range(3)]

    def outward(t):
        """Six times the signed volume between the triangle and the centre:
        positive when it faces away from it. Added up over a closed surface
        this is the volume inside, whatever the shape."""
        a, b, c = ([v[axis] - centre[axis] for axis in range(3)] for v in (verts[k] for k in tris[t][:3]))
        return (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            + a[1] * (b[2] * c[0] - b[0] * c[2])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )

    chosen = {}
    flipped = [None] * len(pairs)
    for seed in range(len(pairs)):
        if flipped[seed] is not None:
            continue
        flipped[seed] = False
        surface = [seed]
        for n in surface:  # grows while it is walked
            for m, flip in neighbours[n]:
                if flipped[m] is None:
                    flipped[m] = flipped[n] != flip
                    surface.append(m)
        volumes = [outward(pairs[n][0]) * (-1 if flipped[n] else 1) for n in surface]
        total = sum(volumes)
        # A flat surface has no volume against the centre, only rounding noise,
        # so the sum is compared with the size of what went into it.
        size = sum(math.prod(math.dist(verts[v], centre) for v in tris[pairs[n][0]][:3]) for n in surface)
        turn = total < -1e-6 * size
        for n in surface:
            first, second = pairs[n]
            keep, drop = (second, first) if flipped[n] != turn else (first, second)
            chosen[keep] = drop
    return chosen


def _build_mesh_object(name, resource, gltf_materials, material_cache, textures, image_cache):
    """Builds one Mesh Object (local space, at the origin) from an
    InstancedMeshResource's primitives - already-triangulated, so this is
    a direct from_pydata() with no triangulation of its own to do.

    Each primitive is already grouped by a single resolved material (see
    openskp.instanced_scene.mesh_resource_for's own docstring) - its
    material_index is used both to pick/build the right Blender Material
    and to set each of its triangles' own polygon.material_index, so the
    mesh's material slots and per-face assignment match the source file's
    own per-face paint, not just one flat color for the whole object.

    A face painted differently on its two sides arrives as two single-sided
    triangle sets with identical vertices and opposite winding (see this
    module's "Front and back paint" section). Each such pair becomes one
    polygon with a front/back material (_get_or_build_pair_material), and
    the twin is dropped, so nothing overlaps.
    """
    verts = []
    vert_uvs = []
    tris = []  # (a, b, c, material_index) per source triangle, in order
    for prim in resource.primitives:
        base = len(verts)
        pos = prim.positions
        uvs = prim.uvs
        for i in range(0, len(pos), 3):
            verts.append((pos[i], pos[i + 1], pos[i + 2]))
        for i in range(0, len(uvs), 2):
            vert_uvs.append((uvs[i], uvs[i + 1]))
        idx = prim.indices
        for i in range(len(idx) // 3):
            tris.append((base + idx[i * 3], base + idx[i * 3 + 1], base + idx[i * 3 + 2], prim.material_index))

    has_uvs = len(vert_uvs) == len(verts) and bool(vert_uvs)
    twin_of = _find_twin_triangles(verts, tris)
    base_of_pair = _choose_front_triangles(verts, tris, twin_of)  # pair's kept triangle -> dropped twin

    slot_by_key = {}
    mesh_materials = []

    def slot_for(front_index, back_index):
        key = front_index if back_index is None else (front_index, back_index)
        if key not in slot_by_key:
            slot_by_key[key] = len(mesh_materials)
            mesh_materials.append(
                _get_or_build_material(front_index, gltf_materials, material_cache, textures, image_cache)
                if back_index is None
                else _get_or_build_pair_material(
                    front_index, back_index, gltf_materials, material_cache, textures, image_cache
                )
            )
        return slot_by_key[key]

    dropped = set(base_of_pair.values())
    kept = []  # (a, b, c, slot, back_uvs or None)
    needs_back_uvs = False
    for t, (a, b, c, material_index) in enumerate(tris):
        if t in dropped:
            continue
        twin = base_of_pair.get(t)
        if twin is None:
            kept.append((a, b, c, slot_for(material_index, None), None))
            continue
        back_index = tris[twin][3]
        back_uvs = None
        if has_uvs and _has_texture(
            gltf_materials[back_index] if 0 <= back_index < len(gltf_materials) else {}
        ):
            needs_back_uvs = True
            by_position = {verts[v]: vert_uvs[v] for v in tris[twin][:3]}
            back_uvs = tuple(by_position[verts[v]] for v in (a, b, c))
        kept.append((a, b, c, slot_for(material_index, back_index), back_uvs))

    # openskp gives every face its own copy of the vertices it uses, so a
    # shared corner arrives as several vertices at one position. Weld them
    # (exact positions: they come from the same SketchUp vertex), or the mesh
    # is several times larger than it should be and every face is detached
    # from its neighbours. A triangle with two corners at one point has no
    # area and would become an invalid polygon once welded: leave it out.
    kept = [tri for tri in kept if len({verts[v] for v in tri[:3]}) == 3]
    welded = {}  # position -> index in out_verts
    out_verts = []
    out_tris = []
    for a, b, c, _, _ in kept:
        corners = []
        for v in (a, b, c):
            index = welded.get(verts[v])
            if index is None:
                index = welded[verts[v]] = len(out_verts)
                out_verts.append(verts[v])
            corners.append(index)
        out_tris.append(tuple(corners))

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(out_verts, [], out_tris)
    for mat in mesh_materials:
        mesh.materials.append(mat)
    for poly, tri in zip(mesh.polygons, kept):
        poly.material_index = tri[3]

    # UVs are per-LOOP (face-corner) in Blender, and a welded vertex is shared
    # by faces that map it differently (a texture seam), so each corner takes
    # the UV of the source vertex it came from, not one UV per welded vertex.
    # Every polygon is a triangle, so polygon i owns loops 3i .. 3i+2, in the
    # corner order of kept[i].
    if has_uvs:
        front_uvs = [uv for tri in kept for v in tri[:3] for uv in vert_uvs[v]]
        uv_layer = mesh.uv_layers.new(name="UVMap")
        uv_layer.data.foreach_set("uv", front_uvs)
        if needs_back_uvs:
            # The back paint's own mapping, per corner; the other polygons just
            # repeat the front mapping so the layer is complete.
            back_uvs = list(front_uvs)
            for i, tri in enumerate(kept):
                if tri[4] is not None:
                    back_uvs[6 * i:6 * i + 6] = [value for uv in tri[4] for value in uv]
            back_layer = mesh.uv_layers.new(name=_BACK_UV_LAYER)
            back_layer.data.foreach_set("uv", back_uvs)

    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    return obj


def _get_or_build_layer_collection(layer_name, layer_collection_cache, root_collection):
    """Returns the Collection that placements on `layer_name` (SketchUp's
    own term is "tag") get linked into, building it exactly once per
    distinct layer and linking it directly under root_collection -
    separate from sources_collection, since a layer is a property of a
    PLACEMENT (InstancedNode.layer), not of a definition: the same
    definition can appear on different layers in different places, so
    grouping by layer only makes sense at the placement level."""
    name = layer_name or "Layer0"
    if name in layer_collection_cache:
        return layer_collection_cache[name]
    coll = bpy.data.collections.new(name)
    root_collection.children.link(coll)
    layer_collection_cache[name] = coll
    return coll


def _get_or_build_collection(
    resource_id, resources_by_id, collection_cache, sources_collection, gltf_materials, material_cache,
    textures, image_cache, stats,
):
    """Returns the Collection holding resource_id's mesh object, building
    it exactly once per unique id and caching it for every later
    placement - the whole point of using build_instanced_scene() over
    build_scene().

    Linked as a child of sources_collection - which import_skp() excludes
    from the View Layer - rather than left unlinked. An unlinked
    Collection is invisible to Blender's own Outliner in its default View
    Layer display mode, so the master mesh object (the only thing with
    real, editable geometry - every visible placement is a Collection-
    Instance Empty, which has none) was undiscoverable through the normal
    workflow entirely: found reported as "selecting an object doesn't
    allow Edit Mode" (github#3), traced to there being no way to even
    select the right object in the first place, not a limitation of Edit
    Mode itself. Excluding rather than fully linking keeps the source
    geometry out of the viewport/render at its local-space origin (where
    it would otherwise show up duplicated), while keeping it selectable
    in the Outliner - the same pattern Blender's own manual recommends
    for library/source collections feeding instances."""
    if resource_id in collection_cache:
        return collection_cache[resource_id]

    resource = resources_by_id[resource_id]
    coll = bpy.data.collections.new(resource.definition_name or resource_id)
    obj = _build_mesh_object(
        resource.definition_name or resource_id, resource, gltf_materials, material_cache, textures, image_cache
    )
    coll.objects.link(obj)
    sources_collection.children.link(coll)
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


def _get_or_build_curve_collection(
    resource_id, curve_resources_by_id, curve_collection_cache, sources_collection, stats
):
    """Returns the Collection holding resource_id's loose-edge object,
    building it exactly once per unique id - same caching strategy as
    _get_or_build_collection (including linking into sources_collection
    for the same Outliner-discoverability reason), independent cache/id
    space since a node can carry a mesh_resource_id, a curve_resource_id,
    or both (a definition can have faces AND loose edges at once)."""
    if resource_id in curve_collection_cache:
        return curve_collection_cache[resource_id]

    resource = curve_resources_by_id[resource_id]
    name = (resource.definition_name or resource_id) + " (lines)"
    coll = bpy.data.collections.new(name)
    obj = _build_curve_object(name, resource)
    coll.objects.link(obj)
    sources_collection.children.link(coll)
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
    layer_collection_cache,
    root_collection,
    sources_collection,
    gltf_materials,
    material_cache,
    textures,
    image_cache,
    stats,
    keep_hierarchy=False,
    parent_object=None,
    is_root=False,
):
    """Walks one InstancedNode, placing an Empty (collection-instance) for
    its own mesh_resource_id and/or curve_resource_id (a definition can
    have both - faces AND loose edges at once) at the composed world
    matrix, then recurses into children with that same world matrix as
    their new parent matrix - matrices are relative-to-parent in openskp's
    own model, composed here into world space.

    With ``keep_hierarchy`` off, every placement is a top-level object (a
    flat result). With it on, SketchUp's own group/component nesting is
    kept as Blender parenting: a group with children becomes an Empty named
    after it, its own geometry and its child groups parented under it, so
    the Outliner shows SketchUp's tree and moving, selecting (Select
    Hierarchy) or hiding (Shift-click the eye) a whole sofa or person is one
    step, not one per part. A group without children doesn't get an extra
    Empty: its geometry's placement is parented directly.

    Each placement Empty is linked into a Collection for its OWN
    node.layer (SketchUp's "tag"), not a single fixed target - a layer is
    a property of the placement, not of the definition it references (the
    same definition can appear on different layers in different places),
    so grouping by layer only makes sense here, at the placement level."""
    local_matrix = _gltf_matrix_to_blender(node.matrix)
    world_matrix = parent_matrix @ local_matrix
    world_bl = _YUP_TO_ZUP @ world_matrix
    sketchup_layer_coll = _get_or_build_layer_collection(node.layer, layer_collection_cache, root_collection)

    def place(obj, parent):
        """``parent`` is ``(object, its world matrix)`` or None."""
        sketchup_layer_coll.objects.link(obj)
        if keep_hierarchy and parent is not None:
            # matrix_basis is stored as loc/rot/scale and can't hold shear (a
            # rotated child inside a non-uniformly scaled component); whatever
            # it drops goes into the parent inverse so the world matrix is exact.
            local = parent[1].inverted_safe() @ world_bl
            obj.parent = parent[0]
            obj.matrix_basis = local
            obj.matrix_parent_inverse = local @ obj.matrix_basis.inverted_safe()
        else:
            obj.matrix_world = world_bl
        if keep_hierarchy:
            if node.definition_name:
                obj["sketchup_definition"] = node.definition_name
            if getattr(node, "guid", ""):
                obj["sketchup_guid"] = node.guid

    own_parent = parent_object
    if keep_hierarchy and node.children and not is_root:
        group = bpy.data.objects.new(node.name or node.definition_name or "Group", None)
        group.empty_display_type = "PLAIN_AXES"
        group.empty_display_size = 0.1
        place(group, parent_object)
        own_parent = (group, world_bl)
        stats["groups"] += 1

    if node.mesh_resource_id is not None:
        coll = _get_or_build_collection(
            node.mesh_resource_id, resources_by_id, collection_cache, sources_collection,
            gltf_materials, material_cache, textures, image_cache, stats
        )
        empty = bpy.data.objects.new(node.name or node.mesh_resource_id, None)
        empty.instance_type = "COLLECTION"
        empty.instance_collection = coll
        place(empty, own_parent)
        stats["placements"] += 1

    if node.curve_resource_id is not None:
        curve_coll = _get_or_build_curve_collection(
            node.curve_resource_id, curve_resources_by_id, curve_collection_cache, sources_collection, stats
        )
        curve_empty = bpy.data.objects.new(
            (node.name or node.curve_resource_id) + " (lines)", None
        )
        curve_empty.instance_type = "COLLECTION"
        curve_empty.instance_collection = curve_coll
        place(curve_empty, own_parent)
        stats["curve_placements"] += 1

    for child in node.children:
        _place_node(
            child,
            world_matrix,
            resources_by_id,
            collection_cache,
            curve_resources_by_id,
            curve_collection_cache,
            layer_collection_cache,
            root_collection,
            sources_collection,
            gltf_materials,
            material_cache,
            textures,
            image_cache,
            stats,
            keep_hierarchy,
            own_parent,
        )


def _find_layer_collection(layer_collection, target_collection):
    """Finds the LayerCollection wrapping target_collection, searching
    from layer_collection downward - needed to set .exclude on a
    just-linked Collection, since that flag lives on the View Layer's own
    parallel LayerCollection tree, not on the Collection datablock itself."""
    if layer_collection.collection == target_collection:
        return layer_collection
    for child in layer_collection.children:
        found = _find_layer_collection(child, target_collection)
        if found is not None:
            return found
    return None


def _build_camera_from_page(page, index):
    """Builds a Blender Camera Object for one SketchUp ``Page`` (saved scene),
    or ``None`` when the scene saves no camera (``eye``/``target`` missing or
    coincident).

    After ``_YUP_TO_ZUP`` converts openskp's glTF Y-up scene nodes to Blender
    Z-up, Blender world space matches SketchUp's right-handed ``+X`` east,
    ``+Y`` north, ``+Z`` up axes scaled by ``0.0254`` m/in. A Blender camera
    looks down local ``-Z`` with local ``+Y`` up and ``+X`` right, and
    ``sensor_fit = 'VERTICAL'`` makes ``angle`` and ``ortho_scale`` match
    SketchUp's vertical ``fov`` (degrees) and ``ortho_height`` (inches).
    """
    if page.eye is None or page.target is None:
        return None
    eye_m = mathutils.Vector(tuple(c * _INCH_TO_M for c in page.eye))
    target_m = mathutils.Vector(tuple(c * _INCH_TO_M for c in page.target))
    view_vec = target_m - eye_m
    dist = view_vec.length
    if dist <= 1e-12:
        return None
    forward = view_vec / dist
    back = -forward

    up_hint = mathutils.Vector(page.up) if page.up is not None else mathutils.Vector((0.0, 0.0, 1.0))
    if up_hint.length <= 1e-12:
        up_hint = mathutils.Vector((0.0, 0.0, 1.0))
    right = forward.cross(up_hint)
    if right.length <= 1e-9:
        fallback = mathutils.Vector((0.0, 1.0, 0.0)) if abs(forward.z) > 0.9 else mathutils.Vector((0.0, 0.0, 1.0))
        right = forward.cross(fallback)
    right.normalize()
    true_up = back.cross(right)
    true_up.normalize()

    cam_name = page.name or f"Scene {index + 1}"
    cam_data = bpy.data.cameras.new(cam_name)
    cam_data.sensor_fit = "VERTICAL"
    cam_data.clip_end = max(1000.0, dist * 10.0)
    if page.parallel:
        cam_data.type = "ORTHO"
        if page.ortho_height > 0.0:
            cam_data.ortho_scale = page.ortho_height * _INCH_TO_M
    else:
        cam_data.type = "PERSP"
        if 0.0 < page.fov < 180.0:
            cam_data.angle = math.radians(page.fov)

    cam_obj = bpy.data.objects.new(cam_name, cam_data)
    cam_obj.matrix_world = mathutils.Matrix(
        (
            (right.x, true_up.x, back.x, eye_m.x),
            (right.y, true_up.y, back.y, eye_m.y),
            (right.z, true_up.z, back.z, eye_m.z),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    return cam_obj


def _import_scene_cameras(pages, root_collection, context):
    """Creates a 'Scenes' collection under ``root_collection`` holding one
    Blender Camera per saved SketchUp scene (``Page``) and makes the first
    scene camera active (#26). Returns the number of cameras created."""
    cameras = []
    for index, page in enumerate(pages or ()):
        cam_obj = _build_camera_from_page(page, index)
        if cam_obj is not None:
            cameras.append(cam_obj)
    if not cameras:
        return 0
    scenes_coll = bpy.data.collections.new("Scenes")
    root_collection.children.link(scenes_coll)
    for cam_obj in cameras:
        scenes_coll.objects.link(cam_obj)
    context.scene.camera = cameras[0]
    return len(cameras)


# Libraries openskp needs that ship as wheels inside this extension. Blender 4.2.0 to
# 4.2.3 can install the wrong build of them (see blender_manifest.toml), and openskp
# only prints a warning and falls back to fan triangulation without mapbox_earcut,
# which silently fills in every hole and cut-out of a face.
_BUNDLED_LIBRARIES = ("mapbox_earcut", "shapely")


class _CapturedWarnings(logging.Handler):
    """Collects the warnings the openskp library logs while it builds a scene
    (a texture format a scene cannot carry, a missing optional dependency, ...),
    so the import can show them to the user instead of leaving them in a log
    nobody reads. Each distinct message is kept once, in the order it came."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        message = record.getMessage()
        if message not in self.messages:
            self.messages.append(message)


class _capture_library_warnings:
    def __enter__(self):
        self.handler = _CapturedWarnings()
        self._logger = logging.getLogger("openskp")
        self._saved_level = self._logger.level
        if self._logger.getEffectiveLevel() > logging.WARNING:
            self._logger.setLevel(logging.WARNING)
        self._logger.addHandler(self.handler)
        return self.handler.messages

    def __exit__(self, *exc_info):
        self._logger.removeHandler(self.handler)
        self._logger.setLevel(self._saved_level)
        return False


def _reconcile_partial_read_warnings(scene, filepath, library_warnings):
    """Returns ``unread_root_entities`` (0 when the file was read in full or
    with an openskp wheel older than openskp#552) and orders any partial-read
    warning first so Blender's one-line status bar report never hides dropped
    geometry behind a texture format note."""
    unread = (
        int(scene.unread_root_entities)
        if hasattr(scene, "unread_root_entities") and scene.unread_root_entities
        else 0
    )
    if not unread:
        for message in library_warnings:
            if "read only in part" in message:
                match = re.search(r"\((\d+)\s+unread\)", message)
                if match:
                    unread = int(match.group(1))
                    break
    if unread > 0 and not any("read only in part" in m for m in library_warnings):
        library_warnings.insert(
            0,
            f"{os.path.basename(filepath)} was read only in part "
            f"({unread} root entities could not be read and were dropped)",
        )
    else:
        library_warnings.sort(key=lambda m: 0 if "read only in part" in m else 1)
    return unread


def check_bundled_libraries():
    """Raises RuntimeError, with something the user can act on, when a library
    bundled with this extension cannot be imported."""
    failed = []
    for name in _BUNDLED_LIBRARIES:
        try:
            importlib.import_module(name)
        except ImportError as exc:
            failed.append(f"{name} ({exc})")
    if failed:
        raise RuntimeError(
            "a library bundled with this add-on could not be loaded: "
            + "; ".join(failed)
            + ". This usually means Blender is older than 4.2.4, which installs the wrong "
            "build of bundled libraries: update Blender, then reinstall the add-on."
        )


def import_skp(filepath, context=None, keep_hierarchy=False):
    """Imports a .skp file into the current (or given) Blender context's
    scene, returning a stats dict. ``keep_hierarchy`` keeps SketchUp's
    group/component nesting as parented Empties (see _place_node)."""
    check_bundled_libraries()
    import openskp

    context = context or bpy.context

    print(f"openskp: parsing {os.path.basename(filepath)}...")
    t0 = time.time()
    with _capture_library_warnings() as library_warnings:
        skp = openskp.SkpFile.open(filepath)
        # A TIFF, TGA or GIF texture is left out of a scene unless asked for (openskp 1.6.0 and newer
        # can keep it in its own format, which Blender reads); older ones have no such option and
        # build the scene as before.
        options = {}
        if "original_textures" in inspect.signature(skp.build_instanced_scene).parameters:
            options["original_textures"] = True
        scene = skp.build_instanced_scene(**options)
        # Saved scenes come from InstancedScene.pages (openskp#593) so the file is
        # never parsed twice; older wheels without the attribute leave it off.
        pages = getattr(scene, "pages", None) or []
    for index, texture in enumerate(getattr(scene, "textures", None) or []):
        if texture.mime_type in _UNLOADABLE_TEXTURES:
            name = os.path.basename(texture.filename) if texture.filename else f"number {index}"
            library_warnings.append(
                f"Texture {name} is a {_UNLOADABLE_TEXTURES[texture.mime_type]}, which Blender cannot use as an "
                "image texture: the materials that use it are built without it"
            )
    unread_root_entities = _reconcile_partial_read_warnings(scene, filepath, library_warnings)
    for message in library_warnings:
        print(f"openskp: warning: {message}")
    print(f"openskp: parsed in {time.time() - t0:.1f}s, building objects...")

    resources_by_id = {r.id: r for r in scene.mesh_resources}
    collection_cache = {}
    curve_resources_by_id = {r.id: r for r in getattr(scene, "curve_resources", None) or []}
    curve_collection_cache = {}
    layer_collection_cache = {}
    layer_hidden = getattr(scene, "layer_hidden", None) or {}
    gltf_materials = getattr(scene, "gltf_materials", None) or []
    material_cache = {}
    textures = getattr(scene, "textures", None) or []
    image_cache = {}
    stats = {
        "unique_meshes": 0,
        "placements": 0,
        "triangles": 0,
        "unique_curve_meshes": 0,
        "curve_placements": 0,
        "curve_runs": 0,
        "groups": 0,
        "unread_root_entities": unread_root_entities,
        # what the library warned about while building the scene (a partial
        # classic read, a texture it could not carry, ...); the operator shows
        # it in Blender's report
        "warnings": library_warnings,
    }

    root_name = os.path.splitext(os.path.basename(filepath))[0]
    root_collection = bpy.data.collections.new(root_name)
    context.scene.collection.children.link(root_collection)

    # Every unique definition's master mesh/curve object lives in its own
    # Collection here, not directly in the viewport - each is linked as a
    # child of THIS collection, which is then excluded from the View
    # Layer (see _find_layer_collection below): kept out of the
    # viewport/render at its local-space origin (where it would otherwise
    # show up duplicated alongside the real, placed instances), while
    # staying visible and selectable in the Outliner - the standard
    # Blender pattern for a source/library collection feeding instances.
    sources_collection = bpy.data.collections.new(f"{root_name} (source geometry)")
    root_collection.children.link(sources_collection)

    t0 = time.time()
    _place_node(
        scene.scene_hierarchy,
        mathutils.Matrix.Identity(4),
        resources_by_id,
        collection_cache,
        curve_resources_by_id,
        curve_collection_cache,
        layer_collection_cache,
        root_collection,
        sources_collection,
        gltf_materials,
        material_cache,
        textures,
        image_cache,
        stats,
        keep_hierarchy,
        is_root=True,
    )
    stats["textures_loaded"] = len(image_cache)
    stats["scenes_imported"] = _import_scene_cameras(pages, root_collection, context)
    print(
        f"openskp: {stats['unique_meshes']} unique meshes "
        f"({stats['triangles']} triangles), {stats['placements']} instances placed; "
        f"{stats['unique_curve_meshes']} unique loose-edge groups "
        f"({stats['curve_runs']} runs), {stats['curve_placements']} placed; "
        f"{len(layer_collection_cache)} layers, {stats['textures_loaded']} textures, "
        f"{stats['scenes_imported']} scene cameras "
        f"in {time.time() - t0:.1f}s"
    )

    # A layer hidden in the source file (SketchUp's own Tags panel
    # visibility) hides its whole Collection here too - Collection-level
    # hide_viewport/hide_render (the same "eye"/camera icons a user can
    # toggle in the Outliner), not the exclude-from-View-Layer mechanism
    # sources_collection uses below: a hidden layer is still meant to be a
    # normal, working part of the scene (just not shown right now), unlike
    # sources_collection's deliberately-excluded master geometry.
    stats["unique_layers"] = len(layer_collection_cache)
    stats["hidden_layers"] = 0
    for name, coll in layer_collection_cache.items():
        if layer_hidden.get(name, False):
            coll.hide_viewport = True
            coll.hide_render = True
            stats["hidden_layers"] += 1

    layer_coll = _find_layer_collection(context.view_layer.layer_collection, sources_collection)
    if layer_coll is not None:
        layer_coll.exclude = True

    return stats

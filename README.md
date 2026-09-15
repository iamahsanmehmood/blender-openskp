# OpenSKP Import/Export for Blender

[![License: GPL v3+](https://img.shields.io/badge/License-GPLv3+-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.html)
[![Latest release](https://img.shields.io/github/v/release/iamahsanmehmood/blender-openskp?label=release)](https://github.com/iamahsanmehmood/blender-openskp/releases/latest)
[![Downloads](https://img.shields.io/github/downloads/iamahsanmehmood/blender-openskp/total?label=downloads)](https://github.com/iamahsanmehmood/blender-openskp/releases)
[![GitHub Stars](https://img.shields.io/github/stars/iamahsanmehmood/blender-openskp?style=social)](https://github.com/iamahsanmehmood/blender-openskp)

Native SketchUp (`.skp`) import **and export** for Blender — built on
[OpenSKP](https://github.com/iamahsanmehmood/openskp), an MIT-licensed,
from-scratch `.skp` reader/writer. No Trimble SDK, no SketchUp
installation — pure Python, vendored directly into this addon. Built as a
standalone [Extension](https://docs.blender.org/manual/en/latest/advanced/extensions/getting_started.html)
(Blender 4.2+), independent of any one BIM add-on's own codebase — see
[Why standalone](#why-standalone) below.

This addon's own integration code is GPL-3.0-or-later (required for
every add-on on the [Blender Extensions Platform](https://extensions.blender.org/)
- confirmed directly against a real upload, not assumed from the docs
alone: MIT is rejected outright, no permissive-license option exists for
an add-on using the `bpy` API). The vendored OpenSKP core underneath
stays MIT - a permissive license can always be incorporated into a GPL
project, just not the other way around, so this doesn't affect OpenSKP
itself or the [FreeCAD addon](https://github.com/iamahsanmehmood/freecad-openskp),
which both remain MIT.

## What it does

**Import** parses a `.skp` file via OpenSKP's `build_instanced_scene()` —
which keeps each unique component/group *definition*'s geometry as one
local-space mesh, separate from the list of everywhere it's placed —
rather than baking every placement into its own copy of the vertex data.
That maps directly onto Blender's own collection-instancing model: one
Collection per unique definition (built exactly once), referenced by an
Empty (`instance_type='COLLECTION'`) per placement. A component placed
1,000 times costs one mesh object, not 1,000, and editing that one mesh
updates every placement at once — matching how SketchUp's own components
behave, and a real architectural improvement over a flat triangle-soup
import.

**Export** (File → Export → SketchUp) flattens the selected mesh objects
(or every visible mesh object, if nothing is selected) into a new `.skp`
file — global (world) transform resolved, so parented/linked objects land
in the right position. Exports Blender's own triangulated view of each
mesh (`calc_loop_triangles`, guaranteed planar per-triangle) rather than
raw n-gons, since nothing in Blender enforces that an edited mesh's faces
stay planar. Round-trip-verified: exported coordinates match the original
geometry exactly (see `tests/test_export.py`).

**Layer export**: each exported object's own Collection becomes its
SketchUp layer/tag — the first Collection it's linked into other than the
scene's own default ones (Blender has no native "tag" concept; a named
Collection is its closest equivalent, and the same convention glTF/FBX/USD
exporters already use). An object left in the default, unrenamed
`Collection` exports to SketchUp's own default layer, same as before this
existed — organizing objects into named Collections is what opts them
into a real tag on export.

**Material export**: each polygon's own material slot
(`poly.material_index` into `obj.data.materials`) becomes a real SketchUp
material, read from the Material's `diffuse_color` (works whether or not
the material has a node tree, unlike reading the Principled BSDF's Base
Color directly). One SketchUp material per unique Blender `Material`
datablock — the same material reused across several objects exports
once, not once per object. A polygon with no material slot exports
unpainted, same as before this existed. Solid colors only, matching
import's own scope — no texture export.

**Materials** (import only, for now) carry over too: each face's resolved
color and opacity — SketchUp's per-face material, or its layer/definition
default when a face has none set directly — becomes a real Blender
Material (Principled BSDF, transparency enabled when the source material
has alpha < 1), assigned per-polygon on the mesh so a multi-colored
component keeps its per-face colors, not one flat color for the whole
object. Reuses OpenSKP's own glTF-style material resolution
(`InstancedScene.gltf_materials` / `LocalPrimitive.material_index`) rather
than reimplementing SketchUp's material-inheritance rules here.

**Textured materials** (import only) carry over too: each mesh's own UV
coordinates (already resolved by openskp per vertex, previously unread by
this addon) land in a real UV layer, and a material with an image texture
gets a real Image Texture node wired into Base Color (and Alpha, for an
image with its own alpha channel) — built from the source file's own
embedded image bytes, packed directly into the `.blend` so nothing
depends on an external file path. The same source image shared by
several materials loads once, not once per material. Export doesn't
write texture images back out yet — a textured material exports as its
resolved solid color only, same as materials export always has.

**Layers** (SketchUp calls them "tags", import only) carry over too, as
Collections: every placement lands in a Collection named for its own
layer, one per distinct layer in the file, and a layer switched off in
SketchUp (its own Tags panel) imports with that Collection's visibility
(the same "eye"/camera icons a user can toggle in the Outliner) already
switched off — not shown by default, but still a normal, working part of
the scene, one click away from visible again. Verified against a real
145-definition structural-framing production file: 13 distinct layers,
2 genuinely hidden in the source file (cladding layers), both correctly
imported hidden.

**Not yet carried over:** no hole reconstruction on export (Blender's
mesh polygons have no native "outer boundary + holes" concept to read
one back from, unlike a real B-rep). Texture images are import-only - a
textured material exports as its resolved solid color, not the image.
Both layer and material export are deliberately NOT wired through an
imported file's own master/source objects (`<file> (source geometry)`) -
those sit in one Collection per unique *definition*, not per layer,
since the same definition can appear on several different tags (and,
less often, several different material contexts) across different
placements - so a straight reimport-then-reexport won't automatically
carry a file's original tags/paint back out yet; that needs exporting
via the placement Empties instead of their shared master mesh, future
scope.

**Editing imported geometry:** what you see placed in the viewport is a
Collection-Instance Empty, not a mesh — pressing Tab on it does nothing
on its own (an Empty has no mesh data of its own to edit). The real,
editable master mesh for every unique definition lives in the Outliner
under **`<filename> (source geometry)`**, deliberately excluded from the
View Layer (so it doesn't render duplicated at the origin) — enable that
collection's checkbox, select the object inside it, then Tab as normal.
Editing it updates every placement at once, matching how SketchUp's own
components behave.

**Exporting imported geometry back out:** select the *master* objects
under `<filename> (source geometry)` (not the placement Empties, which
`export_skp.py`'s `type == 'MESH'` filter skips) — otherwise the export
operator falls back to "every visible mesh in the scene," which won't be
what you expect.

## Why standalone

Raised directly in [IfcOpenShell/IfcOpenShell#9481](https://github.com/IfcOpenShell/IfcOpenShell/issues/9481)
by an IfcOpenShell maintainer: rather than build N one-off integrations
into every BIM add-on's own codebase, provide import/export as a vanilla
Blender extension and let any workflow — Bonsai included — consume it
through the standard File → Import/Export menus. Same reasoning behind
the [FreeCAD addon](https://github.com/iamahsanmehmood/freecad-openskp)
staying standalone too.

## Installation

Not yet published on the [Blender Extensions Platform](https://extensions.blender.org/).
Install manually, either way:

- **Latest release**: download the zip from the [latest release](https://github.com/iamahsanmehmood/blender-openskp/releases/latest).
- **Latest `main`, no release needed**: click the green **Code** button on this page → **Download ZIP** (no git required) — installs the same way, always up to date, useful when a fix has landed but hasn't made it into a packaged release yet. Verified directly: this repo's own zip download installs and runs correctly, no extraction or rebuilding needed.

Then, in Blender: **Edit → Preferences → Get Extensions → (dropdown, top right) → Install from Disk**, and select whichever zip you downloaded. Finally, **File → Import** or **File → Export → SketchUp (.skp)**.

That's it — no separate dependency step. `mapbox_earcut`, `shapely`, and
`defusedxml` (not bundled with Blender itself) ship as
[Python wheels](https://docs.blender.org/manual/en/latest/advanced/extensions/python_wheels.html)
inside the extension package for Windows/macOS (Intel + Apple Silicon)/Linux
x86_64, so Blender installs them automatically, offline, when you install
the extension — verified directly: a fresh install with no manual `pip`
step imports and runs correctly.

## Verification status, stated plainly

Tested end-to-end against real `.skp` fixtures via Blender's headless
mode, both directly (`import_skp.import_skp()`) and through the actual
registered operator (`bpy.ops.import_scene.openskp(...)`):

| Fixture | Unique meshes | Placements | Triangles | Result |
|---|---|---|---|---|
| `SU_File.skp` | 1 | 1 | 104 | matches expected exactly |
| `capilla_quiroz_v17.skp` | 3 | 4 | 871 | matches expected exactly |

Materials are cross-validated the same way: each fixture's resulting
Blender materials and per-polygon `material_index` assignment are checked
against an independently-reparsed `InstancedScene.gltf_materials` (not
just "some material got created") — `capilla_quiroz_v17.skp` alone
produces 13 distinct materials, including two translucent ones (alpha
0.5/0.7), correctly assigned per-face rather than defaulting to one color
for the whole object.

Textured materials are cross-validated the same way, against a real (not
synthetic) case: `capilla_quiroz_v17.skp` carries 3 real textures from a
SketchUp material library (concrete, roofing tile, translucent glass),
each checked for correct pixel dimensions (decoded independently from
the source file's own raw image bytes) and correct deduplication - the
translucent-glass texture used on both windows and the door loads as one
shared Blender Image, not two copies.

Layers: both committed fixtures only use SketchUp's default "Layer0", so
the automated test only pins the mechanics (a "Layer0" Collection exists,
holds every placement, isn't hidden) - real multi-layer/hidden-layer
grouping was verified separately against the same structural-framing
production file mentioned above (13 layers, 2 hidden, matching the source
file's own Tags panel state exactly).

Export was round-tripped through OpenSKP's own independent reader (not
this addon) — a 2m cube at a known world position exported and re-parsed
with every vertex landing exactly on the expected inch-space bounding box.

Layer export was checked the same way: three cubes (one left in the
default Collection, one each in Collections named "Studs" and "Plates")
exported and re-parsed, confirming exactly 3 distinct `Face.layer`
groups of 6 faces each - real per-face assignment, not just "a layer
record got written somewhere."

Material export the same way again: an opaque red cube, a 50%-alpha
translucent blue cube, and a plain unpainted cube, exported and
re-parsed independently - the resulting `Material.color`/`transparency`
values match exactly, and faces split into 3 distinct `Face.material_id`
groups of 6 each (the unpainted cube's own faces correctly carrying no
`material_id` at all, not a stray default).

The wheel-bundled dependencies were verified the same way: the addon was
fully uninstalled, its dependencies removed from Blender's own embedded
interpreter, then reinstalled from a freshly-built `.zip` and imported
against a real fixture with zero manual setup — not just "the manifest
looks right."

**Not yet measured:** performance on a large, real production file (the
FreeCAD addon's README has real numbers for a 358K-face file; this one
doesn't yet — collection-instancing should scale differently than
FreeCAD's B-rep approach, better on heavy component reuse, but that's a
claim to verify, not assert).

## Contributing

Bug reports, feature requests, and PRs are welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, running the tests, and
the PR process. This project follows the
[Contributor Covenant](CODE_OF_CONDUCT.md).

## License

GPL-3.0-or-later — see [LICENSE](LICENSE). (The vendored OpenSKP core
under `vendor/openskp/` stays MIT-licensed; see
[its own repository](https://github.com/iamahsanmehmood/openskp) for
that license text.)

# OpenSKP Import/Export for Blender

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
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

**Not yet carried over, either direction:** materials, layers, and layer
visibility, and no hole reconstruction on export (Blender's mesh polygons
have no native "outer boundary + holes" concept to read one back from,
unlike a real B-rep). Geometry only, for now.

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
Install manually:

1. Download the zip from the [latest release](https://github.com/iamahsanmehmood/blender-openskp/releases/latest).
2. In Blender: **Edit → Preferences → Get Extensions → (dropdown, top right) → Install from Disk**, and select the zip.
3. **File → Import** or **File → Export → SketchUp (.skp)**.

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

Export was round-tripped through OpenSKP's own independent reader (not
this addon) — a 2m cube at a known world position exported and re-parsed
with every vertex landing exactly on the expected inch-space bounding box.

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

MIT — see [LICENSE](LICENSE).

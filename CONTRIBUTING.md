# Contributing to OpenSKP Import/Export for Blender

Thanks for considering a contribution. This is a small addon - the
barrier to contributing should be low, and this doc is scoped to match.

## Code of Conduct

This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md).

## Getting set up

You need a real Blender install (4.2 or newer - this addon uses the
current Extension manifest, not the legacy `bl_info` system) - the
importer/exporter is built against Blender's own `bpy`/`bmesh`/`mathutils`
APIs and can only be exercised inside Blender's own Python environment.

```bash
git clone https://github.com/iamahsanmehmood/blender-openskp.git
```

The `vendor/openskp/` directory is a vendored copy of the pure-Python
[OpenSKP](https://github.com/iamahsanmehmood/openskp) package. It needs
three small dependencies (`mapbox_earcut`, `shapely`, `defusedxml`) that
aren't bundled with Blender - install them into Blender's own embedded
interpreter, not your system Python:

```bash
"<path to Blender>/<version>/python/bin/python.exe" -m pip install --target "<Blender addon-modules path>" mapbox_earcut shapely defusedxml
```

Where the addon-modules path is the one Blender itself already has on
`sys.path` for exactly this purpose - `%APPDATA%\Blender
Foundation\Blender\<version>\scripts\addons\modules` on Windows, the
equivalent under `~/.config` on Linux, `~/Library/Application Support` on
macOS. A plain `pip install` (no `--target`) will silently not work -
Blender's embedded interpreter runs with user-site-packages disabled, so
a normal user-site install is never on its `sys.path` (confirmed
directly, not assumed - this cost real debugging time to track down).

## Running tests

Both test files run under Blender's own headless mode, and behave like a
normal Python script (`__name__ == "__main__"`, unlike FreeCAD's
`freecadcmd`, which sets it to the filename - checked, not assumed):

```bash
blender --background --python tests/test_import.py
blender --background --python tests/test_export.py
```

`tests/fixtures/` holds small, real `.skp` files (borrowed from OpenSKP's
own public test fixtures). If you add a new one, keep it small and make
sure you have the right to distribute it.

## Making changes

### Branch naming

| Type | Format | Example |
|:-----|:-------|:--------|
| Feature | `feat/short-description` | `feat/carry-over-materials` |
| Bug fix | `fix/short-description` | `fix/nested-collection-naming` |
| Docs | `docs/short-description` | `docs/clarify-dependency-install` |

### Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):
`type(scope): short description`.

## Pull requests

1. Fork the repo, branch from `main`.
2. Run both test files against the fixtures and, where the change touches
   the import or export path, against a real `.skp` file of your own -
   numbers (mesh/triangle counts, timings) in the PR description are far
   more useful than "works for me."
3. Update the README if the change affects what's supported or verified -
   this project is deliberately honest about what's *not* yet done
   (materials/layers aren't carried over either direction, for instance);
   don't let that drift out of date.
4. Open a PR using the [template](.github/PULL_REQUEST_TEMPLATE.md).

## Reporting bugs / suggesting features

Use the [Bug Report](.github/ISSUE_TEMPLATE/bug_report.md) or
[Feature Request](.github/ISSUE_TEMPLATE/feature_request.md) templates.
For a bug, attaching the `.skp` file that triggers it (or a minimal one
that reproduces it) is the single biggest time-saver.

## Scope

Import/export of geometry, built on OpenSKP. Anything about the
underlying `.skp` format itself belongs in the
[OpenSKP repo](https://github.com/iamahsanmehmood/openskp), not here -
this addon is a thin Blender-specific bridge on top of it, kept
intentionally independent of any one BIM workflow rather than built into
a specific tool's codebase (see the
[discussion](https://github.com/IfcOpenShell/IfcOpenShell/issues/9481)
that shaped that call - the same reasoning behind the
[FreeCAD addon](https://github.com/iamahsanmehmood/freecad-openskp)
staying standalone too).

Thanks for helping make this better.

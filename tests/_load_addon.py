"""Loads import_skp.py/export_skp.py the way Blender's real extension
loader does - as submodules of a real package - rather than as bare
top-level modules.

Needed because both now use relative imports (``from .vendor import
openskp``) instead of a sys.path.insert() + bare ``import openskp``,
per Blender's own extension guidelines (manually constructing sys.path
entries to import into the global module namespace is a flagged policy
violation - developer.blender.org/docs/features/extensions/moderation/
guidelines - real add-on users hit this as warnings on every vendored
submodule, see github.com/iamahsanmehmood/blender-openskp/issues/2).
A relative import needs a real parent package to resolve against, which
a plain ``sys.path.insert(); import import_skp`` does not provide -
confirmed directly: that used to work before the relative-import change,
then failed with "attempted relative import with no known parent
package" once it didn't, which is what this file fixes.
"""
import importlib.util
import os
import sys

ADDON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG_NAME = "openskp_import_export_test"


def _load_module(name):
    """Loads ADDON_DIR/<name>.py as openskp_import_export_test.<name>,
    with the synthetic parent package registered first so its own
    relative imports (from .vendor import openskp) resolve correctly."""
    if _PKG_NAME not in sys.modules:
        pkg_spec = importlib.util.spec_from_file_location(
            _PKG_NAME,
            os.path.join(ADDON_DIR, "__init__.py"),
            submodule_search_locations=[ADDON_DIR],
        )
        pkg = importlib.util.module_from_spec(pkg_spec)
        sys.modules[_PKG_NAME] = pkg
        pkg_spec.loader.exec_module(pkg)

    full_name = f"{_PKG_NAME}.{name}"
    if full_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            full_name, os.path.join(ADDON_DIR, f"{name}.py")
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = _PKG_NAME
        sys.modules[full_name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[full_name]

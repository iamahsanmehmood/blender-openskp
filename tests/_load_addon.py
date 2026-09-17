"""Loads import_skp.py/export_skp.py the way Blender's real extension
loader does - as submodules of a real package - rather than as bare
top-level modules, so tests exercise the same module identity/`__package__`
that a real installed extension gets.
"""
import importlib.util
import os
import sys

ADDON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG_NAME = "openskp_import_export_test"


def ensure_openskp_importable():
    """Makes ``import openskp`` work under a bare `blender --python
    tests/test_*.py` run, outside of Blender's own extension loader.

    A real installed extension gets its bundled wheels extracted and put
    on sys.path automatically by Blender - that mechanism doesn't run
    here, so this puts the wheel itself on sys.path instead. A .whl is
    just a zip with the package at its root, and Python's zipimporter
    can import directly from a zip/whl path on sys.path without
    extracting it first."""
    wheel_dir = os.path.join(ADDON_DIR, "wheels")
    for name in os.listdir(wheel_dir):
        if name.startswith("openskp-") and name.endswith(".whl"):
            wheel_path = os.path.join(wheel_dir, name)
            if wheel_path not in sys.path:
                sys.path.insert(0, wheel_path)
            return
    raise RuntimeError(f"no openskp-*.whl found in {wheel_dir}")


def _load_module(name):
    """Loads ADDON_DIR/<name>.py as openskp_import_export_test.<name>,
    with the synthetic parent package registered first so it loads the
    same way Blender's real extension loader loads it - as a submodule
    of the add-on package, not a bare top-level script."""
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

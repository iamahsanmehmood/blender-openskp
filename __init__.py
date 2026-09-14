"""OpenSKP Import/Export - native SketchUp (.skp) support for Blender.

Registers File > Import > SketchUp (.skp) and File > Export > SketchUp
(.skp). See import_skp.py / export_skp.py for the actual implementation -
this file only wires them into Blender's operator/menu system.
"""
import bpy
from bpy.props import StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper


class IMPORT_OT_openskp(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.openskp"
    bl_label = "Import SketchUp (.skp)"
    bl_description = "Import a native SketchUp .skp file"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".skp"
    filter_glob: StringProperty(default="*.skp", options={"HIDDEN"})

    def execute(self, context):
        from . import import_skp

        try:
            stats = import_skp.import_skp(self.filepath, context)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            self.report({"ERROR"}, f"OpenSKP import failed: {exc}")
            return {"CANCELLED"}

        self.report(
            {"INFO"},
            f"Imported {stats['unique_meshes']} unique meshes, "
            f"{stats['placements']} instances placed",
        )
        return {"FINISHED"}


class EXPORT_OT_openskp(bpy.types.Operator, ExportHelper):
    bl_idname = "export_scene.openskp"
    bl_label = "Export SketchUp (.skp)"
    bl_description = "Export selected objects to a native SketchUp .skp file"
    bl_options = {"REGISTER"}

    filename_ext = ".skp"
    filter_glob: StringProperty(default="*.skp", options={"HIDDEN"})

    def execute(self, context):
        from . import export_skp

        try:
            stats = export_skp.export_skp(self.filepath, context)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"OpenSKP export failed: {exc}")
            return {"CANCELLED"}

        self.report(
            {"INFO"},
            f"Exported {stats['faces_written']} faces "
            f"({stats['objects_skipped']} objects skipped)",
        )
        return {"FINISHED"}


def _menu_func_import(self, context):
    self.layout.operator(IMPORT_OT_openskp.bl_idname, text="SketchUp (.skp)")


def _menu_func_export(self, context):
    self.layout.operator(EXPORT_OT_openskp.bl_idname, text="SketchUp (.skp)")


_classes = (IMPORT_OT_openskp, EXPORT_OT_openskp)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(_menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(_menu_func_export)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(_menu_func_import)
    bpy.types.TOPBAR_MT_file_export.remove(_menu_func_export)
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)

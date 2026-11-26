"""OpenSKP Import/Export - native SketchUp (.skp) support for Blender.

Registers File > Import > SketchUp (.skp) and File > Export > SketchUp
(.skp). See import_skp.py / export_skp.py for the actual implementation -
this file only wires them into Blender's operator/menu system.
"""
import bpy
from bpy.props import BoolProperty, EnumProperty, StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper


class IMPORT_OT_openskp(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.openskp"
    bl_label = "Import SketchUp (.skp)"
    bl_description = "Import a native SketchUp .skp file"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".skp"
    filter_glob: StringProperty(default="*.skp", options={"HIDDEN"})
    keep_hierarchy: BoolProperty(
        name="Keep SketchUp Hierarchy",
        description=(
            "Keep SketchUp's groups and components as a tree of parented Empties, "
            "instead of placing every part at the top level"
        ),
        default=True,
    )
    linked_duplicates: BoolProperty(
        name="Components as Linked Duplicates",
        description=(
            "Import component parts as mesh objects sharing their mesh data, placed "
            "directly in the scene instead of as collection instances"
        ),
        default=False,
    )
    import_annotations: BoolProperty(
        name="Import Annotations",
        description=(
            "Import SketchUp texts and linear dimensions as Blender text and line "
            "objects in an Annotations collection per tag"
        ),
        default=False,
    )
    unpainted_faces: EnumProperty(
        name="Unpainted Faces",
        description=(
            "How faces with no material painted on them or their enclosing "
            "component/group are coloured"
        ),
        items=(
            (
                "STYLE",
                "Style Colours",
                "Use the SketchUp style's front and back face colours",
            ),
            (
                "TAG",
                "Tag Colour",
                "Use each placement's SketchUp tag colour on both sides",
            ),
        ),
        default="STYLE",
    )

    def execute(self, context):
        from . import import_skp

        try:
            stats = import_skp.import_skp(
                self.filepath,
                context,
                keep_hierarchy=self.keep_hierarchy,
                import_annotations=self.import_annotations,
                linked_duplicates=self.linked_duplicates,
                unpainted_faces=self.unpainted_faces.lower(),
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            self.report({"ERROR"}, f"OpenSKP import failed: {exc}")
            return {"CANCELLED"}

        scenes_note = f", {stats['scenes_imported']} scene cameras" if stats.get("scenes_imported") else ""
        annot_total = stats.get("texts_imported", 0) + stats.get("dimensions_imported", 0)
        annot_note = f", {annot_total} annotations" if annot_total else ""
        self.report(
            {"INFO"},
            f"Imported {stats['unique_meshes']} unique meshes "
            f"({stats['placements']} placed), "
            f"{stats['unique_curve_meshes']} loose-edge groups "
            f"({stats['curve_placements']} placed){scenes_note}{annot_note}",
        )
        warnings = stats.get("warnings") or []
        if warnings:
            # Blender's status bar shows one line; the console has every message
            more = f" (and {len(warnings) - 1} more, see the console)" if len(warnings) > 1 else ""
            self.report({"WARNING"}, f"OpenSKP: {warnings[0]}{more}")
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

        comp_note = (
            f", {stats['definitions_written']} components ({stats['instances_written']} instances)"
            if stats.get("instances_written")
            else ""
        )
        self.report(
            {"INFO"},
            f"Exported {stats['faces_written']} faces{comp_note} "
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

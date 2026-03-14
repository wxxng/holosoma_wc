#!/usr/bin/env python3
"""Generate g1_43dof_<object>.xml variants for grab/contact_meshes objects."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_XML = ROOT / "src/holosoma/holosoma/data/robots/g1/g1_43dof.xml"
GRAB_DIR = ROOT / "src/holosoma/holosoma/data/grab/contact_meshes_obj"
OUT_DIR = ROOT / "src/holosoma/holosoma/data/robots/g1/g1_object"

TABLE_MATERIAL_NAME = "table_light_brown"
TABLE_MESH_NAME = "table_mesh"
TABLE_COLLISION_MESH_NAME = "table_collision_0"
TABLE_MESH_FILE = "../../../grab/contact_meshes/table/table/table.obj"
TABLE_COLLISION_MESH_FILE = "../../../grab/contact_meshes/table/table/table_collision_0.obj"
OBJECT_DENSITY = "200"


def _indent(elem: ET.Element, level: int = 0) -> None:
    """In-place pretty print indentation."""
    indent_str = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent_str + "  "
        for child in elem:
            _indent(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent_str
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent_str


def _ensure_solver_option(root: ET.Element) -> None:
    """Ensure generated XML includes the desired solver option defaults."""
    option = root.find("option")
    if option is None:
        compiler = root.find("compiler")
        insert_idx = (list(root).index(compiler) + 1) if compiler is not None else 0
        option = ET.Element("option")
        root.insert(insert_idx, option)

    option.set("cone", "elliptic")
    option.set("solver", "Newton")
    option.set("impratio", "5")
    option.set("tolerance", "1e-10")
    option.set("iterations", "100")


def _load_object_xml(obj_dir: Path, name: str) -> ET.ElementTree:
    obj_xml = obj_dir / f"{name}.xml"
    if not obj_xml.exists():
        raise FileNotFoundError(f"Object xml not found: {obj_xml}")
    return ET.parse(obj_xml)


def _object_meshes(obj_tree: ET.ElementTree, obj_name: str) -> list[ET.Element]:
    asset = obj_tree.getroot().find("asset")
    meshes: list[ET.Element] = []
    if asset is None:
        return meshes
    for mesh in asset.findall("mesh"):
        new_mesh = ET.Element("mesh", attrib=dict(mesh.attrib))
        file_attr = new_mesh.get("file", "")
        # Re-root to g1/meshes (meshdir) using ../../../grab path
        if file_attr:
            new_mesh.set(
                "file",
                f"../../../grab/contact_meshes_obj/{obj_name}/{Path(file_attr).name}",
            )
        meshes.append(new_mesh)
    return meshes


def _object_geoms(obj_tree: ET.ElementTree) -> list[ET.Element]:
    worldbody = obj_tree.getroot().find("worldbody")
    if worldbody is None:
        return []
    body = worldbody.find("body")
    if body is None:
        return []
    geoms: list[ET.Element] = []
    for geom in body.findall("geom"):
        new_geom = ET.Element("geom", attrib=dict(geom.attrib))
        geom_class = new_geom.get("class", "")
        if geom_class == "collision":
            # Ensure collisions with robot + table, high friction for grasping
            new_geom.set("type", "mesh")
            new_geom.set("contype", "3")
            new_geom.set("conaffinity", "3")
            new_geom.set("condim", "3")
            new_geom.set("friction", "2 0.5 0.5")
            new_geom.set("group", "1")
            new_geom.set("density", OBJECT_DENSITY)
            new_geom.attrib.pop("mass", None)
        elif geom_class == "visual":
            new_geom.set("contype", "0")
            new_geom.set("conaffinity", "0")
        geoms.append(new_geom)
    return geoms


def _ensure_object_default_classes(root: ET.Element) -> None:
    """Add visual/collision default classes for object geoms if not already present.

    Object XMLs from contact_meshes reference class='visual' and class='collision'.
    These must be defined in the robot model's default hierarchy.
    """
    default = root.find("default")
    if default is None:
        return
    # Find the main robot default class (first named <default> child)
    main_default = default.find("default")
    if main_default is None:
        return

    if main_default.find("default[@class='visual']") is None:
        visual = ET.SubElement(main_default, "default")
        visual.set("class", "visual")
        geom_elem = ET.SubElement(visual, "geom")
        geom_elem.set("group", "2")
        geom_elem.set("type", "mesh")
        geom_elem.set("contype", "0")
        geom_elem.set("conaffinity", "0")
        geom_elem.set("density", "0")

    if main_default.find("default[@class='collision']") is None:
        collision = ET.SubElement(main_default, "default")
        collision.set("class", "collision")
        geom_elem = ET.SubElement(collision, "geom")
        geom_elem.set("group", "3")
        geom_elem.set("type", "mesh")
        # geom_elem.set("condim", "6")  # full friction: tangential + torsional + rolling
        geom_elem.set("solref", "0.001 1")
        geom_elem.set("solimp", "0.99 0.99 0.001")


def _ensure_table_assets(root: ET.Element) -> None:
    asset = root.find("asset")
    if asset is None:
        raise RuntimeError("Template xml missing <asset> section.")

    if asset.find(f"material[@name='{TABLE_MATERIAL_NAME}']") is None:
        mat = ET.Element("material")
        mat.set("name", TABLE_MATERIAL_NAME)
        mat.set("rgba", "0.734 0.617 0.507 1")
        asset.insert(0, mat)

    if asset.find(f"mesh[@name='{TABLE_MESH_NAME}']") is None:
        mesh = ET.Element("mesh")
        mesh.set("name", TABLE_MESH_NAME)
        mesh.set("file", TABLE_MESH_FILE)
        asset.append(mesh)

    if asset.find(f"mesh[@name='{TABLE_COLLISION_MESH_NAME}']") is None:
        mesh = ET.Element("mesh")
        mesh.set("name", TABLE_COLLISION_MESH_NAME)
        mesh.set("file", TABLE_COLLISION_MESH_FILE)
        asset.append(mesh)


def _make_table_body() -> ET.Element:
    table = ET.Element("body")
    table.set("name", "table")
    table.set("pos", "0.9960365 0 0.844")

    vis = ET.SubElement(table, "geom")
    vis.set("mesh", TABLE_MESH_NAME)
    vis.set("class", "visual")
    vis.set("material", TABLE_MATERIAL_NAME)
    vis.set("contype", "0")
    vis.set("conaffinity", "0")

    col = ET.SubElement(table, "geom")
    col.set("mesh", TABLE_COLLISION_MESH_NAME)
    col.set("class", "collision")
    col.set("type", "mesh")
    col.set("contype", "2")
    col.set("conaffinity", "3")
    col.set("friction", "0.7 0.005 0.0001")
    return table


def _make_object_body(geoms: list[ET.Element]) -> ET.Element:
    obj_body = ET.Element("body")
    obj_body.set("name", "object")
    obj_body.set("pos", "0.9960365 0 0.894")
    ET.SubElement(obj_body, "freejoint")
    for geom in geoms:
        obj_body.append(geom)
    return obj_body


def generate() -> None:
    if not TEMPLATE_XML.exists():
        raise FileNotFoundError(f"Template xml not found: {TEMPLATE_XML}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for obj_dir in sorted(GRAB_DIR.iterdir()):
        if not obj_dir.is_dir():
            continue
        if obj_dir.name in {"__MACOSX", "_mjcf_out", "table"}:
            continue
        obj_name = obj_dir.name
        obj_xml_path = obj_dir / f"{obj_name}.xml"
        if not obj_xml_path.exists():
            continue

        # Load template and object xml
        template_tree = ET.parse(TEMPLATE_XML)
        template_root = template_tree.getroot()
        obj_tree = _load_object_xml(obj_dir, obj_name)

        # Fix meshdir: g1_43dof.xml uses "./meshes/", output is in g1_object/ → "../meshes"
        compiler = template_root.find("compiler")
        if compiler is not None:
            compiler.set("meshdir", "../meshes")

        _ensure_solver_option(template_root)

        # Ensure visual/collision default classes exist for object geoms
        _ensure_object_default_classes(template_root)

        # Ensure table assets (mesh + material) exist in the first <asset>
        _ensure_table_assets(template_root)

        # Add object meshes to the first <asset> (robot meshes asset)
        asset = template_root.find("asset")
        if asset is None:
            raise RuntimeError("Template xml missing <asset> section.")
        for mesh in _object_meshes(obj_tree, obj_name):
            asset.append(mesh)

        # Find first worldbody (robot's worldbody, not scene setup worldbody)
        worldbody = template_root.find("worldbody")
        if worldbody is None:
            raise RuntimeError("Template xml missing <worldbody> section.")

        # Add table body if not already present
        if worldbody.find("body[@name='table']") is None:
            worldbody.append(_make_table_body())

        # Add object body if not already present
        if worldbody.find("body[@name='object']") is None:
            worldbody.append(_make_object_body(_object_geoms(obj_tree)))
        else:
            # Existing object body: replace geoms only
            obj_body = worldbody.find("body[@name='object']")
            for inertial in list(obj_body.findall("inertial")):
                obj_body.remove(inertial)
            for geom in list(obj_body.findall("geom")):
                obj_body.remove(geom)
            for geom in _object_geoms(obj_tree):
                obj_body.append(geom)

        # Write output
        out_path = OUT_DIR / f"g1_43dof_{obj_name}.xml"

        if out_path.exists():
            import shutil
            backup_path = out_path.with_suffix(".xml.bak")
            shutil.copy2(out_path, backup_path)
            print(f"Backed up existing file to: {backup_path.name}")

        _indent(template_root)
        template_tree.write(out_path, encoding="utf-8", xml_declaration=False)
        print(f"Generated: {out_path.name}")


if __name__ == "__main__":
    generate()

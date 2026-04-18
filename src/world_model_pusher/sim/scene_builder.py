"""Builds a compiled mujoco.MjModel from a SceneConfig via XML manipulation."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco  # type: ignore[import-untyped]
import numpy as np
from lxml import etree  # type: ignore[import-untyped]

from .scene_config import ObjectConfig, SceneConfig

_ASSETS_DIR = Path(__file__).parent.parent.parent.parent / "assets" / "mujoco"
_TABLE_SCENE_XML = _ASSETS_DIR / "table_scene.xml"
_SIMPLE_ARM_XML = _ASSETS_DIR / "simple_arm.xml"
_SO101_SCENE_XML = _ASSETS_DIR / "so101_scene.xml"
_SO101_MESHDIR = _ASSETS_DIR / "trs_so_arm100"


# ---------------------------------------------------------------------------
# Helper: build an XML body element for a primitive ObjectConfig
# ---------------------------------------------------------------------------

def _object_geom_element(cfg: ObjectConfig, name: str) -> etree._Element:
    """Return an lxml ``<geom>`` element for the given object config."""
    geom = etree.Element("geom")
    geom.set("name", f"{name}_geom")
    rgba = " ".join(f"{v:.3f}" for v in cfg.color)
    geom.set("rgba", rgba)
    geom.set("mass", str(cfg.mass))
    geom.set("friction", f"{cfg.friction:.3f} 0.02 0.001")

    shape = cfg.shape
    if shape == "box":
        geom.set("type", "box")
        s = cfg.size
        geom.set("size", f"{s[0]:.4f} {s[1]:.4f} {s[2]:.4f}")
    elif shape == "cylinder":
        geom.set("type", "cylinder")
        geom.set("size", f"{cfg.size[0]:.4f} {cfg.size[1]:.4f}")
    elif shape == "sphere":
        geom.set("type", "sphere")
        geom.set("size", f"{cfg.size[0]:.4f}")
    elif shape == "capsule":
        geom.set("type", "capsule")
        geom.set("size", f"{cfg.size[0]:.4f} {cfg.size[1]:.4f}")
    else:
        geom.set("type", "box")
        geom.set("size", "0.03 0.03 0.03")

    return geom


def _make_object_body(
    cfg: ObjectConfig,
    name: str,
    table_top_z: float,
    colliding: bool,
) -> etree._Element:
    """Build a ``<body>`` element placed on the table surface."""
    body = etree.Element("body")
    body.set("name", name)

    half_z = _half_z_for_object(cfg)
    pos_z = table_top_z + half_z
    yaw = cfg.orientation
    body.set("pos", f"{cfg.pos[0]:.4f} {cfg.pos[1]:.4f} {pos_z:.4f}")
    body.set("euler", f"0 0 {yaw:.4f}")

    free_joint = etree.SubElement(body, "joint")
    free_joint.set("type", "free")

    inertial = etree.SubElement(body, "inertial")
    inertial.set("pos", "0 0 0")
    inertial.set("mass", f"{cfg.mass:.6f}")
    r = cfg.size[0] if cfg.size else 0.03
    ival = max(2.0 / 5.0 * cfg.mass * r * r, 1e-6)
    inertial.set("diaginertia", f"{ival:.6e} {ival:.6e} {ival:.6e}")

    geom = _object_geom_element(cfg, name)
    if not colliding:
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    body.append(geom)
    return body


def _half_z_for_object(cfg: ObjectConfig) -> float:
    shape = cfg.shape
    s = cfg.size
    if shape == "box":
        return s[2] if len(s) > 2 else s[0]
    elif shape == "cylinder":
        return s[1] if len(s) > 1 else s[0]
    elif shape == "sphere":
        return s[0]
    elif shape == "capsule":
        return (s[1] + s[0]) if len(s) > 1 else s[0]
    return 0.03


# ---------------------------------------------------------------------------
# Camera: convert look-at to MuJoCo euler
# ---------------------------------------------------------------------------

def _lookat_to_euler(
        pos: list[float], look_at: list[float]) -> tuple[str, str]:
    """Return (pos_str, euler_str) for a camera pointing from pos toward look_at."""
    px, py, pz = pos
    lx, ly, lz = look_at
    fwd = np.array([lx - px, ly - py, lz - pz], dtype=float)
    dist = np.linalg.norm(fwd)
    if dist < 1e-6:
        fwd = np.array([0.0, 0.0, -1.0])
    else:
        fwd /= dist

    pitch = math.atan2(-fwd[2], math.hypot(fwd[0], fwd[1]))
    yaw = math.atan2(fwd[1], fwd[0]) + math.pi / 2

    pos_str = f"{px:.4f} {py:.4f} {pz:.4f}"
    euler_str = f"{pitch:.4f} 0 {yaw:.4f}"
    return pos_str, euler_str


# ---------------------------------------------------------------------------
# Base MJCF assembly
# ---------------------------------------------------------------------------

def _load_base_xml(config: SceneConfig) -> etree._Element:
    """Load and return the root element of the appropriate base MJCF."""
    parser = etree.XMLParser(remove_blank_text=True)
    if config.robot_type == "so100":
        root = etree.parse(str(_SO101_SCENE_XML), parser).getroot()
        compiler = root.find("compiler")
        if compiler is not None:
            compiler.set("meshdir", str(_SO101_MESHDIR))
        return root

    # Simple arm: merge table_scene.xml and simple_arm.xml
    table_root = etree.parse(str(_TABLE_SCENE_XML), parser).getroot()
    arm_root = etree.parse(str(_SIMPLE_ARM_XML), parser).getroot()

    worldbody = table_root.find("worldbody")
    for child in arm_root.find("worldbody"):
        worldbody.append(child)

    equality = arm_root.find("equality")
    if equality is not None:
        table_root.append(equality)

    return table_root


# ---------------------------------------------------------------------------
# SceneBuilder
# ---------------------------------------------------------------------------

class SceneBuilder:
    """Injects objects, camera, and lighting into a base MJCF and compiles it."""

    def build(
        self,
        config: SceneConfig,
        render_size: tuple[int, int] | None = None,
    ) -> mujoco.MjModel:
        """Parse base MJCF, inject scene elements, compile and return MjModel."""
        root = _load_base_xml(config)

        if render_size is not None:
            h, w = render_size
            visual = root.find("visual")
            if visual is None:
                visual = etree.SubElement(root, "visual")
            global_elem = visual.find("global")
            if global_elem is None:
                global_elem = etree.SubElement(visual, "global")
            global_elem.set("offheight", str(h))
            global_elem.set("offwidth", str(w))

        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("Base MJCF has no <worldbody> element")

        table_top_z = 0.04

        table_geom = worldbody.find(".//geom[@name='table_top']")
        if table_geom is not None:
            rgba = " ".join(f"{v:.3f}" for v in config.table_color)
            table_geom.set("rgba", rgba)
            hz = config.table_size[2]
            hx, hy = config.table_size[0], config.table_size[1]
            table_geom.set("size", f"{hx:.3f} {hy:.3f} {hz:.3f}")
            table_geom.set("friction", f"{config.table_friction:.3f} 0.02 0.001")

        target_body = _make_object_body(
            config.target, "target_object", table_top_z, colliding=True
        )
        worldbody.append(target_body)

        for i, obs_cfg in enumerate(config.obstacles):
            obs_body = _make_object_body(
                obs_cfg, f"obstacle_{i}", table_top_z, colliding=True)
            worldbody.append(obs_body)

        for i, cl_cfg in enumerate(config.clutter):
            cl_body = _make_object_body(
                cl_cfg, f"clutter_{i}", table_top_z, colliding=False)
            worldbody.append(cl_body)

        cam_elem = root.find(".//camera[@name='main_camera']")
        if cam_elem is None:
            cam_elem = etree.SubElement(worldbody, "camera")
            cam_elem.set("name", "main_camera")
        pos_str, euler_str = _lookat_to_euler(
            config.camera.pos, config.camera.look_at)
        cam_elem.set("pos", pos_str)
        cam_elem.set("euler", euler_str)
        cam_elem.set("fovy", f"{config.camera.fov:.1f}")

        for lt in root.findall(".//light"):
            if lt.get("name", "") == "scene_light":
                lt.getparent().remove(lt)  # type: ignore[union-attr]

        light = etree.SubElement(worldbody, "light")
        light.set("name", "scene_light")
        d = config.lighting.direction
        light.set("dir", f"{d[0]:.3f} {d[1]:.3f} {d[2]:.3f}")
        iv = config.lighting.intensity
        light.set("diffuse", f"{iv:.3f} {iv:.3f} {iv:.3f}")
        av = config.lighting.ambient
        light.set("ambient", f"{av:.3f} {av:.3f} {av:.3f}")
        light.set("directional", "true")
        light.set("castshadow", "false")

        xml_str = etree.tostring(root, pretty_print=True, encoding="unicode")
        model = mujoco.MjModel.from_xml_string(xml_str)
        return model

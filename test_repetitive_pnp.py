#!/usr/bin/env python3
"""
Open-box pick-and-place policy test.

Spawns robot + object + table + open box in MuJoCo, generates a trajectory
from the object's spawn position to above the open box, then runs
auto-stabilization followed by BPS policy inference and releases the object
into the box.

No PKL required — robot init pose and object name are provided directly.

Usage:
    python test_repetitive_pnp.py --obj-name cubemedium
    python test_repetitive_pnp.py --obj-name cubemedium --waypoint-z 0.6
    python test_repetitive_pnp.py --obj-name cubemedium --randomize
    python test_repetitive_pnp.py --obj-name cubemedium --randomize --offscreen --record
"""

import argparse
import importlib.util
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import joblib
import mujoco
import numpy as np
import onnxruntime

from g1_robot_common import (
    DOF_NAMES, DEFAULT_DOF_ANGLES,
    ACTION_ORDER_43DOF, ACTION_SCALE, ACTION_OFFSET, ACTION_CLIP_MIN, ACTION_CLIP_MAX,
    BODY_JOINT_NAMES, HAND_JOINT_NAMES,
    name_indices, mj_hinge_addrs, mj_actuator_ids, apply_pd_control,
    quat_rotate_inverse, SimpleVideoRecorder, run_mujoco_loop,
)

# ── Import trajectory utilities from generate_cabinet_path.py ─────────────
_gcp_spec = importlib.util.spec_from_file_location(
    "generate_cabinet_path",
    Path(__file__).parent / "generate_cabinet_path.py",
)
_gcp = importlib.util.module_from_spec(_gcp_spec)
_gcp_spec.loader.exec_module(_gcp)
make_obj_pos_hold_src_tgt = _gcp.make_obj_pos_hold_src_tgt
make_obj_rot_intervals    = _gcp.make_obj_rot_intervals

# Precomputed index maps (config order → body/hand subsets)
BODY_INDICES = name_indices(BODY_JOINT_NAMES, DOF_NAMES)   # (29,)
HAND_INDICES = name_indices(HAND_JOINT_NAMES, DOF_NAMES)   # (14,)

np.set_printoptions(precision=4, suppress=True)

# ── Spawn positions & rotations ────────────────────────────────────────────
OBJ_SPAWN_DEFAULT   = np.array([-0.04082961, -0.98491963, 0.8744048704122025],
                                dtype=np.float32)
TABLE_SPAWN_DEFAULT = np.array([-0.04411870, -1.02271030, 0.8286774158477783],
                                dtype=np.float32)
# Default box XY position (same region as cabinet default)
BOX_XY_DEFAULT      = np.array([-0.08303487, 0.80154856], dtype=np.float32)
OBJ_TABLE_XY_OFFSET_DEFAULT = OBJ_SPAWN_DEFAULT[:2] - TABLE_SPAWN_DEFAULT[:2]
OBJ_RESPAWN_CANDIDATES = ("cubemedium", "cylindermedium", "waterbottle", "apple")
INACTIVE_OBJECT_FAR_BASE_X = 40.0
INACTIVE_OBJECT_FAR_BASE_Y = 40.0
INACTIVE_OBJECT_FAR_STEP_X = 5.0
INACTIVE_OBJECT_FAR_Z = 1.0

# Open box height (from XML: walls top at z=0.21 from body origin)
BOX_HEIGHT          = 0.21   # m  (body_z + BOX_HEIGHT = opening z)
# Box body z = waypoint_z - BOX_BODY_Z_OFFSET
# → opening will be waypoint_z + (BOX_HEIGHT - BOX_BODY_Z_OFFSET) above ground
BOX_BODY_Z_OFFSET   = 0.1    # box body is 0.1 m below waypoint_z
DEFAULT_WAYPOINT_Z  = 0.55   # default release height (middle of [0.3, 0.8])
WAYPOINT_Z_RANGE    = (0.3, 0.8)

# Locomotion-to-table phase constants
DESIRED_TABLE_ROBOT_DIST = 1.0   # m — target distance from table for waypoint
LOCO_KP_XY              = 2.0   # P-gain: xy position error (m)  → vel cmd [-1, 1]
LOCO_KP_YAW             = 1.0   # P-gain: yaw error (rad)        → yaw cmd [-1, 1]
LOCO_KI_XY              = 0.5   # I-gain: xy position error integral
LOCO_KI_YAW             = 0.05  # I-gain: yaw error integral
LOCO_KD_XY              = 0.05  # D-gain: xy position error derivative
LOCO_KD_YAW             = 0.05  # D-gain: yaw error derivative
LOCO_POS_TOL            = 0.2   # m   — convergence: position error threshold
LOCO_YAW_TOL            = 0.3   # rad — convergence: orientation error threshold
LOCO_INTEGRAL_CLIP      = 3.0   # anti-windup clip for integral term
LOCO_TIMEOUT_SEC        = 10.0  # s   — fail and terminate if waypoint not reached in time

# Rotations stored as wxyz (MuJoCo convention)
TABLE_QUAT_WXYZ = np.array([0.70926454, 0.70392106, 0.02927242, 0.02412632], dtype=np.float32)
OBJ_QUAT_WXYZ   = np.array([1, 0, 0, 0], dtype=np.float32)

# Default robot init state
DEFAULT_ROBOT_ROOT_POS = np.array([0.0, 0.0, 0.79603237], dtype=np.float32)
DEFAULT_ROBOT_ROOT_QUAT_XYZW = np.array([-0.02295663, -0.01787878, -0.73244804, 0.6802008],
                                         dtype=np.float32)
DEFAULT_ROBOT_DOF_POS_43 = np.array([
     0.07842931,  0.01744985,  0.18222341,  0.04068308, -0.14910302,
     0.06798502,  0.03149052, -0.0471192 , -0.1895515 ,  0.14110036,
    -0.06096252,  0.08679032,  0.00340176,  0.04049601, -0.01169985,
    -0.088287  ,  1.5588585 ,  0.18558194,  1.5912597 ,  0.29973707,
     0.07963584,  0.1249508 , -0.52633935,  0.65987134,  0.6556139 ,
    -0.24042495, -0.24306054, -0.3165799 , -0.31889918, -0.02020167,
    -1.5575434 , -0.49221078,  1.5078542 , -0.1111234 ,  0.06493038,
    -0.1477766 , -0.6897597 , -0.5842331 , -0.5836255 ,  0.2817436 ,
     0.2844584 ,  0.29808676,  0.30005792,
], dtype=np.float32)

G1_DEX3_JOINT_NAMES = [
    "left_hip_pitch_joint",  "left_hip_roll_joint",  "left_hip_yaw_joint",
    "left_knee_joint",       "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint",      "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",       "waist_roll_joint",     "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "left_hand_thumb_0_joint",  "left_hand_thumb_1_joint",  "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint",  "left_hand_index_1_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "right_hand_thumb_0_joint",  "right_hand_thumb_1_joint",  "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",  "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
]

STABILIZATION_OBS_ORDER_29DOF = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
STABILIZATION_ACTION_ORDER_29DOF = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
G1_MOTION_JOINT_NAMES_29 = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


# ═══════════════════════════════════════════════════════════════════════════
# Robot init state helpers
# ═══════════════════════════════════════════════════════════════════════════

def build_init_q(dof_pos_43_dex3order: np.ndarray) -> np.ndarray:
    config_idx = name_indices(G1_DEX3_JOINT_NAMES, DOF_NAMES)
    init_q = DEFAULT_DOF_ANGLES.copy()
    init_q[config_idx] = dof_pos_43_dex3order
    return init_q


def build_stabilization_refs(
    dof_pos_43_dex3order: np.ndarray,
    T: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    body_in_dex3 = name_indices(G1_MOTION_JOINT_NAMES_29, G1_DEX3_JOINT_NAMES)
    hand_in_dex3 = name_indices(HAND_JOINT_NAMES, G1_DEX3_JOINT_NAMES)
    body_pos_init = dof_pos_43_dex3order[body_in_dex3]
    hand_pos_init = dof_pos_43_dex3order[hand_in_dex3]
    stab_pos = np.tile(body_pos_init, (T, 1)).astype(np.float32)
    stab_vel = np.zeros((T, 29), dtype=np.float32)
    hand_tgt = np.tile(hand_pos_init, (T, 1)).astype(np.float32)
    return stab_pos, stab_vel, hand_tgt


# ═══════════════════════════════════════════════════════════════════════════
# Box XML merge
# ═══════════════════════════════════════════════════════════════════════════

def _clone_et_elem(elem: ET.Element) -> ET.Element:
    return ET.fromstring(ET.tostring(elem, encoding="unicode"))


def _extract_object_payload(
    xml_path: Path,
    obj_name: str,
) -> tuple[ET.Element, list[ET.Element]]:
    """Extract object body and required object meshes from a g1_43dof_<obj>.xml.

    Candidate XMLs may omit mesh `name` attributes and rely on MuJoCo's
    implicit naming from file stem. We assign explicit unique names here and
    rewrite geom mesh references accordingly to avoid missing mesh errors.
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    worldbody = root.find("worldbody")
    asset = root.find("asset")
    if worldbody is None:
        raise RuntimeError(f"Missing worldbody in {xml_path}")
    obj_body = worldbody.find("body[@name='object']")
    if obj_body is None:
        raise RuntimeError(f"Missing object body in {xml_path}")
    obj_body_clone = _clone_et_elem(obj_body)
    used_mesh_names = {
        str(g.get("mesh")) for g in obj_body_clone.findall(".//geom")
        if g.get("mesh") is not None
    }
    mesh_clones: list[ET.Element] = []
    mesh_name_map: dict[str, str] = {}
    if asset is not None:
        for mesh in asset.findall("mesh"):
            file_attr = mesh.get("file")
            raw_name = mesh.get("name")
            src_name = raw_name
            if src_name is None and file_attr:
                src_name = Path(file_attr).stem
            if src_name is None or src_name not in used_mesh_names:
                continue
            unique_name = f"{obj_name}__{src_name}"
            mesh_elem = _clone_et_elem(mesh)
            mesh_elem.set("name", unique_name)
            mesh_clones.append(mesh_elem)
            mesh_name_map[src_name] = unique_name

    for geom in obj_body_clone.findall(".//geom"):
        mesh_ref = geom.get("mesh")
        if mesh_ref in mesh_name_map:
            geom.set("mesh", mesh_name_map[mesh_ref])
    return obj_body_clone, mesh_clones


def build_model_with_box(
    robot_xml_path: str,
    box_xml_path: str,
    box_pos,
    box_quat_wxyz=None,
    object_candidates: list[str] | tuple[str, ...] | None = None,
    base_obj_name: str | None = None,
) -> mujoco.MjModel:
    """Merge open-box geometry into the robot XML and return a MjModel.

    Box worldbody contents (excluding ground plane) are wrapped in a body at
    box_pos/quat. A temp file is written next to the robot XML so relative
    mesh paths resolve correctly.
    """
    robot_tree = ET.parse(robot_xml_path)
    box_tree   = ET.parse(box_xml_path)
    robot_root = robot_tree.getroot()
    box_root   = box_tree.getroot()
    robot_worldbody = robot_root.find("worldbody")
    robot_asset = robot_root.find("asset")
    if robot_worldbody is None:
        raise RuntimeError("Robot XML missing <worldbody>.")
    if robot_asset is None:
        raise RuntimeError("Robot XML missing <asset>.")

    # Add multi-object candidates as separate freejoint bodies:
    # object_<name>. Inactive ones can later be moved underground.
    if object_candidates:
        if base_obj_name is None:
            raise ValueError("base_obj_name is required when object_candidates is used.")
        base_body = robot_worldbody.find("body[@name='object']")
        if base_body is None:
            raise RuntimeError("Base robot XML missing body 'object'.")
        base_body.set("name", f"object_{base_obj_name}")

        existing_mesh_names = {
            m.get("name") for m in robot_asset.findall("mesh") if m.get("name") is not None
        }
        existing_body_names = {
            b.get("name") for b in robot_worldbody.findall("body") if b.get("name") is not None
        }

        for obj_name in dict.fromkeys(object_candidates):
            body_name = f"object_{obj_name}"
            if body_name in existing_body_names:
                continue
            cand_xml = Path(robot_xml_path).parent / f"g1_43dof_{obj_name}.xml"
            if not cand_xml.exists():
                print(f"[WARN] Candidate XML not found, skipping '{obj_name}': {cand_xml}")
                continue
            cand_body, cand_meshes = _extract_object_payload(cand_xml, obj_name)
            cand_body.set("name", body_name)
            robot_worldbody.append(cand_body)
            existing_body_names.add(body_name)
            for mesh_elem in cand_meshes:
                mesh_name = mesh_elem.get("name")
                if mesh_name is None or mesh_name in existing_mesh_names:
                    continue
                robot_asset.append(mesh_elem)
                existing_mesh_names.add(mesh_name)

    # Wrap box worldbody in a positioned body
    cx, cy, cz = float(box_pos[0]), float(box_pos[1]), float(box_pos[2])
    body_attrib = {"name": "box_root", "pos": f"{cx} {cy} {cz}"}
    if box_quat_wxyz is not None:
        w, x, y, z = [float(v) for v in box_quat_wxyz]
        body_attrib["quat"] = f"{w} {x} {y} {z}"
    box_body = ET.SubElement(
        robot_worldbody,
        "body",
        attrib=body_attrib,
    )
    for child in list(box_root.find("worldbody")):
        # Skip ground plane geom and cameras
        if child.tag == "camera":
            continue
        if child.tag == "geom" and child.get("name") == "ground":
            continue
        box_body.append(_clone_et_elem(child))

    robot_dir    = Path(robot_xml_path).parent
    combined_str = ET.tostring(robot_root, encoding="unicode")
    fd, tmp_path = tempfile.mkstemp(suffix=".xml", prefix="_box_combined_", dir=str(robot_dir))
    try:
        with os.fdopen(fd, "w") as f:
            f.write(combined_str)
        model = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.unlink(tmp_path)
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Trajectory generation
# ═══════════════════════════════════════════════════════════════════════════

def _trap_segment_3d(
    p_start: np.ndarray, p_end: np.ndarray,
    v_max: float, a_max: float, dt: float,
) -> list:
    d = p_end - p_start
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return [tuple(p_start.tolist())]
    direction = d / L
    t_acc, t_cruise, t_total, v_peak = _gcp._trap_profile(L, v_max, a_max)
    n = int(np.floor(t_total / dt))
    pts = []
    for k in range(1, n + 1):
        s = _gcp._trap_s_at_t(k * dt, L, a_max, t_acc, t_cruise, v_peak)
        pts.append(tuple((p_start + s * direction).tolist()))
    pts.append(tuple(p_end.tolist()))
    return pts


def generate_box_trajectory(
    obj_spawn: np.ndarray,
    box_xy: np.ndarray,
    waypoint_z: float,
    motion_fps: float,
    hold_sec: float = 2.5,
    tgt_hold_sec: float = 1.0,
    obj_rot0_wxyz: np.ndarray | None = None,
    lift_height: float = 0.15,
    speed_scale: float = 1.0,
) -> dict:
    """Generate object trajectory from spawn to above/into the open box.

    Phases:
      1. Hold at spawn for hold_sec seconds.
      2. Lift object vertically by lift_height using trapezoidal profile.
      3. Move diagonally from lifted spawn to hover point (above box, at
         waypoint_z + 0.15 m for clearance).
      4. Descend into box to waypoint_z (trapezoidal, slower).
      5. Hold at target for tgt_hold_sec seconds.

    Args:
        obj_spawn:    Object world position at spawn (3,).
        box_xy:       Box center XY position in world frame (2,).
        waypoint_z:   Z height of the release point (inside/just below box opening).
        motion_fps:   Trajectory sampling frequency.

    Returns:
        dict with obj_pos (T,3), obj_rot (T,4) wxyz, motion_length, motion_fps.
    """
    dt = 1.0 / motion_fps
    s  = float(speed_scale)

    sp           = obj_spawn.astype(np.float64)
    lifted_spawn = sp.copy()
    lifted_spawn[2] += lift_height

    # Hover point: above box XY at clearance height
    hover_z    = float(waypoint_z) + 0.15
    hover_pt   = np.array([float(box_xy[0]), float(box_xy[1]), hover_z],  dtype=np.float64)
    # Release point: box XY at waypoint_z
    release_pt = np.array([float(box_xy[0]), float(box_xy[1]), float(waypoint_z)], dtype=np.float64)

    print(f"  Box XY: {box_xy.round(3)}")
    print(f"  Waypoint z={waypoint_z:.3f}  hover_z={hover_z:.3f}")
    print(f"  Hover point  (world): {hover_pt.round(3)}")
    print(f"  Release point(world): {release_pt.round(3)}")

    out: list = []

    # 1. Hold at spawn
    hold_frames = max(0, int(round(hold_sec / dt)))
    out.extend([tuple(sp.tolist())] * hold_frames)

    # 2. Lift vertically
    out.extend(_trap_segment_3d(sp, lifted_spawn, v_max=0.3 * s, a_max=0.6 * s, dt=dt))

    # 3. Move diagonally to hover point above box
    out.extend(_trap_segment_3d(lifted_spawn, hover_pt, v_max=0.6 * s, a_max=1.2 * s, dt=dt))

    # 4. Descend into box (slower for clean release)
    out.extend(_trap_segment_3d(hover_pt, release_pt, v_max=0.25 * s, a_max=0.5 * s, dt=dt))

    # 5. Hold at release point
    tgt_hold_frames = max(0, int(round(tgt_hold_sec / dt)))
    if out:
        out.extend([out[-1]] * tgt_hold_frames)

    T_new        = len(out)
    new_obj_pos  = np.array(out, dtype=np.float32)

    if obj_rot0_wxyz is None:
        obj_rot0_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    q0_xyzw = np.array([obj_rot0_wxyz[1], obj_rot0_wxyz[2],
                         obj_rot0_wxyz[3], obj_rot0_wxyz[0]], dtype=np.float32)
    new_rot_xyzw = make_obj_rot_intervals(
        T=T_new, dt=dt, q0_xyzw=q0_xyzw,
        rotation_interval_sec=1.5, dtheta_max_rad=0.0,
    )
    new_obj_rot_wxyz = new_rot_xyzw[:, [3, 0, 1, 2]].astype(np.float32)

    print(f"  Generated trajectory: {T_new} frames ({T_new / motion_fps:.2f}s @ {motion_fps}Hz)")

    return {
        "obj_pos":       new_obj_pos,
        "obj_rot":       new_obj_rot_wxyz,
        "motion_length": T_new,
        "motion_fps":    motion_fps,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Contact force helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_body_geom_ids(model: mujoco.MjModel, body_id: int) -> set:
    return {i for i in range(model.ngeom) if model.geom_bodyid[i] == body_id}


def compute_obj_contact_force(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    obj_geom_ids: set,
    robot_geom_ids: set | None = None,
) -> tuple[float, float, int]:
    _cf = np.zeros(6, dtype=np.float64)
    total_N = 0.0
    robot_N = 0.0
    n = 0
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        obj1 = g1 in obj_geom_ids
        obj2 = g2 in obj_geom_ids
        if not (obj1 or obj2):
            continue
        mujoco.mj_contactForce(model, data, i, _cf)
        fn = abs(float(_cf[0]))
        total_N += fn
        n += 1
        if robot_geom_ids is not None:
            other = g2 if obj1 else g1
            if other in robot_geom_ids:
                robot_N += fn
    return total_N, robot_N, n


# ═══════════════════════════════════════════════════════════════════════════
# Scene helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_freejoint_qpos_addr(model: mujoco.MjModel, body_name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise KeyError(f"Body '{body_name}' not found.")
    jnt_adr = int(model.body_jntadr[body_id])
    if jnt_adr < 0:
        raise RuntimeError(f"Body '{body_name}' has no joints.")
    return int(model.jnt_qposadr[jnt_adr])


def get_freejoint_qvel_addr(model: mujoco.MjModel, body_name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise KeyError(f"Body '{body_name}' not found.")
    jnt_adr = int(model.body_jntadr[body_id])
    if jnt_adr < 0:
        raise RuntimeError(f"Body '{body_name}' has no joints.")
    return int(model.jnt_dofadr[jnt_adr])


def set_table_pose(model: mujoco.MjModel, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    if body_id < 0:
        raise KeyError("Body 'table' not found.")
    model.body_pos[body_id]  = pos.astype(np.float64)
    model.body_quat[body_id] = quat_wxyz.astype(np.float64)


def set_freejoint_pose(
    data: mujoco.MjData, qpos_addr: int,
    pos: np.ndarray, quat_wxyz: np.ndarray,
) -> None:
    data.qpos[qpos_addr:qpos_addr + 3]     = pos
    data.qpos[qpos_addr + 3:qpos_addr + 7] = quat_wxyz


# ═══════════════════════════════════════════════════════════════════════════
# Math helpers for OTT observation
# ═══════════════════════════════════════════════════════════════════════════

def quat_to_rot_mat(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz.astype(np.float64)
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ], dtype=np.float32)


def yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float32)


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1.astype(np.float32)
    w2, x2, y2, z2 = q2.astype(np.float32)
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)


def heading_quat(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    fwd_x = 1.0 - 2.0 * (y*y + z*z)
    fwd_y = 2.0 * (x*y + w*z)
    yaw   = np.arctan2(fwd_y, fwd_x)
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float32)


def rot_mat_to_6d(R: np.ndarray) -> np.ndarray:
    return R[:, :2].flatten().astype(np.float32)


def obj_pos_rel_heading(obj_world, root_world, hquat):
    return quat_rotate_inverse(hquat, obj_world - root_world)


def obj_rot_rel_heading(obj_quat_wxyz, hquat):
    return rot_mat_to_6d(quat_to_rot_mat(hquat).T @ quat_to_rot_mat(obj_quat_wxyz))


def trap_alpha(t: float, T: float, blend: float = 0.3) -> float:
    t   = float(np.clip(t, 0.0, T))
    t_a = blend * T
    v_p = 1.0 / (T - t_a)
    if t <= t_a:
        return 0.5 * (v_p / t_a) * t * t
    elif t <= T - t_a:
        return 0.5 * v_p * t_a + v_p * (t - t_a)
    else:
        t_rem = T - t
        return 1.0 - 0.5 * (v_p / t_a) * t_rem * t_rem


# ═══════════════════════════════════════════════════════════════════════════
# Observation history buffer
# ═══════════════════════════════════════════════════════════════════════════

class OTTObsBuffer:
    HIST = 5

    def __init__(self):
        self.obj_pos     = np.zeros((self.HIST,  3), dtype=np.float32)
        self.obj_rot_6d  = np.zeros((self.HIST,  6), dtype=np.float32)
        self.hand_jpos   = np.zeros((self.HIST, 14), dtype=np.float32)
        self.hand_jvel   = np.zeros((self.HIST, 14), dtype=np.float32)
        self.body_jpos   = np.zeros((self.HIST, 29), dtype=np.float32)
        self.body_jvel   = np.zeros((self.HIST, 29), dtype=np.float32)
        self.body_angvel = np.zeros((self.HIST,  3), dtype=np.float32)
        self.body_pgrav  = np.zeros((self.HIST,  3), dtype=np.float32)

    def push(self, obj_pos, obj_rot_6d, hand_jpos, hand_jvel,
             body_jpos, body_jvel, body_angvel, body_pgrav):
        for buf, val in (
            (self.obj_pos,     obj_pos),
            (self.obj_rot_6d,  obj_rot_6d),
            (self.hand_jpos,   hand_jpos),
            (self.hand_jvel,   hand_jvel),
            (self.body_jpos,   body_jpos),
            (self.body_jvel,   body_jvel),
            (self.body_angvel, body_angvel),
            (self.body_pgrav,  body_pgrav),
        ):
            buf[:-1] = buf[1:]
            buf[-1]  = val

    def reset(self):
        for buf in (self.obj_pos, self.obj_rot_6d,
                    self.hand_jpos, self.hand_jvel,
                    self.body_jpos, self.body_jvel,
                    self.body_angvel, self.body_pgrav):
            buf[:] = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Observation builders
# ═══════════════════════════════════════════════════════════════════════════

def build_ott_obs(
    data, dof_qpos_addrs, dof_qvel_addrs,
    robot_qpos_addr, obj_qpos_addr, torso_body_id,
    motion, motion_timestep, obs_buf, obj_bps,
) -> np.ndarray:
    motion_length = motion["motion_length"]

    joint_pos    = data.qpos[dof_qpos_addrs].astype(np.float32)
    joint_vel    = data.qvel[dof_qvel_addrs].astype(np.float32)
    base_quat    = data.qpos[robot_qpos_addr + 3:robot_qpos_addr + 7].astype(np.float32)
    base_ang_vel = data.qvel[robot_qpos_addr + 3:robot_qpos_addr + 6].astype(np.float32)

    torso_pos  = data.xpos[torso_body_id].astype(np.float32)
    torso_quat = data.xquat[torso_body_id].astype(np.float32)
    hquat      = heading_quat(torso_quat)

    proj_grav = quat_rotate_inverse(base_quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))

    obj_pos_world  = data.qpos[obj_qpos_addr:obj_qpos_addr + 3].astype(np.float32)
    obj_quat_world = data.qpos[obj_qpos_addr + 3:obj_qpos_addr + 7].astype(np.float32)

    cur_obj_pos    = obj_pos_rel_heading(obj_pos_world, torso_pos, hquat)
    cur_obj_rot_6d = obj_rot_rel_heading(obj_quat_world, hquat)

    joint_pos_rel = joint_pos - DEFAULT_DOF_ANGLES
    obs_buf.push(
        cur_obj_pos, cur_obj_rot_6d,
        joint_pos_rel[HAND_INDICES], joint_vel[HAND_INDICES],
        joint_pos_rel[BODY_INDICES], joint_vel[BODY_INDICES],
        base_ang_vel, proj_grav,
    )

    short_idx = np.clip(motion_timestep + np.arange(1, 11), 0, motion_length - 1)
    mo_pos = np.array([obj_pos_rel_heading(motion["obj_pos"][i], torso_pos, hquat)
                       for i in short_idx])
    mo_ori = np.array([obj_rot_rel_heading(motion["obj_rot"][i], hquat)
                       for i in short_idx])

    long_idx = np.clip(motion_timestep + np.array([20, 40, 60, 80, 100]), 0, motion_length - 1)
    mo_pos_long = np.array([obj_pos_rel_heading(motion["obj_pos"][i], torso_pos, hquat)
                            for i in long_idx])
    mo_ori_long = np.array([obj_rot_rel_heading(motion["obj_rot"][i], hquat)
                            for i in long_idx])

    obs = np.concatenate([
        obs_buf.obj_pos.flatten(),     obs_buf.obj_rot_6d.flatten(),
        obs_buf.hand_jpos.flatten(),   obs_buf.hand_jvel.flatten(),
        obs_buf.body_jpos.flatten(),   obs_buf.body_jvel.flatten(),
        obs_buf.body_angvel.flatten(), obs_buf.body_pgrav.flatten(),
        obj_bps,
        mo_pos.flatten(), mo_ori.flatten(),
        mo_pos_long.flatten(), mo_ori_long.flatten(),
    ])
    return obs.reshape(1, -1).astype(np.float32)


def build_stabilization_obs(
    joint_pos, joint_vel, base_quat_wxyz, base_ang_vel,
    last_action_29, motion_cmd_seq, obs_idx,
) -> np.ndarray:
    dof_pos_rel = joint_pos[obs_idx] - DEFAULT_DOF_ANGLES[obs_idx]
    proj_grav   = quat_rotate_inverse(base_quat_wxyz,
                                      np.array([0.0, 0.0, -1.0], dtype=np.float32))
    obs = np.concatenate([dof_pos_rel, joint_vel[obs_idx], base_ang_vel,
                          proj_grav, last_action_29, motion_cmd_seq.flatten()])
    return obs.reshape(1, -1).astype(np.float32)


def build_locomotion_obs(
    data: mujoco.MjData,
    dof_qpos_addrs: np.ndarray,
    dof_qvel_addrs: np.ndarray,
    robot_qpos_addr: int,
    robot_qvel_addr: int,
    vel_command: np.ndarray,
    last_raw_action: np.ndarray,
) -> np.ndarray:
    """Build 138D locomotion observation, matching test_loco_mw.py."""
    joint_pos = data.qpos[dof_qpos_addrs].astype(np.float32)
    joint_vel = data.qvel[dof_qvel_addrs].astype(np.float32)
    base_quat_wxyz = data.qpos[robot_qpos_addr + 3:robot_qpos_addr + 7].astype(np.float32)
    base_ang_vel = data.qvel[robot_qvel_addr + 3:robot_qvel_addr + 6].astype(np.float32)
    joint_pos_rel = joint_pos - DEFAULT_DOF_ANGLES
    body_pos = joint_pos_rel[BODY_INDICES]
    body_vel = joint_vel[BODY_INDICES]
    proj_grav = quat_rotate_inverse(base_quat_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float32))
    proprio_body = np.concatenate([body_pos, body_vel, base_ang_vel, proj_grav], axis=0)
    hand_pos = joint_pos_rel[HAND_INDICES]
    hand_vel = joint_vel[HAND_INDICES]
    proprio_hand = np.concatenate([hand_pos, hand_vel], axis=0)
    obs = np.concatenate([
        vel_command.astype(np.float32),
        proprio_body.astype(np.float32),
        proprio_hand.astype(np.float32),
        last_raw_action.astype(np.float32),
    ], axis=0)
    return obs.reshape(1, -1).astype(np.float32)


def wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def yaw_from_quat_wxyz(q_wxyz: np.ndarray) -> float:
    w, x, y, z = [float(v) for v in q_wxyz]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def world_to_local_xy(v_world_xy: np.ndarray, yaw: float) -> np.ndarray:
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return np.array([
        c * float(v_world_xy[0]) + s * float(v_world_xy[1]),
        -s * float(v_world_xy[0]) + c * float(v_world_xy[1]),
    ], dtype=np.float32)


def select_table_waypoint_xy(
    robot_xy: np.ndarray,
    table_xy: np.ndarray,
    table_quat_wxyz: np.ndarray,
    desired_dist: float,
) -> np.ndarray:
    """Pick nearest among 0/90/180/270deg table-frame points at desired_dist."""
    table_yaw = yaw_from_quat_wxyz(table_quat_wxyz)
    x_axis = np.array([np.cos(table_yaw), np.sin(table_yaw)], dtype=np.float32)
    y_axis = np.array([-np.sin(table_yaw), np.cos(table_yaw)], dtype=np.float32)
    candidates = np.stack([
        table_xy + desired_dist * x_axis,
        table_xy + desired_dist * y_axis,
        table_xy - desired_dist * x_axis,
        table_xy - desired_dist * y_axis,
    ], axis=0).astype(np.float32)
    d = np.linalg.norm(candidates - robot_xy.reshape(1, 2), axis=1)
    return candidates[int(np.argmin(d))]


def build_locomotion_velocity_command(
    robot_xy: np.ndarray,
    robot_yaw: float,
    waypoint_xy: np.ndarray,
    table_xy: np.ndarray,
    dt: float,
    integ_xy: np.ndarray,
    prev_err_xy: np.ndarray,
    integ_yaw: float,
    prev_err_yaw: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Generate local-frame [vx, vy, yaw_rate] command via PID and update PID state."""
    pos_err_world = waypoint_xy - robot_xy
    pos_err_local = world_to_local_xy(pos_err_world, robot_yaw)
    desired_yaw = float(np.arctan2(float(table_xy[1] - robot_xy[1]),
                                   float(table_xy[0] - robot_xy[0])))
    yaw_err = wrap_to_pi(desired_yaw - robot_yaw)

    integ_xy = np.clip(integ_xy + pos_err_local * dt, -LOCO_INTEGRAL_CLIP, LOCO_INTEGRAL_CLIP)
    d_err_xy = (pos_err_local - prev_err_xy) / max(dt, 1e-6)
    cmd_xy = (LOCO_KP_XY * pos_err_local) + (LOCO_KI_XY * integ_xy) + (LOCO_KD_XY * d_err_xy)

    integ_yaw = float(np.clip(integ_yaw + yaw_err * dt, -LOCO_INTEGRAL_CLIP, LOCO_INTEGRAL_CLIP))
    d_err_yaw = (yaw_err - prev_err_yaw) / max(dt, 1e-6)
    cmd_yaw = (LOCO_KP_YAW * yaw_err) + (LOCO_KI_YAW * integ_yaw) + (LOCO_KD_YAW * d_err_yaw)

    vel_cmd = np.array([
        float(np.clip(cmd_xy[0], -1.0, 1.0)),
        float(np.clip(cmd_xy[1], -1.0, 1.0)),
        float(np.clip(cmd_yaw, -1.0, 1.0)),
    ], dtype=np.float32)
    return vel_cmd, integ_xy, pos_err_local, integ_yaw, yaw_err


# ═══════════════════════════════════════════════════════════════════════════
# Main simulation
# ═══════════════════════════════════════════════════════════════════════════

def run(args):
    root = Path(__file__).parent

    # ── Robot init state ───────────────────────────────────────────────────
    root_pos_init  = np.asarray(args.robot_root_pos, dtype=np.float32)
    root_quat_xyzw = np.asarray(args.robot_root_quat_xyzw, dtype=np.float32)
    root_quat_wxyz = np.array([root_quat_xyzw[3], root_quat_xyzw[0],
                                root_quat_xyzw[1], root_quat_xyzw[2]], dtype=np.float32)
    dof_pos_43 = np.asarray(args.robot_dof_pos, dtype=np.float32)
    if dof_pos_43.shape != (43,):
        raise ValueError(f"--robot-dof-pos must have 43 values, got {dof_pos_43.shape}")
    init_q = build_init_q(dof_pos_43)

    # ── Resolve XML ────────────────────────────────────────────────────────
    xml_dir  = root / "src/holosoma/holosoma/data/robots/g1/g1_object"
    xml_path = str(xml_dir / f"g1_43dof_{args.obj_name}.xml")
    if not Path(xml_path).exists():
        raise FileNotFoundError(f"No XML for obj_name='{args.obj_name}': {xml_path}")

    box_xml_path = str(Path(args.box_xml).expanduser())
    if not Path(box_xml_path).is_absolute():
        box_xml_path = str(root / args.box_xml)
    if not Path(box_xml_path).exists():
        raise FileNotFoundError(f"Box XML not found: {box_xml_path}")

    object_candidates = list(dict.fromkeys([args.obj_name, *OBJ_RESPAWN_CANDIDATES]))
    print(f"Robot XML: {xml_path}")
    print(f"Box XML  : {box_xml_path}")
    print(f"Object candidates: {object_candidates}")

    # ── Spawn positions ────────────────────────────────────────────────────
    OBJ_SPAWN   = OBJ_SPAWN_DEFAULT.copy()
    TABLE_SPAWN = TABLE_SPAWN_DEFAULT.copy()

    # ── Box position & waypoint height ────────────────────────────────────
    box_xy = np.asarray(args.box_pos[:2], dtype=np.float32)

    if args.waypoint_z is not None:
        waypoint_z = float(args.waypoint_z)
    else:
        lo, hi = float(WAYPOINT_Z_RANGE[0]), float(WAYPOINT_Z_RANGE[1])
        waypoint_z = float(np.random.uniform(lo, hi))

    # box body z: opening will be at (body_z + BOX_HEIGHT) which is slightly
    # above waypoint_z so the object descends into the box at release.
    box_body_z = waypoint_z - BOX_BODY_Z_OFFSET
    box_pos    = np.array([box_xy[0], box_xy[1], box_body_z], dtype=np.float32)
    print(f"Waypoint z: {waypoint_z:.3f}  →  box body z={box_body_z:.3f}  "
          f"(opening at ~{box_body_z + BOX_HEIGHT:.3f})")

    def _sample_table_opposite_box(cur_box_xy: np.ndarray) -> np.ndarray:
        """Sample table XY 1.0–2.0 m from robot, 90–270° from bin direction.

        The bin direction is the angle from robot (origin) to the box.
        90–270° from that puts the table in the opposite hemisphere.
        """
        theta_bin   = float(np.arctan2(float(cur_box_xy[1]), float(cur_box_xy[0])))
        r_table     = float(np.random.uniform(1.0, 2.0))
        theta_table = theta_bin + np.pi + float(np.random.uniform(-np.pi / 2, np.pi / 2))
        return np.array([r_table * np.cos(theta_table),
                         r_table * np.sin(theta_table)], dtype=np.float32)

    def _sample_box_spawn_randomized() -> tuple[np.ndarray, np.ndarray, float, float, float]:
        """Sample box spawn using existing cabinet randomization rule."""
        r = float(np.random.uniform(0.5, 2.0))
        theta = float(np.random.uniform(0.0, np.pi))
        new_box_xy = np.array([r * np.cos(theta), r * np.sin(theta)], dtype=np.float32)
        new_waypoint_z = float(np.random.uniform(*WAYPOINT_Z_RANGE))
        new_box_body_z = new_waypoint_z - BOX_BODY_Z_OFFSET
        new_box_pos = np.array([new_box_xy[0], new_box_xy[1], new_box_body_z], dtype=np.float32)
        return new_box_pos, new_box_xy, new_waypoint_z, r, theta

    # ── Box position randomization (same scheme as cabinet) ───────────────
    if args.randomize:
        box_pos, box_xy, waypoint_z, r, theta = _sample_box_spawn_randomized()
        box_body_z = float(box_pos[2])
        print(f"Randomized box: r={r:.3f}  theta={np.degrees(theta):.1f}°  "
              f"pos={list(np.round(box_pos, 3))}  waypoint_z={waypoint_z:.3f}")
        print(f"Spawn table XY fixed: {TABLE_SPAWN[:2].round(3)}")

    # ── Generate placement trajectory ──────────────────────────────────────
    motion_fps = float(args.motion_fps)
    print(f"Generating box trajectory @ {motion_fps}Hz …")
    motion = generate_box_trajectory(
        obj_spawn    = OBJ_SPAWN,
        box_xy       = box_xy,
        waypoint_z   = waypoint_z,
        motion_fps   = motion_fps,
        hold_sec     = args.hold_sec,
        tgt_hold_sec = args.tgt_hold_sec,
        obj_rot0_wxyz= OBJ_QUAT_WXYZ,
        speed_scale  = args.traj_speed_scale,
    )
    motion_length = motion["motion_length"]

    _viz_sample_step = max(1, int(0.5 * motion_fps))
    _viz_full_traj   = list(motion["obj_pos"][::_viz_sample_step])

    # ── Camera placement ───────────────────────────────────────────────────
    # Place camera on the side perpendicular to robot→box axis
    _box_to_origin = -box_pos[:2]
    _dist = float(np.linalg.norm(_box_to_origin))
    if _dist > 1e-6:
        _perp = np.array([-_box_to_origin[1], _box_to_origin[0]]) / _dist
    else:
        _perp = np.array([1.0, 0.0])
    _cam_xy     = box_pos[:2] + 2.0 * _perp
    _cam_pos    = np.array([_cam_xy[0], _cam_xy[1], float(args.camera_pos[2])], dtype=np.float32)
    _cam_target = np.array([float(box_pos[0]), float(box_pos[1]),
                             waypoint_z * 0.5], dtype=np.float32)
    print(f"Camera: pos={list(np.round(_cam_pos, 3))}  target={list(np.round(_cam_target, 3))}")

    # ── Load policy ────────────────────────────────────────────────────────
    policy_path = args.policy or str(
        root / "src/holosoma_inference/holosoma_inference/models/wbt/object/bps_policy.onnx"
    )
    print(f"Loading OTT policy: {policy_path}")
    session = onnxruntime.InferenceSession(policy_path)
    inp = session.get_inputs()[0]
    out_node = session.get_outputs()[0]
    print(f"  Input:  {inp.name} {inp.shape}")
    print(f"  Output: {out_node.name} {out_node.shape}")

    # ── Load MuJoCo model (robot + box) ───────────────────────────────────
    print(f"Building combined model (robot + box at {list(np.round(box_pos, 3))}) …")
    model = build_model_with_box(
        xml_path,
        box_xml_path,
        box_pos,
        object_candidates=object_candidates,
        base_obj_name=args.obj_name,
    )
    data  = mujoco.MjData(model)
    model.opt.timestep = 1.0 / args.sim_hz
    sim_dt           = float(model.opt.timestep)
    steps_per_policy = args.sim_hz // args.policy_hz
    policy_dt        = sim_dt * steps_per_policy
    print(f"Model: {model.nq} qpos  {model.nv} qvel  {model.nu} actuators  dt={sim_dt}s")
    print(f"Physics {1/sim_dt:.0f}Hz | Policy {1/policy_dt:.0f}Hz ({steps_per_policy} substeps)")

    dof_qpos_addrs, dof_qvel_addrs = mj_hinge_addrs(model, DOF_NAMES)
    actuator_ids    = mj_actuator_ids(model, DOF_NAMES)
    robot_qpos_addr = get_freejoint_qpos_addr(model, "pelvis")
    robot_qvel_addr = get_freejoint_qvel_addr(model, "pelvis")
    torso_body_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    if torso_body_id < 0:
        raise KeyError("Body 'torso_link' not found.")
    act_idx = name_indices(ACTION_ORDER_43DOF, DOF_NAMES)

    object_states: dict[str, dict[str, object]] = {}
    for obj_name in object_candidates:
        body_name = f"object_{obj_name}"
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            continue
        object_states[obj_name] = {
            "body_name": body_name,
            "body_id": body_id,
            "qpos_addr": get_freejoint_qpos_addr(model, body_name),
            "qvel_addr": get_freejoint_qvel_addr(model, body_name),
            "geom_ids": get_body_geom_ids(model, body_id),
        }
    if args.obj_name not in object_states:
        raise KeyError(
            f"Active object body 'object_{args.obj_name}' not found in model. "
            f"available={sorted(object_states.keys())}"
        )
    active_obj_name = args.obj_name
    obj_qpos_addr = int(object_states[active_obj_name]["qpos_addr"])
    obj_qvel_addr = int(object_states[active_obj_name]["qvel_addr"])
    _obj_body_id = int(object_states[active_obj_name]["body_id"])
    _obj_geom_ids = set(object_states[active_obj_name]["geom_ids"])  # type: ignore[arg-type]
    print(f"Object bodies loaded: {sorted(object_states.keys())} | active='{active_obj_name}'")
    available_obj_candidates = [n for n in OBJ_RESPAWN_CANDIDATES if n in object_states]
    if not available_obj_candidates:
        available_obj_candidates = [args.obj_name]
    geom_render_contact_defaults = {
        gid: (
            model.geom_rgba[gid].copy(),
            int(model.geom_contype[gid]),
            int(model.geom_conaffinity[gid]),
        )
        for state in object_states.values()
        for gid in state["geom_ids"]  # type: ignore[index]
    }

    # ── Contact force monitoring ───────────────────────────────────────────
    # Exclude world, object, table, and full box subtree from robot geoms
    _exclude_body_ids = {0}
    _exclude_body_ids.update(int(s["body_id"]) for s in object_states.values())
    table_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    if table_body_id < 0:
        raise KeyError("Body 'table' not found.")
    _exclude_body_ids.add(table_body_id)
    _box_root_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box_root")
    if _box_root_bid >= 0:
        _queue = [_box_root_bid]
        while _queue:
            _b = _queue.pop()
            _exclude_body_ids.add(_b)
            for _c in range(model.nbody):
                if int(model.body_parentid[_c]) == _b and _c not in _exclude_body_ids:
                    _queue.append(_c)

    def _set_box_root_pose(new_box_pos: np.ndarray) -> None:
        if _box_root_bid < 0:
            return
        model.body_pos[_box_root_bid] = new_box_pos.astype(np.float64)
        # 런타임에서는 model body pose 변경 후 forward만 수행.
        mujoco.mj_forward(model, data)

    def _set_table_pose_runtime(new_table_pos: np.ndarray, new_table_quat_wxyz: np.ndarray) -> None:
        model.body_pos[table_body_id] = new_table_pos.astype(np.float64)
        model.body_quat[table_body_id] = new_table_quat_wxyz.astype(np.float64)
        # 런타임에서는 model body pose 변경 후 forward만 수행.
        mujoco.mj_forward(model, data)

    _robot_geom_ids = {i for i in range(model.ngeom)
                       if model.geom_bodyid[i] not in _exclude_body_ids}
    print(f"Contact monitoring: obj_geoms={len(_obj_geom_ids)}  "
          f"robot_geoms={len(_robot_geom_ids)}")

    # ── Master policy setup (always loaded; used for hand-open/post-hold/return-to-default) ──
    stabilize_duration = float(args.stabilize_sec)
    stabilize_enabled  = stabilize_duration > 0.0

    stab_policy_path = args.stabilize_policy or str(
        root / "src/holosoma_inference/holosoma_inference/models/wbt/base/master_policy.onnx"
    )
    print(f"Loading master policy: {stab_policy_path}")
    stabilize_session    = onnxruntime.InferenceSession(stab_policy_path)
    stabilize_input_name = stabilize_session.get_inputs()[0].name

    stabilize_obs_idx = name_indices(STABILIZATION_OBS_ORDER_29DOF, DOF_NAMES)
    stabilize_act_idx = name_indices(STABILIZATION_ACTION_ORDER_29DOF, DOF_NAMES)
    stab_in_43 = name_indices(STABILIZATION_ACTION_ORDER_29DOF, ACTION_ORDER_43DOF)
    stabilize_act_scale_29    = ACTION_SCALE[stab_in_43]
    stabilize_act_offset_29   = ACTION_OFFSET[stab_in_43]
    stabilize_act_clip_min_29 = ACTION_CLIP_MIN[stab_in_43]
    stabilize_act_clip_max_29 = ACTION_CLIP_MAX[stab_in_43]
    stabilize_hand_idx        = name_indices(HAND_JOINT_NAMES, DOF_NAMES)

    stabilize_motion_dof_pos_29, _, stabilize_hand_target_q = \
        build_stabilization_refs(dof_pos_43, motion_length)
    print(f"  Phase-1 stabilization: {'enabled' if stabilize_enabled else 'disabled'}  "
          f"({stabilize_duration:.1f}s)")

    loco_policy_path = args.loco_policy or str(
        root / "src/holosoma_inference/holosoma_inference/models/loco/g1_43dof/walk_prior_dr_0306.onnx"
    )
    print(f"Loading locomotion policy: {loco_policy_path}")
    loco_session = onnxruntime.InferenceSession(loco_policy_path)
    loco_input_name = loco_session.get_inputs()[0].name

    # ── Load object BPS codes (for runtime object switching) ───────────────
    bps_dir = root / "src/holosoma/holosoma/data/objects_new/objects_new"
    obj_bps_map: dict[str, np.ndarray] = {}
    for obj_name in object_candidates:
        bps_path = bps_dir / obj_name / f"{obj_name}_bps.pkl"
        if not bps_path.exists():
            print(f"[WARN] BPS file missing for '{obj_name}': {bps_path}")
            continue
        bps_data = joblib.load(str(bps_path))
        obj_bps_code = bps_data["bps_code"]
        if hasattr(obj_bps_code, "numpy"):
            obj_bps_code = obj_bps_code.numpy()
        obj_bps_map[obj_name] = obj_bps_code.flatten().astype(np.float32)
    if args.obj_name not in obj_bps_map:
        raise FileNotFoundError(
            f"BPS for base object '{args.obj_name}' not found. loaded={sorted(obj_bps_map.keys())}"
        )
    obj_bps = obj_bps_map[active_obj_name].copy()
    print(f"Loaded BPS codes for: {sorted(obj_bps_map.keys())} | active='{active_obj_name}'")

    obs_buf = OTTObsBuffer()

    def _set_active_object_pose(selected_obj_name: str, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
        nonlocal active_obj_name, obj_qpos_addr, obj_qvel_addr, _obj_body_id, _obj_geom_ids, obj_bps
        if selected_obj_name not in object_states:
            raise KeyError(f"Unknown object candidate '{selected_obj_name}'")
        for hidden_idx, (obj_name, state) in enumerate(object_states.items()):
            qpos_addr_i = int(state["qpos_addr"])
            qvel_addr_i = int(state["qvel_addr"])
            if obj_name == selected_obj_name:
                set_freejoint_pose(data, qpos_addr_i, pos, quat_wxyz)
            else:
                hidden_pos = np.array([
                    INACTIVE_OBJECT_FAR_BASE_X + hidden_idx * INACTIVE_OBJECT_FAR_STEP_X,
                    INACTIVE_OBJECT_FAR_BASE_Y,
                    INACTIVE_OBJECT_FAR_Z,
                ], dtype=np.float32)
                set_freejoint_pose(data, qpos_addr_i, hidden_pos, OBJ_QUAT_WXYZ)
            for gid in state["geom_ids"]:  # type: ignore[index]
                rgba0, contype0, conaff0 = geom_render_contact_defaults[int(gid)]
                if obj_name == selected_obj_name:
                    model.geom_rgba[int(gid)] = rgba0
                    model.geom_contype[int(gid)] = contype0
                    model.geom_conaffinity[int(gid)] = conaff0
                else:
                    model.geom_rgba[int(gid)] = rgba0
                    model.geom_rgba[int(gid), 3] = 0.0
                    model.geom_contype[int(gid)] = 0
                    model.geom_conaffinity[int(gid)] = 0
            data.qvel[qvel_addr_i:qvel_addr_i + 6] = 0.0
        active_obj_name = selected_obj_name
        obj_qpos_addr = int(object_states[active_obj_name]["qpos_addr"])
        obj_qvel_addr = int(object_states[active_obj_name]["qvel_addr"])
        _obj_body_id = int(object_states[active_obj_name]["body_id"])
        _obj_geom_ids = set(object_states[active_obj_name]["geom_ids"])  # type: ignore[arg-type]
        if active_obj_name in obj_bps_map:
            obj_bps = obj_bps_map[active_obj_name].copy()
        else:
            obj_bps = obj_bps_map[args.obj_name].copy()
        mujoco.mj_forward(model, data)
        print(
            f"  [Object active] '{active_obj_name}' qpos_addr={obj_qpos_addr}",
            flush=True,
        )

    # ── Initialize scene ───────────────────────────────────────────────────
    def _reset_scene():
        nonlocal waypoint_z, box_xy, box_pos, box_body_z

        if args.randomize:
            # Robot: ±0.1 m xy, ±0.5 rad yaw
            robot_pos = root_pos_init.copy()
            robot_pos[:2] += np.random.uniform(-0.1, 0.1, 2).astype(np.float32)
            robot_yaw_noise = float(np.random.uniform(-0.5, 0.5))
            robot_quat = quat_mul_wxyz(yaw_to_quat_wxyz(robot_yaw_noise), root_quat_wxyz)
            # Joints: ±0.1 rad
            joint_q = init_q + np.random.uniform(-0.1, 0.1, init_q.shape).astype(np.float32)
            # Table: ±0.2 m z
            table_z_noise = float(np.random.uniform(-0.2, 0.2))
            table_pos = TABLE_SPAWN.copy()
            table_pos[2] += table_z_noise
            # Object: ±0.1 m xy, yaw ±180°, z tracks table height
            obj_pos = OBJ_SPAWN.copy()
            obj_pos[:2] += np.random.uniform(-0.1, 0.1, 2).astype(np.float32)
            obj_pos[2]  += table_z_noise
            obj_yaw  = float(np.random.uniform(-np.pi, np.pi))
            obj_quat = yaw_to_quat_wxyz(obj_yaw)
            selected_obj = str(np.random.choice(available_obj_candidates))
            print(
                f"  [Spawn rand] robot_xy={robot_pos[:2].round(3)}  "
                f"robot_yaw={np.degrees(robot_yaw_noise):+.1f}°  "
                f"table_z={table_pos[2]:.3f} (Δ{table_z_noise:+.3f})  "
                f"obj='{selected_obj}' obj_xy={obj_pos[:2].round(3)}  obj_yaw={np.degrees(obj_yaw):+.1f}°",
                flush=True,
            )
        else:
            robot_pos  = root_pos_init
            robot_quat = root_quat_wxyz
            joint_q    = init_q
            table_pos  = TABLE_SPAWN
            obj_pos    = OBJ_SPAWN
            obj_quat   = OBJ_QUAT_WXYZ
            selected_obj = args.obj_name

        _set_table_pose_runtime(table_pos, TABLE_QUAT_WXYZ)
        data.qpos[robot_qpos_addr:robot_qpos_addr + 3]     = robot_pos
        data.qpos[robot_qpos_addr + 3:robot_qpos_addr + 7] = robot_quat
        data.qpos[dof_qpos_addrs] = joint_q
        data.qvel[:] = 0.0
        _set_active_object_pose(selected_obj, obj_pos, obj_quat)
        return obj_pos

    def _spawn_object_on_table(table_pos: np.ndarray, choose_candidate: bool = True) -> str:
        """Spawn a fresh object instance on table with the same randomization rule."""
        selected_obj = active_obj_name
        respawn_candidates = available_obj_candidates
        if choose_candidate:
            selected_obj = str(np.random.choice(respawn_candidates))

        table_z_noise = float(table_pos[2] - TABLE_SPAWN[2])
        obj_pos = OBJ_SPAWN.copy()
        obj_pos[:2] = (table_pos[:2] + OBJ_TABLE_XY_OFFSET_DEFAULT
                       + np.random.uniform(-0.1, 0.1, 2).astype(np.float32))
        obj_pos[2] = float(OBJ_SPAWN[2] + table_z_noise)
        obj_yaw = float(np.random.uniform(-np.pi, np.pi))
        obj_quat = yaw_to_quat_wxyz(obj_yaw)

        _set_active_object_pose(selected_obj, obj_pos, obj_quat)
        print(
            f"  [Object respawn] active='{selected_obj}' xy={obj_pos[:2].round(3)} "
            f"z={obj_pos[2]:.3f} yaw={np.degrees(obj_yaw):+.1f}°",
            flush=True,
        )
        return selected_obj

    _first_obj_spawn = _reset_scene()

    # Regenerate trajectory from actual (possibly randomized) object spawn
    if args.randomize:
        motion = generate_box_trajectory(
            obj_spawn    = _first_obj_spawn,
            box_xy       = box_xy,
            waypoint_z   = waypoint_z,
            motion_fps   = motion_fps,
            hold_sec     = args.hold_sec,
            tgt_hold_sec = args.tgt_hold_sec,
            obj_rot0_wxyz= OBJ_QUAT_WXYZ,
            speed_scale  = args.traj_speed_scale,
        )
        motion_length = motion["motion_length"]
        _viz_full_traj[:] = list(motion["obj_pos"][::_viz_sample_step])
        stabilize_motion_dof_pos_29, _, stabilize_hand_target_q = \
            build_stabilization_refs(dof_pos_43, motion_length)
    print(f"Scene initialized: obj={OBJ_SPAWN}  table={TABLE_SPAWN}  box={list(np.round(box_pos, 3))}")

    # ── View-only mode ─────────────────────────────────────────────────────
    if getattr(args, "view_only", False):
        print("View-only mode: opening interactive viewer. Press ESC/Q to close.")
        import mujoco.viewer as _mj_viewer
        mujoco.mj_forward(model, data)
        _mj_viewer.launch(model, data)
        return

    # ── Video recorder ─────────────────────────────────────────────────────
    video_recorder = None
    if args.record or args.offscreen:
        video_recorder = SimpleVideoRecorder(
            model,
            name="test_repetitive_pnp",
            camera_pos=tuple(_cam_pos.tolist()),
            camera_target=tuple(_cam_target.tolist()),
            width=args.video_width,
            height=args.video_height,
            save_dir=args.video_dir,
            output_format=args.video_format,
        )

    # ── Simulation state ───────────────────────────────────────────────────
    target_q        = data.qpos[dof_qpos_addrs].astype(np.float32).copy()
    policy_step     = 0
    motion_timestep = 0
    motion_time_acc = 0.0
    max_iterations = max(1, int(getattr(args, "iteration", 5)))
    if int(getattr(args, "iteration", 5)) < 1:
        print(f"[WARN] --iteration must be >=1, got {args.iteration}. Using 1.", flush=True)
    completed_iterations = 0
    current_iteration = 1
    stab_cmd_t      = 0
    stab_cmd_acc    = 0.0
    stab_last_a29   = np.zeros(29, dtype=np.float32)
    stabilize_active      = stabilize_enabled
    stabilize_start_sim_t = float(data.time)

    _left_thumb_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_thumb_0_link")
    _right_thumb_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_thumb_0_link")
    _left_hand_names  = [n for n in HAND_JOINT_NAMES if n.startswith("left_")]
    _right_hand_names = [n for n in HAND_JOINT_NAMES if n.startswith("right_")]
    _left_hand_dof_idx  = name_indices(_left_hand_names,  DOF_NAMES)
    _right_hand_dof_idx = name_indices(_right_hand_names, DOF_NAMES)
    _left_hand_open_q   = DEFAULT_DOF_ANGLES[_left_hand_dof_idx].copy()
    _right_hand_open_q  = DEFAULT_DOF_ANGLES[_right_hand_dof_idx].copy()
    _motion_29_dof_idx  = name_indices(G1_MOTION_JOINT_NAMES_29, DOF_NAMES)

    release_with_hotdex      = getattr(args, "release_with_hotdex", False)
    hotdex_when_post_release = getattr(args, "hotdex_when_post_release", False)
    HAND_OPEN_DURATION         = 1.0
    POST_HOLD_DURATION         = 2.0
    RETURN_TO_DEFAULT_DURATION = 5.0
    RETURN_ARM_RAISE_DURATION  = 3.0
    hand_open_active           = False
    post_hold_active           = False
    return_to_default_active   = False
    post_hold_start_sim_t      = 0.0
    hand_open_start_sim_t      = 0.0
    return_to_default_start_t  = 0.0
    hand_open_start_q_7        = np.zeros(7,  dtype=np.float32)
    hand_open_target_q_7       = np.zeros(7,  dtype=np.float32)
    hand_open_active_idx       = _left_hand_dof_idx.copy()
    final_body_cmd_29          = np.zeros(29, dtype=np.float32)
    return_to_default_start_q  = np.zeros(29, dtype=np.float32)  # G1_MOTION_JOINT_NAMES_29 order
    _default_q_29              = DEFAULT_DOF_ANGLES[_motion_29_dof_idx].copy()  # target
    _return_shoulder_roll_idx_29 = name_indices(
        ["left_shoulder_roll_joint", "right_shoulder_roll_joint"],
        G1_MOTION_JOINT_NAMES_29,
    )
    _return_shoulder_roll_raise_target = np.array([1.5, -1.5], dtype=np.float32)
    PRE_HOTDEX_INTERP_DURATION = 2.0
    pre_hotdex_active          = False
    pre_hotdex_start_t         = 0.0
    pre_hotdex_start_body_29   = np.zeros(29, dtype=np.float32)
    pre_hotdex_start_hand_14   = np.zeros(len(HAND_JOINT_NAMES), dtype=np.float32)
    locomotion_active          = False
    loco_waypoint_xy           = np.zeros(2, dtype=np.float32)
    loco_last_raw_action       = np.zeros(43, dtype=np.float32)
    loco_int_xy                = np.zeros(2, dtype=np.float32)
    loco_prev_err_xy           = np.zeros(2, dtype=np.float32)
    loco_int_yaw               = 0.0
    loco_prev_err_yaw          = 0.0
    loco_start_sim_t           = 0.0
    force_stop                 = False

    _viz_short_pos: list = []
    _viz_long_pos:  list = []

    print(f"\nStarting placement inference: {motion_length} frames = "
          f"{motion_length / motion_fps:.2f}s  |  stabilize={stabilize_duration:.1f}s")
    print(f"Cycle plan: {max_iterations} iteration(s). Running iteration {current_iteration}/{max_iterations}.")

    def _restart_hotdex_cycle() -> None:
        nonlocal target_q, motion_timestep, motion_time_acc, policy_step
        nonlocal stab_last_a29, stabilize_active, stabilize_start_sim_t, stab_cmd_t, stab_cmd_acc
        nonlocal hand_open_active, post_hold_active, return_to_default_active, locomotion_active
        nonlocal motion, motion_length, stabilize_motion_dof_pos_29, stabilize_hand_target_q
        nonlocal loco_last_raw_action, loco_int_xy, loco_prev_err_xy, loco_int_yaw, loco_prev_err_yaw
        nonlocal loco_start_sim_t, force_stop
        nonlocal pre_hotdex_active, pre_hotdex_start_t

        policy_step = motion_timestep = 0
        motion_time_acc = stab_cmd_acc = 0.0
        stab_cmd_t               = 0
        hand_open_active         = False
        post_hold_active         = False
        return_to_default_active = False
        locomotion_active        = False
        obs_buf.reset()
        stab_last_a29[:] = 0.0
        loco_last_raw_action[:] = 0.0
        loco_int_xy[:] = 0.0
        loco_prev_err_xy[:] = 0.0
        loco_int_yaw = 0.0
        loco_prev_err_yaw = 0.0
        loco_start_sim_t = 0.0
        force_stop = False
        _viz_short_pos[:] = []
        _viz_long_pos[:] = []

        active_obj_spawn = data.qpos[obj_qpos_addr:obj_qpos_addr + 3].astype(np.float32).copy()
        motion = generate_box_trajectory(
            obj_spawn    = active_obj_spawn,
            box_xy       = box_xy,
            waypoint_z   = waypoint_z,
            motion_fps   = motion_fps,
            hold_sec     = args.hold_sec,
            tgt_hold_sec = args.tgt_hold_sec,
            obj_rot0_wxyz= OBJ_QUAT_WXYZ,
            speed_scale  = args.traj_speed_scale,
        )
        motion_length = motion["motion_length"]
        _viz_full_traj[:] = list(motion["obj_pos"][::_viz_sample_step])
        stabilize_motion_dof_pos_29, _, stabilize_hand_target_q = \
            build_stabilization_refs(dof_pos_43, motion_length)

        target_q = data.qpos[dof_qpos_addrs].astype(np.float32).copy()
        pre_hotdex_start_body_29[:] = target_q[_motion_29_dof_idx]
        pre_hotdex_start_hand_14[:] = target_q[stabilize_hand_idx]
        pre_hotdex_start_t          = float(data.time)
        pre_hotdex_active     = True
        stabilize_active      = False

        print(
            f"  [Cycle] Restart HotDex iteration {current_iteration}/{max_iterations} "
            f"(obj={active_obj_name}, obj_spawn={active_obj_spawn.round(3)}, box_xy={box_xy.round(3)}).",
            flush=True,
        )

    def _on_policy_step():
        nonlocal target_q, motion_time_acc, motion_timestep, policy_step
        nonlocal stabilize_active, stab_last_a29, stab_cmd_t, stab_cmd_acc, stabilize_start_sim_t
        nonlocal hand_open_active, hand_open_start_sim_t
        nonlocal post_hold_active, post_hold_start_sim_t
        nonlocal return_to_default_active, return_to_default_start_t, return_to_default_start_q
        nonlocal hand_open_start_q_7, hand_open_target_q_7, hand_open_active_idx, final_body_cmd_29
        nonlocal locomotion_active, loco_waypoint_xy, loco_last_raw_action
        nonlocal loco_int_xy, loco_prev_err_xy, loco_int_yaw, loco_prev_err_yaw
        nonlocal loco_start_sim_t, force_stop
        nonlocal waypoint_z, box_xy, box_pos, box_body_z
        nonlocal completed_iterations, current_iteration
        nonlocal pre_hotdex_active, pre_hotdex_start_t

        # ── Phase 0: Pre-hotdex interpolation to default pose (via stabilize policy) ──
        if pre_hotdex_active:
            t_elapsed = float(data.time) - pre_hotdex_start_t
            alpha = float(np.clip(t_elapsed / PRE_HOTDEX_INTERP_DURATION, 0.0, 1.0))

            interp_cmd_29 = (pre_hotdex_start_body_29
                             + alpha * (stabilize_motion_dof_pos_29[0] - pre_hotdex_start_body_29))
            cmd_pos = np.tile(interp_cmd_29, (10, 1))
            cmd_vel = np.zeros_like(cmd_pos)
            cmd_seq = np.concatenate([cmd_pos, cmd_vel], axis=1).reshape(1, -1).astype(np.float32)

            jpos = data.qpos[dof_qpos_addrs].astype(np.float32)
            jvel = data.qvel[dof_qvel_addrs].astype(np.float32)
            bq   = data.qpos[robot_qpos_addr + 3:robot_qpos_addr + 7].astype(np.float32)
            bav  = data.qvel[robot_qvel_addr + 3:robot_qvel_addr + 6].astype(np.float32)
            obs_stab = build_stabilization_obs(jpos, jvel, bq, bav,
                                               stab_last_a29, cmd_seq, stabilize_obs_idx)
            raw29 = stabilize_session.run(None, {stabilize_input_name: obs_stab})[0][0]
            stab_last_a29 = raw29.copy()
            scaled29 = np.clip(raw29 * stabilize_act_scale_29 + stabilize_act_offset_29,
                               stabilize_act_clip_min_29, stabilize_act_clip_max_29)
            target_q = jpos.copy()
            target_q[stabilize_act_idx] = scaled29
            target_q[stabilize_hand_idx] = (pre_hotdex_start_hand_14
                                             + alpha * (stabilize_hand_target_q[0]
                                                        - pre_hotdex_start_hand_14))

            if t_elapsed >= PRE_HOTDEX_INTERP_DURATION:
                pre_hotdex_active     = False
                stabilize_active      = stabilize_enabled
                stabilize_start_sim_t = float(data.time)
                print(
                    f"  [Pre-hotdex interp] Done — "
                    f"{'stabilize' if stabilize_enabled else 'OTT inference'} start.",
                    flush=True,
                )
            return

        # ── Phase 1: Stabilization ─────────────────────────────────────────
        if stabilize_active:
            jpos = data.qpos[dof_qpos_addrs].astype(np.float32)
            jvel = data.qvel[dof_qvel_addrs].astype(np.float32)
            bq   = data.qpos[robot_qpos_addr + 3:robot_qpos_addr + 7].astype(np.float32)
            bav  = data.qvel[robot_qvel_addr + 3:robot_qvel_addr + 6].astype(np.float32)

            t_clamp = min(stab_cmd_t, motion_length - 1)
            cmd_pos = stabilize_motion_dof_pos_29[t_clamp:t_clamp + 10] \
                if t_clamp + 10 <= motion_length \
                else np.tile(stabilize_motion_dof_pos_29[-1], (10, 1))
            cmd_vel = np.zeros_like(cmd_pos)
            cmd_seq = np.concatenate([cmd_pos, cmd_vel], axis=1).reshape(1, -1).astype(np.float32)

            obs_stab = build_stabilization_obs(jpos, jvel, bq, bav,
                                               stab_last_a29, cmd_seq, stabilize_obs_idx)
            raw29 = stabilize_session.run(None, {stabilize_input_name: obs_stab})[0][0]
            stab_last_a29 = raw29.copy()
            scaled29 = np.clip(raw29 * stabilize_act_scale_29 + stabilize_act_offset_29,
                               stabilize_act_clip_min_29, stabilize_act_clip_max_29)
            target_q = jpos.copy()
            target_q[stabilize_act_idx] = scaled29
            target_q[stabilize_hand_idx] = stabilize_hand_target_q[t_clamp]

            stab_cmd_acc += policy_dt
            stab_cmd_t    = min(int(stab_cmd_acc * motion_fps), motion_length - 1)

            if float(data.time) - stabilize_start_sim_t >= stabilize_duration:
                stabilize_active = False
                print(f"  [t={float(data.time):.2f}s] Stabilization done — OTT inference start.",
                      flush=True)
            return

        # ── Phase 3: Hand opening ──────────────────────────────────────────
        if hand_open_active:
            t_elapsed = float(data.time) - hand_open_start_sim_t
            alpha = trap_alpha(t_elapsed, HAND_OPEN_DURATION, blend=0.3)

            jpos = data.qpos[dof_qpos_addrs].astype(np.float32)
            if release_with_hotdex:
                obs = build_ott_obs(
                    data, dof_qpos_addrs, dof_qvel_addrs,
                    robot_qpos_addr, obj_qpos_addr, torso_body_id,
                    motion, motion_length - 1, obs_buf, obj_bps,
                )
                raw_action = session.run(None, {inp.name: obs})[0][0]
                scaled = np.clip(raw_action * ACTION_SCALE + ACTION_OFFSET,
                                 ACTION_CLIP_MIN, ACTION_CLIP_MAX)
                target_q = jpos.copy()
                target_q[act_idx] = scaled
            else:
                jvel = data.qvel[dof_qvel_addrs].astype(np.float32)
                bq   = data.qpos[robot_qpos_addr + 3:robot_qpos_addr + 7].astype(np.float32)
                bav  = data.qvel[robot_qvel_addr + 3:robot_qvel_addr + 6].astype(np.float32)
                cmd_pos = np.tile(final_body_cmd_29, (10, 1))
                cmd_seq = np.concatenate([cmd_pos, np.zeros_like(cmd_pos)],
                                         axis=1).reshape(1, -1).astype(np.float32)
                obs_stab = build_stabilization_obs(jpos, jvel, bq, bav,
                                                   stab_last_a29, cmd_seq, stabilize_obs_idx)
                raw29 = stabilize_session.run(None, {stabilize_input_name: obs_stab})[0][0]
                stab_last_a29 = raw29.copy()
                scaled29 = np.clip(raw29 * stabilize_act_scale_29 + stabilize_act_offset_29,
                                   stabilize_act_clip_min_29, stabilize_act_clip_max_29)
                target_q = jpos.copy()
                target_q[stabilize_act_idx] = scaled29

            target_q[hand_open_active_idx] = (hand_open_start_q_7
                                               + alpha * (hand_open_target_q_7 - hand_open_start_q_7))

            if t_elapsed >= HAND_OPEN_DURATION:
                hand_open_active      = False
                # Move table to the opposite side from the box before post-hold starts
                if args.randomize:
                    _new_txy       = _sample_table_opposite_box(box_xy)
                    _table_respawn_pos = TABLE_SPAWN.copy()
                    _table_respawn_pos[:2] = _new_txy
                    _set_table_pose_runtime(_table_respawn_pos, TABLE_QUAT_WXYZ)
                    _actual_txy = data.xpos[table_body_id, :2].astype(np.float32).copy()
                    print(
                        f"  [Table respawn] target_xy={_new_txy.round(3)} actual_xy={_actual_txy.round(3)}",
                        flush=True,
                    )
                    _spawn_object_on_table(_table_respawn_pos, choose_candidate=True)
                post_hold_active      = True
                post_hold_start_sim_t = float(data.time)
                print(f"  [Hand open] Complete — holding for {POST_HOLD_DURATION:.1f}s.", flush=True)
            return

        # ── Phase 4: Post-hold ─────────────────────────────────────────────
        if post_hold_active:
            t_elapsed = float(data.time) - post_hold_start_sim_t
            jpos = data.qpos[dof_qpos_addrs].astype(np.float32)
            if hotdex_when_post_release:
                obs = build_ott_obs(
                    data, dof_qpos_addrs, dof_qvel_addrs,
                    robot_qpos_addr, obj_qpos_addr, torso_body_id,
                    motion, motion_length - 1, obs_buf, obj_bps,
                )
                raw_action = session.run(None, {inp.name: obs})[0][0]
                scaled = np.clip(raw_action * ACTION_SCALE + ACTION_OFFSET,
                                 ACTION_CLIP_MIN, ACTION_CLIP_MAX)
                target_q = jpos.copy()
                target_q[act_idx] = scaled
            else:
                jvel = data.qvel[dof_qvel_addrs].astype(np.float32)
                bq   = data.qpos[robot_qpos_addr + 3:robot_qpos_addr + 7].astype(np.float32)
                bav  = data.qvel[robot_qvel_addr + 3:robot_qvel_addr + 6].astype(np.float32)
                cmd_pos = np.tile(final_body_cmd_29, (10, 1))
                cmd_seq = np.concatenate([cmd_pos, np.zeros_like(cmd_pos)],
                                         axis=1).reshape(1, -1).astype(np.float32)
                obs_stab = build_stabilization_obs(jpos, jvel, bq, bav,
                                                   stab_last_a29, cmd_seq, stabilize_obs_idx)
                raw29 = stabilize_session.run(None, {stabilize_input_name: obs_stab})[0][0]
                stab_last_a29 = raw29.copy()
                scaled29 = np.clip(raw29 * stabilize_act_scale_29 + stabilize_act_offset_29,
                                   stabilize_act_clip_min_29, stabilize_act_clip_max_29)
                target_q = jpos.copy()
                target_q[stabilize_act_idx] = scaled29

            if t_elapsed >= POST_HOLD_DURATION:
                post_hold_active          = False
                return_to_default_active  = True
                return_to_default_start_t = float(data.time)
                return_to_default_start_q = data.qpos[dof_qpos_addrs].astype(np.float32)[_motion_29_dof_idx].copy()
                print(f"  [Post-hold] Done — starting {RETURN_TO_DEFAULT_DURATION:.0f}s return to default.",
                      flush=True)
            return

        # ── Phase 5: Return to default pose ───────────────────────────────
        if return_to_default_active:
            t_elapsed = float(data.time) - return_to_default_start_t

            # Build 10-frame lookahead command.
            # First 3s: shoulder roll fixed at raise targets.
            # Last 2s: shoulder roll interpolates to default.
            cmd_pos = np.zeros((10, 29), dtype=np.float32)
            for k in range(10):
                t_k = t_elapsed + (k + 1) * policy_dt
                full_alpha = float(np.clip(t_k / RETURN_TO_DEFAULT_DURATION, 0.0, 1.0))
                cmd_k = (return_to_default_start_q
                         + full_alpha * (_default_q_29 - return_to_default_start_q))

                if t_k <= RETURN_ARM_RAISE_DURATION:
                    cmd_k[_return_shoulder_roll_idx_29] = _return_shoulder_roll_raise_target
                else:
                    arm_alpha = float(np.clip(
                        (t_k - RETURN_ARM_RAISE_DURATION)
                        / max(RETURN_TO_DEFAULT_DURATION - RETURN_ARM_RAISE_DURATION, 1e-6),
                        0.0, 1.0,
                    ))
                    cmd_k[_return_shoulder_roll_idx_29] = (
                        _return_shoulder_roll_raise_target
                        + arm_alpha * (
                            _default_q_29[_return_shoulder_roll_idx_29]
                            - _return_shoulder_roll_raise_target
                        )
                    )
                cmd_pos[k] = cmd_k.astype(np.float32)

            jpos = data.qpos[dof_qpos_addrs].astype(np.float32)
            jvel = data.qvel[dof_qvel_addrs].astype(np.float32)
            bq   = data.qpos[robot_qpos_addr + 3:robot_qpos_addr + 7].astype(np.float32)
            bav  = data.qvel[robot_qvel_addr + 3:robot_qvel_addr + 6].astype(np.float32)
            cmd_seq = np.concatenate([cmd_pos, np.zeros_like(cmd_pos)],
                                     axis=1).reshape(1, -1).astype(np.float32)
            obs_stab = build_stabilization_obs(jpos, jvel, bq, bav,
                                               stab_last_a29, cmd_seq, stabilize_obs_idx)
            raw29 = stabilize_session.run(None, {stabilize_input_name: obs_stab})[0][0]
            stab_last_a29 = raw29.copy()
            scaled29 = np.clip(raw29 * stabilize_act_scale_29 + stabilize_act_offset_29,
                               stabilize_act_clip_min_29, stabilize_act_clip_max_29)
            target_q = jpos.copy()
            target_q[stabilize_act_idx] = scaled29

            if t_elapsed >= RETURN_TO_DEFAULT_DURATION:
                return_to_default_active = False
                locomotion_active = True
                robot_xy = data.qpos[robot_qpos_addr:robot_qpos_addr + 2].astype(np.float32)
                table_xy = data.xpos[table_body_id, :2].astype(np.float32).copy()
                table_quat = model.body_quat[table_body_id].astype(np.float32)
                loco_waypoint_xy = select_table_waypoint_xy(
                    robot_xy=robot_xy,
                    table_xy=table_xy,
                    table_quat_wxyz=table_quat,
                    desired_dist=DESIRED_TABLE_ROBOT_DIST,
                )
                loco_last_raw_action[:] = 0.0
                loco_int_xy[:] = 0.0
                loco_prev_err_xy[:] = 0.0
                loco_int_yaw = 0.0
                loco_prev_err_yaw = 0.0
                loco_start_sim_t = float(data.time)
                print(
                    f"  [Return-to-default] Done — locomotion start, waypoint={loco_waypoint_xy.round(3)} "
                    f"(table={table_xy.round(3)}).",
                    flush=True,
                )
            return

        # ── Phase 6: Locomotion to table waypoint ─────────────────────────
        if locomotion_active:
            loco_elapsed = float(data.time) - loco_start_sim_t
            if loco_elapsed >= LOCO_TIMEOUT_SEC:
                locomotion_active = False
                print(
                    f"  [Locomotion] Timeout ({LOCO_TIMEOUT_SEC:.1f}s) — failed to reach waypoint. Continuing to hotdex.",
                    flush=True,
                )
                box_pos, box_xy, waypoint_z, r_new_box, theta_new_box = _sample_box_spawn_randomized()
                box_body_z = float(box_pos[2])
                _set_box_root_pose(box_pos)
                completed_iterations += 1
                print(
                    f"  [Cycle] Completed {completed_iterations}/{max_iterations}",
                    flush=True,
                )
                if completed_iterations >= max_iterations:
                    force_stop = True
                    print(
                        f"  [Cycle] Reached iteration limit ({max_iterations}). Stopping.",
                        flush=True,
                    )
                    return
                current_iteration = completed_iterations + 1
                _restart_hotdex_cycle()
                return

            robot_xy = data.qpos[robot_qpos_addr:robot_qpos_addr + 2].astype(np.float32)
            robot_quat = data.qpos[robot_qpos_addr + 3:robot_qpos_addr + 7].astype(np.float32)
            robot_yaw = yaw_from_quat_wxyz(robot_quat)
            table_xy = data.xpos[table_body_id, :2].astype(np.float32).copy()

            vel_command, loco_int_xy, loco_prev_err_xy, loco_int_yaw, loco_prev_err_yaw = \
                build_locomotion_velocity_command(
                    robot_xy=robot_xy,
                    robot_yaw=robot_yaw,
                    waypoint_xy=loco_waypoint_xy,
                    table_xy=table_xy,
                    dt=policy_dt,
                    integ_xy=loco_int_xy,
                    prev_err_xy=loco_prev_err_xy,
                    integ_yaw=loco_int_yaw,
                    prev_err_yaw=loco_prev_err_yaw,
                )

            obs_loco = build_locomotion_obs(
                data=data,
                dof_qpos_addrs=dof_qpos_addrs,
                dof_qvel_addrs=dof_qvel_addrs,
                robot_qpos_addr=robot_qpos_addr,
                robot_qvel_addr=robot_qvel_addr,
                vel_command=vel_command,
                last_raw_action=loco_last_raw_action,
            )
            raw_action = loco_session.run(None, {loco_input_name: obs_loco})[0].squeeze()
            loco_last_raw_action = raw_action.astype(np.float32).copy()
            scaled = np.clip(raw_action * ACTION_SCALE + ACTION_OFFSET,
                             ACTION_CLIP_MIN, ACTION_CLIP_MAX)
            target_q = data.qpos[dof_qpos_addrs].astype(np.float32).copy()
            target_q[act_idx] = scaled.astype(np.float32)

            pos_err = float(np.linalg.norm(loco_waypoint_xy - robot_xy))
            face_yaw = float(np.arctan2(float(table_xy[1] - robot_xy[1]),
                                        float(table_xy[0] - robot_xy[0])))
            yaw_err = abs(wrap_to_pi(face_yaw - robot_yaw))
            if pos_err <= LOCO_POS_TOL and yaw_err <= LOCO_YAW_TOL:
                box_pos, box_xy, waypoint_z, r_new_box, theta_new_box = _sample_box_spawn_randomized()
                box_body_z = float(box_pos[2])
                _set_box_root_pose(box_pos)
                box_world_xy = (
                    data.xpos[_box_root_bid, :2].astype(np.float32).copy()
                    if _box_root_bid >= 0 else box_xy.copy()
                )
                print(
                    f"  [Bin respawn] r={r_new_box:.3f} theta={np.degrees(theta_new_box):.1f}° "
                    f"pos={list(np.round(box_pos, 3))} waypoint_z={waypoint_z:.3f} "
                    f"actual_xy={box_world_xy.round(3)}",
                    flush=True,
                )
                locomotion_active = False
                print(
                    f"  [Locomotion] Done — pos_err={pos_err:.3f}m yaw_err={yaw_err:.3f}rad",
                    flush=True,
                )
                completed_iterations += 1
                print(
                    f"  [Cycle] Completed {completed_iterations}/{max_iterations}",
                    flush=True,
                )
                if completed_iterations >= max_iterations:
                    force_stop = True
                    print(
                        f"  [Cycle] Reached iteration limit ({max_iterations}). Stopping.",
                        flush=True,
                    )
                    return
                current_iteration = completed_iterations + 1
                _restart_hotdex_cycle()
            return

        # ── Phase 2: OTT inference ─────────────────────────────────────────
        si = np.clip(motion_timestep + np.arange(1, 11), 0, motion_length - 1)
        _viz_short_pos[:] = list(motion["obj_pos"][si])
        li = np.clip(motion_timestep + np.array([20, 40, 60, 80, 100]), 0, motion_length - 1)
        _viz_long_pos[:] = list(motion["obj_pos"][li])

        obs = build_ott_obs(
            data, dof_qpos_addrs, dof_qvel_addrs,
            robot_qpos_addr, obj_qpos_addr, torso_body_id,
            motion, motion_timestep, obs_buf, obj_bps,
        )
        raw_action = session.run(None, {inp.name: obs})[0][0]
        scaled = np.clip(raw_action * ACTION_SCALE + ACTION_OFFSET,
                         ACTION_CLIP_MIN, ACTION_CLIP_MAX)
        target_q = data.qpos[dof_qpos_addrs].astype(np.float32).copy()
        target_q[act_idx] = scaled

        motion_time_acc += policy_dt
        motion_timestep  = min(int(motion_time_acc * motion_fps), motion_length - 1)
        policy_step     += 1

        obj_z = float(data.qpos[obj_qpos_addr + 2])

        # Trigger hand open when OTT trajectory ends
        if motion_timestep >= motion_length - 1:
            full_q    = data.qpos[dof_qpos_addrs].astype(np.float32)
            obj_pos_w = data.qpos[obj_qpos_addr:obj_qpos_addr + 3]
            d_left  = (np.linalg.norm(data.xpos[_left_thumb_body_id]  - obj_pos_w)
                       if _left_thumb_body_id  >= 0 else np.inf)
            d_right = (np.linalg.norm(data.xpos[_right_thumb_body_id] - obj_pos_w)
                       if _right_thumb_body_id >= 0 else np.inf)
            if d_left <= d_right:
                hand_open_active_idx = _left_hand_dof_idx
                hand_open_target_q_7 = _left_hand_open_q.copy()
                active_label = "left"
            else:
                hand_open_active_idx = _right_hand_dof_idx
                hand_open_target_q_7 = _right_hand_open_q.copy()
                active_label = "right"
            final_body_cmd_29     = full_q[_motion_29_dof_idx].copy()
            hand_open_start_q_7   = full_q[hand_open_active_idx].copy()
            hand_open_active      = True
            hand_open_start_sim_t = float(data.time)
            print(f"  [Hand open] hand={active_label} — opening from grasp pose.", flush=True)

        if policy_step % 50 == 0:
            total_f, robot_f, n_con = compute_obj_contact_force(
                model, data, _obj_geom_ids, _robot_geom_ids)
            print(
                f"  [t={motion_time_acc:6.2f}s] step={policy_step} "
                f"motion={motion_timestep}/{motion_length}  "
                f"base_z={float(data.qpos[robot_qpos_addr+2]):.3f}  obj_z={obj_z:.3f}  "
                f"contact: robot={robot_f:.1f}N total={total_f:.1f}N n={n_con}",
                flush=True,
            )

    def _apply_ctrl(d):
        apply_pd_control(d, target_q, dof_qpos_addrs, dof_qvel_addrs, actuator_ids)

    def _on_reset():
        nonlocal target_q, motion_timestep, motion_time_acc, policy_step
        nonlocal stab_last_a29, stabilize_active, stabilize_start_sim_t, stab_cmd_t, stab_cmd_acc
        nonlocal hand_open_active, post_hold_active, return_to_default_active, locomotion_active
        nonlocal motion, motion_length, stabilize_motion_dof_pos_29, stabilize_hand_target_q
        nonlocal waypoint_z, box_xy, box_pos, box_body_z
        nonlocal loco_last_raw_action, loco_int_xy, loco_prev_err_xy, loco_int_yaw, loco_prev_err_yaw
        nonlocal loco_start_sim_t, force_stop
        nonlocal completed_iterations, current_iteration
        nonlocal pre_hotdex_active, pre_hotdex_start_t

        policy_step = motion_timestep = 0
        motion_time_acc = stab_cmd_acc = 0.0
        stab_cmd_t              = 0
        hand_open_active        = False
        post_hold_active        = False
        return_to_default_active = False
        locomotion_active       = False
        pre_hotdex_active       = False
        obs_buf.reset()
        stab_last_a29[:] = 0.0
        loco_last_raw_action[:] = 0.0
        loco_int_xy[:] = 0.0
        loco_prev_err_xy[:] = 0.0
        loco_int_yaw = 0.0
        loco_prev_err_yaw = 0.0
        loco_start_sim_t = 0.0
        force_stop = False
        completed_iterations = 0
        current_iteration = 1

        # Re-randomize box position and waypoint_z each episode
        if args.randomize:
            box_pos, box_xy, waypoint_z, r, theta = _sample_box_spawn_randomized()
            box_body_z = float(box_pos[2])
            _set_box_root_pose(box_pos)
            print(
                f"  [Reset] box r={r:.3f} theta={np.degrees(theta):.1f}° "
                f"pos={list(np.round(box_pos, 3))} waypoint_z={waypoint_z:.3f}",
                flush=True,
            )

        obj_spawn = _reset_scene()

        if args.randomize:
            motion = generate_box_trajectory(
                obj_spawn    = obj_spawn,
                box_xy       = box_xy,
                waypoint_z   = waypoint_z,
                motion_fps   = motion_fps,
                hold_sec     = args.hold_sec,
                tgt_hold_sec = args.tgt_hold_sec,
                obj_rot0_wxyz= OBJ_QUAT_WXYZ,
                speed_scale  = args.traj_speed_scale,
            )
            motion_length = motion["motion_length"]
            _viz_full_traj[:] = list(motion["obj_pos"][::_viz_sample_step])
            stabilize_motion_dof_pos_29, _, stabilize_hand_target_q = \
                build_stabilization_refs(dof_pos_43, motion_length)

        target_q = data.qpos[dof_qpos_addrs].astype(np.float32).copy()
        stabilize_active      = stabilize_enabled
        stabilize_start_sim_t = float(data.time)

    def _should_stop():
        if force_stop:
            return True
        if (pre_hotdex_active or stabilize_active or hand_open_active or post_hold_active
                or return_to_default_active or locomotion_active):
            return False
        return motion_timestep >= motion_length - 1

    def _should_stop_wrapper():
        if force_stop:
            return True
        base_z = float(data.qpos[robot_qpos_addr + 2])
        if base_z < 0.2:
            print(f"  [Fall detected] base_z={base_z:.3f} < 0.2 — stopping.", flush=True)
            return True
        if args.offscreen:
            return _should_stop()
        return False

    _ID_MAT     = np.eye(3).flatten().astype(np.float64)
    _SHORT_RGBA = np.array([0.2, 0.9, 0.2, 0.8], dtype=np.float32)
    _LONG_RGBA  = np.array([0.9, 0.2, 0.2, 0.8], dtype=np.float32)
    _FULL_RGBA  = np.array([0.2, 0.8, 0.9, 0.6], dtype=np.float32)
    _WAYPOINT_RGBA = np.array([1.0, 0.1, 0.1, 0.95], dtype=np.float32)
    _PT_SIZE    = np.array([0.015, 0.0, 0.0], dtype=np.float64)
    _PT_SIZE_SM = np.array([0.010, 0.0, 0.0], dtype=np.float64)
    _PT_SIZE_WAYPOINT = np.array([0.03, 0.0, 0.0], dtype=np.float64)

    def _add_traj_markers(scn):
        for pos in _viz_full_traj:
            if scn.ngeom >= scn.maxgeom:
                return
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                _PT_SIZE_SM, np.asarray(pos, dtype=np.float64),
                                _ID_MAT, _FULL_RGBA)
            scn.ngeom += 1
        for pts, rgba in ((_viz_short_pos, _SHORT_RGBA), (_viz_long_pos, _LONG_RGBA)):
            for pos in pts:
                if scn.ngeom >= scn.maxgeom:
                    return
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                    _PT_SIZE, np.asarray(pos, dtype=np.float64),
                                    _ID_MAT, rgba)
                scn.ngeom += 1
        if locomotion_active and scn.ngeom < scn.maxgeom:
            waypoint_pos = np.array(
                [float(loco_waypoint_xy[0]), float(loco_waypoint_xy[1]), 0.7],
                dtype=np.float64,
            )
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                _PT_SIZE_WAYPOINT,
                waypoint_pos,
                _ID_MAT,
                _WAYPOINT_RGBA,
            )
            scn.ngeom += 1

    try:
        record_stride = max(1, round(args.policy_hz / args.record_hz))
        run_mujoco_loop(
            model, data,
            sim_dt=sim_dt,
            steps_per_policy=steps_per_policy,
            on_policy_step=_on_policy_step,
            apply_ctrl=_apply_ctrl,
            on_reset=_on_reset,
            should_stop=_should_stop_wrapper,
            video_recorder=video_recorder,
            offscreen=args.offscreen,
            on_render=_add_traj_markers,
            record_stride=record_stride,
        )
    finally:
        if video_recorder is not None:
            actual_record_hz = args.policy_hz / record_stride
            video_recorder.save(fps=actual_record_hz)
            video_recorder.cleanup()

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Open-box placement: spawn robot+object+box, generate trajectory, "
                    "run stabilization then OTT BPS policy and release into box."
    )
    parser.add_argument(
        "--obj-name", required=True,
        help="Object name (e.g. cubemedium). Determines robot XML and BPS code.",
    )
    parser.add_argument(
        "--box-pos", type=float, nargs=2, default=list(BOX_XY_DEFAULT),
        metavar=("X", "Y"),
        help="Box XY spawn position. Default: same region as cabinet default.",
    )
    parser.add_argument(
        "--waypoint-z", type=float, default=None,
        metavar="Z",
        help=f"Object release height (m). If omitted, sampled uniformly from "
             f"[{WAYPOINT_Z_RANGE[0]}, {WAYPOINT_Z_RANGE[1]}].",
    )
    parser.add_argument(
        "--box-xml", type=str,
        default="src/holosoma/holosoma/data/objects/open_box.xml",
        help="Path to open-box MJCF XML.",
    )
    parser.add_argument(
        "--motion-fps", type=float, default=120.0,
        help="Trajectory sampling frequency in Hz. Default: 120.",
    )
    parser.add_argument(
        "--traj-speed-scale", type=float, default=1.0,
        metavar="S",
        help="Scale factor for trajectory speed. <1 = slower. Default: 1.0.",
    )
    parser.add_argument(
        "--hold-sec", type=float, default=2.5,
        help="Seconds to hold at spawn before moving. Default: 2.5.",
    )
    parser.add_argument(
        "--tgt-hold-sec", type=float, default=1.0,
        help="Seconds to hold at release point before hand opens. Default: 1.0.",
    )
    parser.add_argument(
        "--robot-root-pos", type=float, nargs=3,
        default=list(DEFAULT_ROBOT_ROOT_POS),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--robot-root-quat-xyzw", type=float, nargs=4,
        default=list(DEFAULT_ROBOT_ROOT_QUAT_XYZW),
        metavar=("X", "Y", "Z", "W"),
    )
    parser.add_argument(
        "--robot-dof-pos", type=float, nargs=43,
        default=list(DEFAULT_ROBOT_DOF_POS_43),
        metavar="Q",
    )
    parser.add_argument("--policy", default=None,
                        help="Path to OTT ONNX policy.")
    parser.add_argument("--sim-hz",    type=int, default=200)
    parser.add_argument("--policy-hz", type=int, default=50)
    parser.add_argument(
        "--stabilize-sec", "--stabilize_sec", dest="stabilize_sec",
        type=float, default=0.0,
        help="Seconds to run master_policy stabilization before OTT. Default: 0.",
    )
    parser.add_argument("--stabilize-policy", default=None)
    parser.add_argument("--loco-policy", default=None,
                        help="Path to locomotion ONNX policy. Default: walk_prior_dr_0306.onnx")
    parser.add_argument("--record",    action="store_true")
    parser.add_argument("--offscreen", action="store_true",
                        help="Run headless, record video, stop when trajectory ends.")
    parser.add_argument("--release-with-hotdex", "--release_with_hotdex",
                        dest="release_with_hotdex", action="store_true")
    parser.add_argument("--hotdex-when-post-release", "--hotdex_when_post_release",
                        dest="hotdex_when_post_release", action="store_true")
    parser.add_argument("--view-only", "--view_only", dest="view_only", action="store_true",
                        help="Open interactive viewer; no policy runs.")
    parser.add_argument("--video-dir",    type=str, default="logs/videos")
    parser.add_argument("--video-width",  type=int, default=960)
    parser.add_argument("--video-height", type=int, default=540)
    parser.add_argument("--video-format", type=str, default="h264", choices=["h264", "mp4"])
    parser.add_argument("--record-hz", "--record_hz", dest="record_hz",
                        type=float, default=25.0,
                        help="Video recording frequency in Hz. Default: 25.")
    parser.add_argument("--camera-pos",    type=float, nargs=3, default=[-2.0, 0.5, 1.5])
    parser.add_argument("--camera-target", type=float, nargs=3, default=[0.5, 0.0, 1.0])
    parser.add_argument(
        "--iteration", "--iteratrion", dest="iteration",
        type=int, default=5,
        help="Number of full cycles (hotdex -> bin placing -> locomotion). Default: 5.",
    )
    parser.add_argument(
        "--randomize", action="store_true",
        help="Randomize box XY position (r∈[0.5,2.0], θ∈[0°,180°]) and waypoint_z "
             f"∈{WAYPOINT_Z_RANGE} each episode. Also randomizes robot xy ±0.1 m, "
             "robot yaw ±0.5 rad, object xy ±0.1 m, object yaw ±180°, "
             "joint positions ±0.1 rad, table z ±0.2 m.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

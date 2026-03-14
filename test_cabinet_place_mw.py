#!/usr/bin/env python3
"""
Cabinet placement policy test.

Spawns robot + object + table + cabinet in MuJoCo, generates a trajectory
from the object's spawn position to a chosen cabinet shelf, then runs
auto-stabilization followed by BPS policy inference.

No PKL required — robot init pose and object name are provided directly.

Usage:
    python test_cabinet_place_mw.py --obj-name cubemedium --shelf 2
    python test_cabinet_place_mw.py --obj-name cubemedium --stabilize-sec 2.0
    python test_cabinet_place_mw.py --obj-name cubemedium --shelf 1 --offscreen --record
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

# Cabinet shelf body z-positions (from open_cabinet.xml, relative to cabinet root body)
SHELF_HEIGHTS = {1: 0.5, 2: 0.79, 3: 1.08, 4: 1.37}
SHELF_CHOICES = [2, 3]   # shelves available for random selection

# ── Spawn positions & rotations ────────────────────────────────────────────
# Positions are in robot-relative coordinates (robot xy = 0,0); z is world absolute (ground = 0).
OBJ_SPAWN_DEFAULT   = np.array([-0.04082961, -0.98491963, 0.8744048704122025],
                                dtype=np.float32)
TABLE_SPAWN_DEFAULT = np.array([-0.04411870, -1.02271030, 0.8286774158477783],
                                dtype=np.float32)
CABINET_SPAWN_DEFAULT = np.array([-0.08303487, 0.80154856, 0.0], dtype=np.float32)

# Rotations stored as wxyz (MuJoCo convention), converted from provided xyzw values
# table  xyzw: [0.70392106, 0.02927242, 0.02412632, 0.70926454]
TABLE_QUAT_WXYZ = np.array([0.70926454, 0.70392106, 0.02927242, 0.02412632], dtype=np.float32)
# object xyzw: [0.5647669434547424, 0.4294104278087616, 0.4254690110683441, 0.5618017911911011]
OBJ_QUAT_WXYZ   = np.array([1,0,0,0], dtype=np.float32)
# cabinet: 180° around Z axis so it faces the robot  (wxyz)
CABINET_QUAT_WXYZ = np.array([0.70710678, 0, 0, 0.70710678], dtype=np.float32)

# Default robot init state (provided by user; dof_pos in G1_DEX3_JOINT_NAMES order)
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

# 43-DOF motion clip / retargeting joint order
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

# Stabilization (master_policy) obs/action joint orders
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
    """Convert 43-DOF dof_pos (G1_DEX3_JOINT_NAMES order) → DOF_NAMES config order."""
    config_idx = name_indices(G1_DEX3_JOINT_NAMES, DOF_NAMES)
    init_q = DEFAULT_DOF_ANGLES.copy()
    init_q[config_idx] = dof_pos_43_dex3order
    return init_q


def build_stabilization_refs(
    dof_pos_43_dex3order: np.ndarray,
    T: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build static stabilization reference arrays by holding init pose.

    Returns:
        stabilize_motion_dof_pos_29  (T, 29)
        stabilize_motion_dof_vel_29  (T, 29)  zeros
        stabilize_hand_target_q      (T, 14)
    """
    body_in_dex3 = name_indices(G1_MOTION_JOINT_NAMES_29, G1_DEX3_JOINT_NAMES)
    hand_in_dex3 = name_indices(HAND_JOINT_NAMES, G1_DEX3_JOINT_NAMES)

    body_pos_init = dof_pos_43_dex3order[body_in_dex3]   # (29,)
    hand_pos_init = dof_pos_43_dex3order[hand_in_dex3]   # (14,)

    stab_pos = np.tile(body_pos_init, (T, 1)).astype(np.float32)   # (T, 29)
    stab_vel = np.zeros((T, 29), dtype=np.float32)
    hand_tgt = np.tile(hand_pos_init, (T, 1)).astype(np.float32)   # (T, 14)
    return stab_pos, stab_vel, hand_tgt


# ═══════════════════════════════════════════════════════════════════════════
# Cabinet XML merge
# ═══════════════════════════════════════════════════════════════════════════

def build_combined_model(
    robot_xml_path: str,
    cabinet_xml_path: str,
    cabinet_pos,
    cabinet_quat_wxyz=None,
) -> mujoco.MjModel:
    """Merge cabinet geometry into the robot XML and return a MjModel.

    Cabinet worldbody contents are wrapped in a body at cabinet_pos/quat so all
    shelf positions become offsets from that origin. A temp file is written
    next to the robot XML so relative mesh paths resolve correctly.
    """
    robot_tree   = ET.parse(robot_xml_path)
    cabinet_tree = ET.parse(cabinet_xml_path)
    robot_root   = robot_tree.getroot()
    cabinet_root = cabinet_tree.getroot()

    # Merge assets (fix texture path to absolute)
    robot_assets   = robot_root.find("asset")
    cabinet_assets = cabinet_root.find("asset")
    if cabinet_assets is not None and robot_assets is not None:
        for child in list(cabinet_assets):
            if child.tag == "texture" and "file" in child.attrib:
                abs_path = (Path(cabinet_xml_path).parent / child.attrib["file"]).resolve()
                child.set("file", str(abs_path))
            robot_assets.append(child)

    # Wrap cabinet worldbody in a positioned body
    cx, cy, cz = float(cabinet_pos[0]), float(cabinet_pos[1]), float(cabinet_pos[2])
    body_attrib = {"name": "cabinet_root", "pos": f"{cx} {cy} {cz}"}
    if cabinet_quat_wxyz is not None:
        w, x, y, z = [float(v) for v in cabinet_quat_wxyz]
        body_attrib["quat"] = f"{w} {x} {y} {z}"
    cabinet_body = ET.SubElement(
        robot_root.find("worldbody"),
        "body",
        attrib=body_attrib,
    )
    for child in list(cabinet_root.find("worldbody")):
        if child.tag != "camera":
            cabinet_body.append(child)

    # Write temp XML adjacent to robot XML and load
    robot_dir    = Path(robot_xml_path).parent
    combined_str = ET.tostring(robot_root, encoding="unicode")
    fd, tmp_path = tempfile.mkstemp(suffix=".xml", prefix="_cabinet_combined_", dir=str(robot_dir))
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
    """Trapezoidal-velocity profile segment in 3D. Returns list of (x,y,z) tuples."""
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


def generate_cabinet_trajectory(
    obj_spawn: np.ndarray,
    cabinet_pos: np.ndarray,
    cabinet_quat_wxyz: np.ndarray,
    shelf: int,
    motion_fps: float,
    hold_sec: float = 2.5,
    tgt_hold_sec: float = 1.0,
    obj_rot0_wxyz: np.ndarray | None = None,
    lift_height: float = 0.15,
    speed_scale: float = 1.0,
) -> dict:
    """Generate object trajectory from spawn to cabinet shelf.

    Phases:
      1. Hold at spawn for hold_sec seconds.
      2. Lift object vertically by lift_height using trapezoidal profile.
      3. Move from lifted spawn to entrance waypoint (in front of cabinet opening).
      4. Enter cabinet: waypoint → shelf target (slower).
      5. Hold at target for tgt_hold_sec seconds.

    Returns a motion dict with:
        obj_pos        (T, 3)
        obj_rot        (T, 4)  wxyz
        motion_length  int
        motion_fps     float
    """
    dt = 1.0 / motion_fps
    R = quat_to_rot_mat(cabinet_quat_wxyz).astype(np.float64)  # cabinet → world

    shelf_body_z = SHELF_HEIGHTS[shelf]
    shelf_tgt_z  = shelf_body_z + 0.04   # slightly above shelf surface
    wp_z_offset  = 0.08                  # raised height above target
    # Points in cabinet body frame
    wp_cab   = np.array([0.3, 0.0, shelf_tgt_z + wp_z_offset], dtype=np.float64)  # entrance, raised
    wp2_cab  = np.array([0.8, 0.0, shelf_tgt_z + wp_z_offset], dtype=np.float64)  # above target, same height as wp

    # Transform to world frame
    cp = cabinet_pos.astype(np.float64)
    wp_world   = cp + R @ wp_cab
    wp2_world  = cp + R @ wp2_cab

    sp = obj_spawn.astype(np.float64)
    lifted_spawn = sp.copy()
    lifted_spawn[2] += lift_height

    print(f"  Shelf {shelf}: body_z={shelf_body_z}")
    print(f"  Entrance waypoint (world): {wp_world.round(3)}")
    print(f"  Final destination (world): {wp2_world.round(3)}")

    # ── Build trajectory segments ─────────────────────────────────────────
    out: list = []

    # 1. Hold at spawn
    hold_frames = max(0, int(round(hold_sec / dt)))
    out.extend([tuple(sp.tolist())] * hold_frames)

    s = float(speed_scale)
    # 2. Lift vertically at spawn
    out.extend(_trap_segment_3d(sp, lifted_spawn, v_max=0.3*s, a_max=0.6*s, dt=dt))

    # 3. Move smoothly from lifted spawn to entrance waypoint (diagonal, no separate descent)
    out.extend(_trap_segment_3d(lifted_spawn, wp_world, v_max=0.6*s, a_max=1.2*s, dt=dt))

    # 4. Enter cabinet horizontally: entrance waypoint → above target (same height)
    out.extend(_trap_segment_3d(wp_world, wp2_world, v_max=0.3*s, a_max=0.6*s, dt=dt))

    # 5. Hold at above-target (wp2 is the final destination)
    tgt_hold_frames = max(0, int(round(tgt_hold_sec / dt)))
    if out:
        out.extend([out[-1]] * tgt_hold_frames)

    T_new = len(out)
    new_obj_pos = np.array(out, dtype=np.float32)

    # Object rotation: constant from provided initial rotation
    if obj_rot0_wxyz is None:
        obj_rot0_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    q0_xyzw = np.array([obj_rot0_wxyz[1], obj_rot0_wxyz[2], obj_rot0_wxyz[3], obj_rot0_wxyz[0]],
                       dtype=np.float32)
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
    """Return the set of geom indices attached to a body."""
    return {i for i in range(model.ngeom) if model.geom_bodyid[i] == body_id}


def compute_obj_contact_force(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    obj_geom_ids: set,
    robot_geom_ids: set | None = None,
) -> tuple[float, float, int]:
    """Return (total_N, robot_N, n_contacts) for contacts involving the object.

    total_N      — sum of normal forces from ALL contacts on the object (N)
    robot_N      — sum of normal forces from robot-only contacts (N)
    n_contacts   — number of active contacts on the object
    """
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
    data.qpos[qpos_addr:qpos_addr + 3] = pos
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
    """Pure yaw rotation (around world Z) as wxyz quaternion."""
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float32)


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions."""
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
    yaw = np.arctan2(fwd_y, fwd_x)
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float32)


def rot_mat_to_6d(R: np.ndarray) -> np.ndarray:
    return R[:, :2].flatten().astype(np.float32)


def obj_pos_rel_heading(obj_world, root_world, hquat):
    return quat_rotate_inverse(hquat, obj_world - root_world)


def obj_rot_rel_heading(obj_quat_wxyz, hquat):
    return rot_mat_to_6d(quat_to_rot_mat(hquat).T @ quat_to_rot_mat(obj_quat_wxyz))


def trap_alpha(t: float, T: float, blend: float = 0.3) -> float:
    """Trapezoidal velocity profile: position ratio [0, 1] at time t.

    Velocity ramps up over (blend*T), cruises, then ramps down symmetrically.
    blend=0.3 means 30% accel + 40% cruise + 30% decel.
    """
    t = float(np.clip(t, 0.0, T))
    t_a = blend * T                     # accel / decel duration
    v_p = 1.0 / (T - t_a)              # peak velocity (area under trapezoid = 1)
    if t <= t_a:
        return 0.5 * (v_p / t_a) * t * t
    elif t <= T - t_a:
        return 0.5 * v_p * t_a + v_p * (t - t_a)
    else:
        t_rem = T - t
        return 1.0 - 0.5 * (v_p / t_a) * t_rem * t_rem


def rot_to_rpy_xyz(R: np.ndarray) -> np.ndarray:
    """Decompose rotation matrix R into XYZ (roll=X, pitch=Y, yaw=Z) Euler angles."""
    pitch = float(np.arcsin(np.clip(-float(R[2, 0]), -1.0, 1.0)))
    if abs(np.cos(pitch)) > 1e-6:
        roll = float(np.arctan2(float(R[2, 1]), float(R[2, 2])))
        yaw  = float(np.arctan2(float(R[1, 0]), float(R[0, 0])))
    else:
        roll = float(np.arctan2(-float(R[1, 2]), float(R[1, 1])))
        yaw  = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float32)


def compute_camera_from_scene(
    cabinet_pos: np.ndarray,
    table_pos: np.ndarray,
    cabinet_quat_wxyz: np.ndarray,
    camera_height: float,
    shelf: int,
    fov_deg: float = 45.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute camera pos/target so cabinet and table fit within 75% of FOV.

    Places the camera on the perpendicular bisector of the cabinet-table segment
    at the distance L where angle(cabinet, camera, table) = 0.75 * fov.
    Of the two candidates, picks the one whose vector from cabinet has a positive
    dot product with the cabinet's front (local +x) direction.

    Returns:
        camera_pos    (3,) float32  — xy auto-computed, z = camera_height
        camera_target (3,) float32  — midpoint of cabinet & table, z = mid of table and shelf
    """
    target_angle = 0.75 * np.radians(fov_deg)

    cab_2d = cabinet_pos[:2].astype(np.float64)
    tbl_2d = table_pos[:2].astype(np.float64)

    # Cabinet front vector: local +x axis in world xy (opening direction)
    R            = quat_to_rot_mat(cabinet_quat_wxyz).astype(np.float64)
    cab_front_2d = R[:2, 0]

    # Midpoint and half-distance
    mid_2d    = (cab_2d + tbl_2d) / 2.0
    ab        = tbl_2d - cab_2d
    ab_norm   = float(np.linalg.norm(ab))
    half_dist = ab_norm / 2.0

    ab_dir   = ab / ab_norm if ab_norm > 1e-6 else np.array([1.0, 0.0])
    perp_dir = np.array([-ab_dir[1], ab_dir[0]])  # 90° CCW from ab_dir

    # Camera on perpendicular bisector at distance L:
    #   angle(cab, cam, tbl) = 2 * arctan(half_dist / L)  →  L = half_dist / tan(α/2)
    L = half_dist / np.tan(target_angle / 2.0)

    cam_p = mid_2d + L * perp_dir
    cam_n = mid_2d - L * perp_dir

    # Pick candidate that is on the front (opening) side of the cabinet
    dot_p = float(np.dot(cam_p - cab_2d, cab_front_2d))
    dot_n = float(np.dot(cam_n - cab_2d, cab_front_2d))
    cam_2d = cam_p if dot_p >= dot_n else cam_n

    cam_pos = np.array([cam_2d[0], cam_2d[1], camera_height], dtype=np.float32)

    # Target: midpoint xy, z halfway between table surface and selected shelf
    shelf_z_abs = float(cabinet_pos[2]) + SHELF_HEIGHTS[shelf]
    target_z    = (float(table_pos[2]) + shelf_z_abs) / 2.0
    cam_target  = np.array([float(mid_2d[0]), float(mid_2d[1]), target_z], dtype=np.float32)

    return cam_pos, cam_target


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


# ═══════════════════════════════════════════════════════════════════════════
# Main simulation
# ═══════════════════════════════════════════════════════════════════════════

def run(args):
    root = Path(__file__).parent

    # ── Robot init state ───────────────────────────────────────────────────
    root_pos_init  = np.asarray(args.robot_root_pos, dtype=np.float32)
    root_quat_xyzw = np.asarray(args.robot_root_quat_xyzw, dtype=np.float32)
    root_quat_wxyz = np.array([root_quat_xyzw[3],
                                root_quat_xyzw[0],
                                root_quat_xyzw[1],
                                root_quat_xyzw[2]], dtype=np.float32)
    dof_pos_43     = np.asarray(args.robot_dof_pos, dtype=np.float32)
    if dof_pos_43.shape != (43,):
        raise ValueError(f"--robot-dof-pos must have 43 values, got {dof_pos_43.shape}")

    init_q = build_init_q(dof_pos_43)   # DOF_NAMES config order

    # ── Resolve XML ────────────────────────────────────────────────────────
    xml_dir  = root / "src/holosoma/holosoma/data/robots/g1/g1_object"
    xml_path = str(xml_dir / f"g1_43dof_{args.obj_name}.xml")
    if not Path(xml_path).exists():
        raise FileNotFoundError(f"No XML for obj_name='{args.obj_name}': {xml_path}")

    cabinet_xml_path = str(Path(args.cabinet_xml).expanduser())
    if not Path(cabinet_xml_path).is_absolute():
        cabinet_xml_path = str(root / args.cabinet_xml)
    if not Path(cabinet_xml_path).exists():
        raise FileNotFoundError(f"Cabinet XML not found: {cabinet_xml_path}")

    print(f"Robot XML  : {xml_path}")
    print(f"Cabinet XML: {cabinet_xml_path}")

    # ── Shelf selection ────────────────────────────────────────────────────
    shelf = args.shelf if args.shelf is not None else int(np.random.choice(SHELF_CHOICES))
    print(f"Target shelf: {shelf}  (cabinet XML body z={SHELF_HEIGHTS[shelf]}m)")

    # ── Spawn positions ────────────────────────────────────────────────────
    OBJ_SPAWN   = OBJ_SPAWN_DEFAULT.copy()
    TABLE_SPAWN = TABLE_SPAWN_DEFAULT.copy()
    cabinet_pos = np.asarray(args.cabinet_pos, dtype=np.float32)

    # ── Cabinet height randomization ───────────────────────────────────────
    if args.cabinet_z_offset is not None:
        cabinet_z_offset = float(args.cabinet_z_offset)
    else:
        lo, hi = float(args.cabinet_z_range[0]), float(args.cabinet_z_range[1])
        cabinet_z_offset = float(np.random.uniform(lo, hi))
    cabinet_pos[2] += cabinet_z_offset
    print(f"Cabinet z offset: {cabinet_z_offset:+.3f}  →  cabinet_pos={list(np.round(cabinet_pos, 3))}")

    # ── Cabinet quaternion (needed for trajectory + model) ─────────────────
    cabinet_quat = np.asarray(args.cabinet_quat_wxyz, dtype=np.float32)

    # ── Cabinet pose randomization ──────────────────────────────────────────
    if args.randomize:
        r     = float(np.random.uniform(0.5, 2.0))
        theta = float(np.random.uniform(0, np.pi))
        cabinet_pos[0] = r * np.cos(theta)
        cabinet_pos[1] = r * np.sin(theta)
        base_yaw  = float(np.arctan2(cabinet_pos[1], cabinet_pos[0]))
        noise_yaw = float(np.random.uniform(-np.pi / 4, np.pi / 4))
        yaw       = base_yaw + noise_yaw
        cabinet_quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float32)
        print(f"Randomized cabinet: r={r:.3f}  theta={np.degrees(theta):.1f}°  "
              f"yaw_base={np.degrees(base_yaw):.1f}°  yaw_noise={np.degrees(noise_yaw):.1f}°  "
              f"pos={list(np.round(cabinet_pos, 3))}")

    # ── Generate placement trajectory ──────────────────────────────────────
    motion_fps = float(args.motion_fps)
    print(f"Generating cabinet trajectory @ {motion_fps}Hz …")
    motion = generate_cabinet_trajectory(
        obj_spawn         = OBJ_SPAWN,
        cabinet_pos       = cabinet_pos,
        cabinet_quat_wxyz = cabinet_quat,
        shelf             = shelf,
        motion_fps        = motion_fps,
        hold_sec          = args.hold_sec,
        tgt_hold_sec      = args.tgt_hold_sec,
        obj_rot0_wxyz     = OBJ_QUAT_WXYZ,
        speed_scale       = args.traj_speed_scale,
    )
    motion_length = motion["motion_length"]

    # Pre-compute full trajectory markers (sampled at 0.5s intervals, static)
    _viz_sample_step = max(1, int(0.5 * motion_fps))
    _viz_full_traj   = list(motion["obj_pos"][::_viz_sample_step])

    # ── Camera placement from scene geometry ────────────────────────────────
    _cam_pos, _cam_target = compute_camera_from_scene(
        cabinet_pos, TABLE_SPAWN, cabinet_quat,
        camera_height=float(args.camera_pos[2]),
        shelf=shelf,
    )
    print(f"Camera: pos={list(np.round(_cam_pos, 3))}  target={list(np.round(_cam_target, 3))}")

    # ── Load policy ────────────────────────────────────────────────────────
    policy_path = args.policy or str(
        root / "src/holosoma_inference/holosoma_inference/models/wbt/object/bps_policy.onnx"
    )
    print(f"Loading OTT policy: {policy_path}")
    session = onnxruntime.InferenceSession(policy_path)
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    print(f"  Input:  {inp.name} {inp.shape}")
    print(f"  Output: {out.name} {out.shape}")

    # ── Load MuJoCo model (robot + cabinet) ───────────────────────────────
    print(f"Building combined model (robot + cabinet at {list(cabinet_pos)}) …")
    model = build_combined_model(xml_path, cabinet_xml_path, cabinet_pos, cabinet_quat)
    data  = mujoco.MjData(model)
    model.opt.timestep = 1.0 / args.sim_hz
    sim_dt           = float(model.opt.timestep)
    steps_per_policy = args.sim_hz // args.policy_hz
    policy_dt        = sim_dt * steps_per_policy
    print(f"Model: {model.nq} qpos  {model.nv} qvel  {model.nu} actuators  dt={sim_dt}s")
    print(f"Physics {1/sim_dt:.0f}Hz | Policy {1/policy_dt:.0f}Hz ({steps_per_policy} substeps)")

    dof_qpos_addrs, dof_qvel_addrs = mj_hinge_addrs(model, DOF_NAMES)
    actuator_ids    = mj_actuator_ids(model, DOF_NAMES)
    obj_qpos_addr   = get_freejoint_qpos_addr(model, "object")
    robot_qpos_addr = get_freejoint_qpos_addr(model, "pelvis")
    _ = get_freejoint_qvel_addr(model, "object")   # reserved / unused
    robot_qvel_addr = get_freejoint_qvel_addr(model, "pelvis")
    torso_body_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    if torso_body_id < 0:
        raise KeyError("Body 'torso_link' not found.")
    act_idx = name_indices(ACTION_ORDER_43DOF, DOF_NAMES)

    # ── Contact force monitoring ───────────────────────────────────────────
    _obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    _obj_geom_ids = get_body_geom_ids(model, _obj_body_id)

    # Exclude: world(0), object, table, and the full cabinet subtree (BFS from cabinet_root)
    _exclude_body_ids = {0, _obj_body_id}
    _table_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    if _table_bid >= 0:
        _exclude_body_ids.add(_table_bid)
    _cab_root_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cabinet_root")
    if _cab_root_bid >= 0:
        _queue = [_cab_root_bid]
        while _queue:
            _b = _queue.pop()
            _exclude_body_ids.add(_b)
            for _c in range(model.nbody):
                if int(model.body_parentid[_c]) == _b and _c not in _exclude_body_ids:
                    _queue.append(_c)

    _robot_geom_ids = {i for i in range(model.ngeom)
                       if model.geom_bodyid[i] not in _exclude_body_ids}
    print(f"Contact monitoring: obj_geoms={len(_obj_geom_ids)}  "
          f"robot_geoms={len(_robot_geom_ids)}")

    # ── Stabilization setup ────────────────────────────────────────────────
    stabilize_duration = float(args.stabilize_sec)
    stabilize_enabled  = stabilize_duration > 0.0
    stabilize_session  = stabilize_input_name = None
    stabilize_obs_idx  = stabilize_act_idx    = None
    stabilize_act_scale_29 = stabilize_act_offset_29 = None
    stabilize_act_clip_min_29 = stabilize_act_clip_max_29 = None
    stabilize_hand_idx = None
    stabilize_motion_dof_pos_29 = None
    stabilize_hand_target_q     = None

    if stabilize_enabled:
        stab_policy_path = args.stabilize_policy or str(
            root / "src/holosoma_inference/holosoma_inference/models/wbt/base/master_policy.onnx"
        )
        print(f"Loading stabilization policy: {stab_policy_path}")
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

        # Static reference: hold init pose for all T frames
        stabilize_motion_dof_pos_29, _, stabilize_hand_target_q = \
            build_stabilization_refs(dof_pos_43, motion_length)

        print(f"  Stabilization: {stabilize_duration:.1f}s with static hold reference.")

    # ── Load object BPS code ───────────────────────────────────────────────
    bps_dir  = root / "src/holosoma/holosoma/data/objects_new/objects_new"
    bps_path = bps_dir / args.obj_name / f"{args.obj_name}_bps.pkl"
    if not bps_path.exists():
        raise FileNotFoundError(f"BPS file not found: {bps_path}")
    bps_data     = joblib.load(str(bps_path))
    obj_bps_code = bps_data["bps_code"]
    if hasattr(obj_bps_code, "numpy"):
        obj_bps_code = obj_bps_code.numpy()
    obj_bps = obj_bps_code.flatten().astype(np.float32)
    print(f"Loaded BPS code: {obj_bps.shape} for '{args.obj_name}'")

    obs_buf = OTTObsBuffer()

    # ── Initialize scene ───────────────────────────────────────────────────
    def _reset_scene():
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
            obj_yaw = float(np.random.uniform(-np.pi, np.pi))
            obj_quat = yaw_to_quat_wxyz(obj_yaw)
            print(
                f"  [Spawn rand] robot_xy={robot_pos[:2].round(3)}  "
                f"robot_yaw={np.degrees(robot_yaw_noise):+.1f}°  "
                f"table_z={table_pos[2]:.3f} (Δ{table_z_noise:+.3f})  "
                f"obj_xy={obj_pos[:2].round(3)}  obj_yaw={np.degrees(obj_yaw):+.1f}°",
                flush=True,
            )
        else:
            robot_pos  = root_pos_init
            robot_quat = root_quat_wxyz
            joint_q    = init_q
            table_pos  = TABLE_SPAWN
            obj_pos    = OBJ_SPAWN
            obj_quat   = OBJ_QUAT_WXYZ

        set_table_pose(model, table_pos, TABLE_QUAT_WXYZ)
        data.qpos[robot_qpos_addr:robot_qpos_addr + 3] = robot_pos
        data.qpos[robot_qpos_addr + 3:robot_qpos_addr + 7] = robot_quat
        data.qpos[dof_qpos_addrs] = joint_q
        data.qvel[:] = 0.0
        set_freejoint_pose(data, obj_qpos_addr, obj_pos, obj_quat)
        mujoco.mj_forward(model, data)
        return obj_pos

    _first_obj_spawn = _reset_scene()
    # Regenerate trajectory from the actual (possibly randomized) object spawn position
    if args.randomize:
        motion = generate_cabinet_trajectory(
            obj_spawn         = _first_obj_spawn,
            cabinet_pos       = cabinet_pos,
            cabinet_quat_wxyz = cabinet_quat,
            shelf             = shelf,
            motion_fps        = motion_fps,
            hold_sec          = args.hold_sec,
            tgt_hold_sec      = args.tgt_hold_sec,
            obj_rot0_wxyz     = OBJ_QUAT_WXYZ,
            speed_scale       = args.traj_speed_scale,
        )
        motion_length = motion["motion_length"]
        _viz_full_traj[:] = list(motion["obj_pos"][::_viz_sample_step])
        if stabilize_enabled:
            stabilize_motion_dof_pos_29, _, stabilize_hand_target_q = \
                build_stabilization_refs(dof_pos_43, motion_length)
    print(f"Scene initialized: obj={OBJ_SPAWN}  table={TABLE_SPAWN}  cabinet={list(cabinet_pos)}")

    # ── View-only mode ─────────────────────────────────────────────────────
    if getattr(args, "view_only", False):
        print("View-only mode: opening interactive viewer (no policy). Press ESC/Q to close.")
        import mujoco.viewer as _mj_viewer
        mujoco.mj_forward(model, data)
        _mj_viewer.launch(model, data)
        return

    # ── Video recorder ─────────────────────────────────────────────────────
    video_recorder = None
    if args.record or args.offscreen:
        video_recorder = SimpleVideoRecorder(
            model,
            name="test_cabinet_place_mw",
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
    stab_cmd_t      = 0
    stab_cmd_acc    = 0.0
    stab_last_a29   = np.zeros(29, dtype=np.float32)
    stabilize_active      = stabilize_enabled
    stabilize_start_sim_t = float(data.time)

    # ── Post-OTT phase precomputations ────────────────────────────────────────
    _left_thumb_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_thumb_0_link")
    _right_thumb_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_thumb_0_link")
    _left_hand_names  = [n for n in HAND_JOINT_NAMES if n.startswith("left_")]
    _right_hand_names = [n for n in HAND_JOINT_NAMES if n.startswith("right_")]
    _left_hand_dof_idx  = name_indices(_left_hand_names,  DOF_NAMES)
    _right_hand_dof_idx = name_indices(_right_hand_names, DOF_NAMES)
    _left_hand_open_q   = DEFAULT_DOF_ANGLES[_left_hand_dof_idx].copy()
    _right_hand_open_q  = DEFAULT_DOF_ANGLES[_right_hand_dof_idx].copy()
    # Index to read current body pose in G1_MOTION_JOINT_NAMES_29 order (= master policy cmd order)
    _motion_29_dof_idx  = name_indices(G1_MOTION_JOINT_NAMES_29, DOF_NAMES)

    # ── Post-OTT phase state ──────────────────────────────────────────────────
    release_with_hotdex          = getattr(args, "release_with_hotdex", False)
    hotdex_when_post_release     = getattr(args, "hotdex_when_post_release", False)
    HAND_OPEN_DURATION      = 1.0
    POST_HOLD_DURATION      = 2.0
    hand_open_active        = False
    post_hold_active        = False
    post_hold_start_sim_t   = 0.0
    hand_open_start_sim_t   = 0.0
    hand_open_start_q_7     = np.zeros(7, dtype=np.float32)   # active hand joints at phase start
    hand_open_target_q_7    = np.zeros(7, dtype=np.float32)   # active hand open target
    hand_open_active_idx    = _left_hand_dof_idx.copy()       # DOF_NAMES indices for active hand
    final_body_cmd_29       = np.zeros(29, dtype=np.float32)  # final body pose as master policy cmd

    _viz_short_pos: list = []
    _viz_long_pos:  list = []

    print(f"\nStarting placement inference: {motion_length} frames = "
          f"{motion_length / motion_fps:.2f}s  |  stabilize={stabilize_duration:.1f}s")

    def _on_policy_step():
        nonlocal target_q, motion_time_acc, motion_timestep, policy_step
        nonlocal stabilize_active, stab_last_a29, stab_cmd_t, stab_cmd_acc, stabilize_start_sim_t
        nonlocal hand_open_active, hand_open_start_sim_t
        nonlocal post_hold_active, post_hold_start_sim_t
        nonlocal hand_open_start_q_7, hand_open_target_q_7, hand_open_active_idx, final_body_cmd_29

        # ── Phase 1: Stabilization ───────────────────────────────────────────
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

        # ── Phase 3: Hand opening ────────────────────────────────────────────
        # Body held by master policy (or hotdex); active hand opened via PD.
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

            # Override active hand joints only: interpolate from grasp → open
            target_q[hand_open_active_idx] = (hand_open_start_q_7
                                               + alpha * (hand_open_target_q_7 - hand_open_start_q_7))

            if t_elapsed >= HAND_OPEN_DURATION:
                hand_open_active      = False
                post_hold_active      = True
                post_hold_start_sim_t = float(data.time)
                print(f"  [Hand open] Complete — holding for {POST_HOLD_DURATION:.1f}s.", flush=True)
            return

        # ── Phase 4: Post-hold (record 2s after release) ─────────────────────
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
                post_hold_active = False
                print(f"  [Post-hold] Done.", flush=True)
            return

        # ── Phase 2: OTT inference ────────────────────────────────────────────
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
            full_q = data.qpos[dof_qpos_addrs].astype(np.float32)
            # Detect active hand: whichever thumb_0_link is closer to the object
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
        nonlocal hand_open_active, post_hold_active
        nonlocal motion, motion_length, stabilize_motion_dof_pos_29, stabilize_hand_target_q
        policy_step = motion_timestep = 0
        motion_time_acc = stab_cmd_acc = 0.0
        stab_cmd_t       = 0
        hand_open_active = False
        post_hold_active = False
        obs_buf.reset()
        stab_last_a29[:] = 0.0
        obj_spawn = _reset_scene()
        # Regenerate trajectory from the actual (possibly randomized) object spawn position
        if args.randomize:
            motion = generate_cabinet_trajectory(
                obj_spawn         = obj_spawn,
                cabinet_pos       = cabinet_pos,
                cabinet_quat_wxyz = cabinet_quat,
                shelf             = shelf,
                motion_fps        = motion_fps,
                hold_sec          = args.hold_sec,
                tgt_hold_sec      = args.tgt_hold_sec,
                obj_rot0_wxyz     = OBJ_QUAT_WXYZ,
                speed_scale       = args.traj_speed_scale,
            )
            motion_length = motion["motion_length"]
            _viz_full_traj[:] = list(motion["obj_pos"][::_viz_sample_step])
            if stabilize_enabled:
                stabilize_motion_dof_pos_29, _, stabilize_hand_target_q = \
                    build_stabilization_refs(dof_pos_43, motion_length)
        target_q = data.qpos[dof_qpos_addrs].astype(np.float32).copy()
        stabilize_active      = stabilize_enabled
        stabilize_start_sim_t = float(data.time)
        print(f"\n  [Reset] shelf={shelf}", flush=True)

    def _should_stop():
        if stabilize_active or hand_open_active or post_hold_active:
            return False
        return motion_timestep >= motion_length - 1

    _ID_MAT     = np.eye(3).flatten().astype(np.float64)
    _SHORT_RGBA = np.array([0.2, 0.9, 0.2, 0.8], dtype=np.float32)
    _LONG_RGBA  = np.array([0.9, 0.2, 0.2, 0.8], dtype=np.float32)
    _FULL_RGBA  = np.array([0.2, 0.8, 0.9, 0.6], dtype=np.float32)  # cyan, static full path
    _PT_SIZE    = np.array([0.015, 0.0, 0.0], dtype=np.float64)
    _PT_SIZE_SM = np.array([0.010, 0.0, 0.0], dtype=np.float64)

    def _add_traj_markers(scn):
        # Static full-path preview (cyan, drawn first so live markers overlay)
        for pos in _viz_full_traj:
            if scn.ngeom >= scn.maxgeom:
                return
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                _PT_SIZE_SM, np.asarray(pos, dtype=np.float64),
                                _ID_MAT, _FULL_RGBA)
            scn.ngeom += 1
        # Live lookahead markers
        for pts, rgba in ((_viz_short_pos, _SHORT_RGBA), (_viz_long_pos, _LONG_RGBA)):
            for pos in pts:
                if scn.ngeom >= scn.maxgeom:
                    return
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                    _PT_SIZE, np.asarray(pos, dtype=np.float64),
                                    _ID_MAT, rgba)
                scn.ngeom += 1

    try:
        run_mujoco_loop(
            model, data,
            sim_dt=sim_dt,
            steps_per_policy=steps_per_policy,
            on_policy_step=_on_policy_step,
            apply_ctrl=_apply_ctrl,
            on_reset=_on_reset,
            should_stop=_should_stop if args.offscreen else None,
            video_recorder=video_recorder,
            offscreen=args.offscreen,
            on_render=_add_traj_markers,
        )
    finally:
        if video_recorder is not None:
            video_recorder.save(fps=1.0 / policy_dt)
            video_recorder.cleanup()

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Cabinet placement: spawn robot+object+cabinet, generate trajectory, "
                    "run stabilization then OTT BPS policy."
    )
    parser.add_argument(
        "--obj-name", required=True,
        help="Object name (e.g. cubemedium). Determines robot XML and BPS code.",
    )
    parser.add_argument(
        "--shelf", type=int, choices=[1, 2, 3, 4], default=None,
        help="Cabinet shelf to target: 1=bottom … 4=top. Default: random.",
    )
    parser.add_argument(
        "--cabinet-pos", type=float, nargs=3, default=list(CABINET_SPAWN_DEFAULT),
        metavar=("X", "Y", "Z"),
        help="Cabinet spawn position. Default: 0.0 1.0 0.0.",
    )
    parser.add_argument(
        "--cabinet-quat-wxyz", type=float, nargs=4, default=list(CABINET_QUAT_WXYZ),
        metavar=("W", "X", "Y", "Z"),
        help="Cabinet spawn quaternion (wxyz). Default: 180° around Z (0 0 0 1).",
    )
    parser.add_argument(
        "--cabinet-z-offset", type=float, default=None,
        metavar="DZ",
        help="Fixed cabinet z offset (m). If omitted, sampled uniformly from --cabinet-z-range.",
    )
    parser.add_argument(
        "--cabinet-z-range", type=float, nargs=2, default=[-0.4, -0.05],
        metavar=("LO", "HI"),
        help="Range [lo, hi] for random cabinet z offset. Default: -0.25 -0.05.",
    )
    parser.add_argument(
        "--cabinet-xml", type=str,
        default="src/holosoma/holosoma/data/objects/open_cabinet.xml",
        help="Path to cabinet MJCF XML.",
    )
    parser.add_argument(
        "--motion-fps", type=float, default=120.0,
        help="Trajectory sampling frequency in Hz. Default: 120.",
    )
    parser.add_argument(
        "--traj-speed-scale", type=float, default=1.0,
        metavar="S",
        help="Scale factor for trajectory speed (lift/move/enter). <1 = slower. Default: 1.0.",
    )
    parser.add_argument(
        "--hold-sec", type=float, default=2.5,
        help="Seconds to hold at spawn before moving to cabinet. Default: 2.5.",
    )
    parser.add_argument(
        "--tgt-hold-sec", type=float, default=1.0,
        help="Seconds to hold at cabinet shelf after arrival. Default: 1.0.",
    )
    # Robot init pose (defaults = values from the initial state)
    parser.add_argument(
        "--robot-root-pos", type=float, nargs=3,
        default=list(DEFAULT_ROBOT_ROOT_POS),
        metavar=("X", "Y", "Z"),
        help="Robot root position [x y z]. Default: from initial state.",
    )
    parser.add_argument(
        "--robot-root-quat-xyzw", type=float, nargs=4,
        default=list(DEFAULT_ROBOT_ROOT_QUAT_XYZW),
        metavar=("X", "Y", "Z", "W"),
        help="Robot root quaternion (xyzw). Default: from initial state.",
    )
    parser.add_argument(
        "--robot-dof-pos", type=float, nargs=43,
        default=list(DEFAULT_ROBOT_DOF_POS_43),
        metavar="Q",
        help="43-DOF joint positions in G1_DEX3_JOINT_NAMES order. Default: from initial state.",
    )
    parser.add_argument(
        "--policy", default=None,
        help="Path to OTT ONNX policy (default: models/wbt/object/bps_policy.onnx).",
    )
    parser.add_argument("--sim-hz",    type=int, default=200)
    parser.add_argument("--policy-hz", type=int, default=50)
    parser.add_argument(
        "--stabilize-sec", "--stabilize_sec", dest="stabilize_sec",
        type=float, default=0.0,
        help="Seconds to run master_policy stabilization before OTT. Default: 0.",
    )
    parser.add_argument(
        "--stabilize-policy", default=None,
        help="Path to master_policy.onnx for stabilization.",
    )
    parser.add_argument("--record",    action="store_true")
    parser.add_argument("--offscreen", action="store_true",
                        help="Run headless, record video, stop when trajectory ends.")
    parser.add_argument("--release-with-hotdex", action="store_true",
                        help="Use hotdex (OTT) policy during hand-open phase "
                             "instead of master policy. Hand PD targets are still overwritten.")
    parser.add_argument("--hotdex-when-post-release", action="store_true",
                        help="Use hotdex (OTT) policy during post-hold phase "
                             "instead of master policy.")
    parser.add_argument("--view-only", action="store_true",
                        help="Open interactive viewer with scene loaded; no policy runs.")
    parser.add_argument("--video-dir",    type=str, default="logs/videos")
    parser.add_argument("--video-width",  type=int, default=1920)
    parser.add_argument("--video-height", type=int, default=1080)
    parser.add_argument("--video-format", type=str, default="h264", choices=["h264", "mp4"])
    parser.add_argument("--camera-pos",    type=float, nargs=3, default=[-2.0, 0.5, 1.5])
    parser.add_argument("--camera-target", type=float, nargs=3, default=[0.5, 0.0, 1.0])
    parser.add_argument(
        "--randomize", action="store_true",
        help="Randomize cabinet xy position (r∈[0.5,2.0], θ∈[-90°,90°]) and orient to face "
             "origin with ±45° noise. Also randomizes robot xy ±0.1 m, robot yaw ±0.5 rad, "
             "object xy ±0.2 m, object yaw ±180°, and joint positions ±0.1 rad each reset.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

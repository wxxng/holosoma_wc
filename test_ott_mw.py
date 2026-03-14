#!/usr/bin/env python3
"""
MuJoCo object trajectory tracking (OTT) policy test.

Loads a motion clip PKL containing robot + object + table data, initializes
the MuJoCo scene (g1_43dof_cylindermedium.xml) from the first frame, then runs
the OTT policy in a simulation loop.

Observation layout (1152 dims): task (obj pos/rot, 5-step hist) +
proprio_hand (joint pos/vel, 5-step hist) + proprio_body (joint pos/vel/angvel/gravity,
5-step hist) + command (BPS code + short/long-horizon object trajectory).

Usage:
    python test_ott_mw.py --pkl <motion_clip.pkl> [--clip-key <key>]
    python test_ott_mw.py --pkl <motion_clip.pkl> --stabilize-sec 1.0
    python test_ott_mw.py --pkl <motion_clip.pkl> --record
    python test_ott_mw.py --pkl <motion_clip.pkl> --offscreen
    python test_ott_mw.py --pkl <motion_clip.pkl> --sim-hz 200 --policy-hz 50
"""

import argparse
from pathlib import Path

import joblib
import mujoco
import numpy as np
import onnxruntime
import torch

from g1_robot_common import (
    DOF_NAMES, DEFAULT_DOF_ANGLES,
    ACTION_ORDER_43DOF, ACTION_SCALE, ACTION_OFFSET, ACTION_CLIP_MIN, ACTION_CLIP_MAX,
    BODY_JOINT_NAMES, HAND_JOINT_NAMES,
    name_indices, mj_hinge_addrs, mj_actuator_ids, apply_pd_control,
    quat_rotate_inverse, SimpleVideoRecorder, run_mujoco_loop,
)

# Precomputed index maps (config order → body/hand subsets)
BODY_INDICES = name_indices(BODY_JOINT_NAMES, DOF_NAMES)   # (29,)
HAND_INDICES = name_indices(HAND_JOINT_NAMES, DOF_NAMES)   # (14,)

np.set_printoptions(precision=4, suppress=True)

def _quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    return np.roll(np.asarray(q_xyzw, dtype=np.float32), shift=1, axis=-1)


# 43-DOF motion clip joint order (GRAB / DEX3 retargeting output)
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

# Stabilization (master_policy) observation input order (29 body joints)
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

# Stabilization (master_policy) action output order (29 body joints)
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

# Body-only motion command sequence order (29 DOF)
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
# Motion clip loading
# ═══════════════════════════════════════════════════════════════════════════

def load_motion_clip(pkl_path: str, clip_key: str | None = None) -> dict:
    """Load OTT motion clip PKL.

    Expected PKL format:
        {clip_key: {fps, dof_pos, global_translation_extend,
                    global_rotation_extend, obj_name, obj_pos, obj_rot,
                    table_pos, table_rot, contact_mask}}

    Returns dict with:
        clip_key        : str
        motion_fps      : float
        motion_length   : int
        dof_pos         : np.ndarray [T, 43]   43-DOF in G1_DEX3_JOINT_NAMES order
        init_q          : np.ndarray [43]       first-frame joints in DOF_NAMES config order
        root_pos_init   : np.ndarray [3]        world-frame root position at frame 0
        root_quat_init  : np.ndarray [4]        root quaternion at frame 0 (wxyz, converted from xyzw)
        obj_pos         : np.ndarray [T, 3]     object world position per frame
        obj_name        : str                    object name (e.g. "cylindermedium")
        obj_pos         : np.ndarray [T, 3]     object world position per frame
        obj_rot         : np.ndarray [T, 4]     object quaternion per frame (wxyz, converted from xyzw)
        table_pos       : np.ndarray [3]        table world position (constant)
        table_rot       : np.ndarray [4]        table quaternion (wxyz, converted from xyzw)
    """
    raw = joblib.load(pkl_path)
    if not isinstance(raw, dict):
        raise TypeError(f"Motion PKL must be a dict of clips, got {type(raw)}")

    if clip_key is None:
        clip_key = next(iter(raw.keys()))
        print(f"No clip_key specified, using first clip: '{clip_key}'")
    elif clip_key not in raw:
        raise KeyError(f"clip_key '{clip_key}' not found. Available: {list(raw.keys())}")

    clip = raw[clip_key]
    motion_fps    = float(clip["fps"])
    dof_pos       = np.asarray(clip["dof_pos"], dtype=np.float32)   # [T, 43]
    motion_length = dof_pos.shape[0]

    obj_name = clip["obj_name"]
    if isinstance(obj_name, (list, np.ndarray)):
        obj_name = obj_name[0]
    obj_name = str(obj_name).strip()

    print(f"Motion clip '{clip_key}': {motion_length} frames @ {motion_fps}Hz, {dof_pos.shape[1]} DOF, obj='{obj_name}'")

    # Root pose at frame 0: [frame, body_joint=0, dim]
    # global_translation_extend: [T, J, 3]
    # global_rotation_extend:    [T, J, 4] — quaternion stored as xyzw, converted to wxyz.
    global_trans = np.asarray(clip["global_translation_extend"], dtype=np.float32)
    global_rot   = np.asarray(clip["global_rotation_extend"],   dtype=np.float32)
    root_pos_init  = global_trans[0, 0, :]              # [3]  xyz world position
    root_quat_init = _quat_xyzw_to_wxyz(global_rot[0, 0, :])   # xyzw → wxyz (MuJoCo conv.)
    # Full root pose sequences for all frames (used by --init-timestep)
    root_pos_all  = global_trans[:, 0, :].copy()                  # [T, 3]
    root_quat_all = _quat_xyzw_to_wxyz(global_rot[:, 0, :])       # [T, 4] wxyz

    print(f"  global_translation_extend shape: {global_trans.shape}")
    print(f"  global_rotation_extend    shape: {global_rot.shape}")
    print(f"  root_quat_init (wxyz)           : {root_quat_init}")

    # Initial joint config reordered to DOF_NAMES (MuJoCo config order)
    config_idx = name_indices(G1_DEX3_JOINT_NAMES, DOF_NAMES)
    init_q = DEFAULT_DOF_ANGLES.copy()
    init_q[config_idx] = dof_pos[0]

    # Object trajectory per frame — obj_rot stored as xyzw, convert to wxyz for MuJoCo
    obj_pos = np.asarray(clip["obj_pos"], dtype=np.float32)          # [T, 3]
    obj_rot = _quat_xyzw_to_wxyz(np.asarray(clip["obj_rot"], dtype=np.float32))  # [T, 4] xyzw→wxyz

    # Table pose (fixed — take first frame if time-varying)
    # table_rot stored as xyzw, convert to wxyz for MuJoCo
    table_pos = np.asarray(clip["table_pos"], dtype=np.float32)
    table_rot = _quat_xyzw_to_wxyz(np.asarray(clip["table_rot"], dtype=np.float32))
    if table_pos.ndim == 2:
        table_pos = table_pos[0]
    if table_rot.ndim == 2:
        table_rot = table_rot[0]

    print(f"  obj_pos shape   : {obj_pos.shape}")
    print(f"  obj_rot shape   : {obj_rot.shape}")
    print(f"  table_pos       : {table_pos}")

    return {
        "clip_key":       clip_key,
        "motion_fps":     motion_fps,
        "motion_length":  motion_length,
        "dof_pos":        dof_pos,
        "config_idx":     config_idx,
        "init_q":         init_q,
        "root_pos_init":  root_pos_init,
        "root_quat_init": root_quat_init,
        "root_pos_all":   root_pos_all,
        "root_quat_all":  root_quat_all,
        "obj_name":       obj_name,
        "obj_pos":        obj_pos,
        "obj_rot":        obj_rot,
        "table_pos":      table_pos,
        "table_rot":      table_rot,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Scene helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_freejoint_qpos_addr(model: mujoco.MjModel, body_name: str) -> int:
    """Return the qpos start address of a freejoint attached to the named body."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise KeyError(f"Body '{body_name}' not found in model.")
    jnt_adr = int(model.body_jntadr[body_id])
    if jnt_adr < 0:
        raise RuntimeError(f"Body '{body_name}' has no joints.")
    return int(model.jnt_qposadr[jnt_adr])


def get_freejoint_qvel_addr(model: mujoco.MjModel, body_name: str) -> int:
    """Return the qvel start address of a freejoint attached to the named body."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise KeyError(f"Body '{body_name}' not found in model.")
    jnt_adr = int(model.body_jntadr[body_id])
    if jnt_adr < 0:
        raise RuntimeError(f"Body '{body_name}' has no joints.")
    return int(model.jnt_dofadr[jnt_adr])


def set_table_pose(model: mujoco.MjModel, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
    """Update table fixed-body pose in the model (no freejoint — modify model geometry)."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    if body_id < 0:
        raise KeyError("Body 'table' not found in model.")
    model.body_pos[body_id]  = pos.astype(np.float64)
    model.body_quat[body_id] = quat_wxyz.astype(np.float64)   # MuJoCo wxyz


def set_freejoint_pose(
    data: mujoco.MjData,
    qpos_addr: int,
    pos: np.ndarray,
    quat_wxyz: np.ndarray,
) -> None:
    """Set freejoint pose (pos + wxyz quaternion) in data.qpos."""
    data.qpos[qpos_addr:qpos_addr + 3] = pos
    data.qpos[qpos_addr + 3:qpos_addr + 7] = quat_wxyz   # MuJoCo: wxyz


# ═══════════════════════════════════════════════════════════════════════════
# Math helpers for OTT observation
# ═══════════════════════════════════════════════════════════════════════════

def quat_to_rot_mat(q_wxyz: np.ndarray) -> np.ndarray:
    """Unit quaternion (wxyz) → 3×3 rotation matrix."""
    w, x, y, z = q_wxyz.astype(np.float64)
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ], dtype=np.float32)


def heading_quat(q_wxyz: np.ndarray) -> np.ndarray:
    """Extract yaw-only quaternion (wxyz) from full body quaternion.

    The heading frame is the world frame rotated only by the robot's yaw angle,
    removing pitch and roll.
    """
    w, x, y, z = q_wxyz
    fwd_x = 1.0 - 2.0 * (y*y + z*z)
    fwd_y = 2.0 * (x*y + w*z)
    yaw = np.arctan2(fwd_y, fwd_x)
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float32)


def rot_mat_to_6d(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → 6D representation.

    Matches training convention: R[:, :2].flatten() (row-major) →
    [R00, R01, R10, R11, R20, R21]
    """
    return R[:, :2].flatten().astype(np.float32)


def obj_pos_rel_heading(
    obj_world: np.ndarray,
    root_world: np.ndarray,
    hquat: np.ndarray,
) -> np.ndarray:
    """Object position relative to robot root, expressed in heading frame."""
    return quat_rotate_inverse(hquat, obj_world - root_world)


def obj_rot_rel_heading(
    obj_quat_wxyz: np.ndarray,
    hquat: np.ndarray,
) -> np.ndarray:
    """Object orientation in torso heading frame, as 6D rotation representation."""
    R_h   = quat_to_rot_mat(hquat)
    R_obj = quat_to_rot_mat(obj_quat_wxyz)
    return rot_mat_to_6d(R_h.T @ R_obj)


# ═══════════════════════════════════════════════════════════════════════════
# Observation history buffer
# ═══════════════════════════════════════════════════════════════════════════

class OTTObsBuffer:
    """5-step history buffer for OTT task and proprio observations.

    Each attribute stores rows [oldest, ..., newest] (oldest at index 0,
    newest at index -1).  Call push() each policy step, then read the
    flattened slabs directly in build_ott_obs().
    """

    HIST = 5

    def __init__(self):
        self.obj_pos     = np.zeros((self.HIST,  3), dtype=np.float32)  # task
        self.obj_rot_6d  = np.zeros((self.HIST,  6), dtype=np.float32)  # task
        self.hand_jpos   = np.zeros((self.HIST, 14), dtype=np.float32)  # proprio_hand
        self.hand_jvel   = np.zeros((self.HIST, 14), dtype=np.float32)  # proprio_hand
        self.body_jpos   = np.zeros((self.HIST, 29), dtype=np.float32)  # proprio_body
        self.body_jvel   = np.zeros((self.HIST, 29), dtype=np.float32)  # proprio_body
        self.body_angvel = np.zeros((self.HIST,  3), dtype=np.float32)  # proprio_body
        self.body_pgrav  = np.zeros((self.HIST,  3), dtype=np.float32)  # proprio_body

    def push(
        self,
        obj_pos, obj_rot_6d,
        hand_jpos, hand_jvel,
        body_jpos, body_jvel, body_angvel, body_pgrav,
    ):
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
        for buf in (
            self.obj_pos, self.obj_rot_6d,
            self.hand_jpos, self.hand_jvel,
            self.body_jpos, self.body_jvel, self.body_angvel, self.body_pgrav,
        ):
            buf[:] = 0.0


def _as_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _split_pose_wxyz(pose_wxyz) -> tuple[np.ndarray, np.ndarray]:
    pose = _as_numpy(pose_wxyz).astype(np.float64).reshape(-1)
    if pose.size != 7:
        raise ValueError(f"Expected pose_wxyz to have 7 elements [x y z w x y z], got shape {pose.shape}")
    pos = pose[:3].astype(np.float32)
    quat = pose[3:7].astype(np.float32)
    return pos, quat


def _reorder_joints_to_config(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    joint_order: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Map debug joint arrays in joint_order -> DOF_NAMES config order."""
    if len(joint_order) == 0:
        raise ValueError("joint_order is empty in debug robot_state.")
    joint_pos = _as_numpy(joint_pos).astype(np.float32).reshape(-1)
    joint_vel = _as_numpy(joint_vel).astype(np.float32).reshape(-1)
    if joint_pos.size != len(joint_order) or joint_vel.size != len(joint_order):
        raise ValueError(
            f"joint_pos/joint_vel size mismatch: pos={joint_pos.size}, vel={joint_vel.size}, order={len(joint_order)}"
        )

    cfg_pos = DEFAULT_DOF_ANGLES.copy()
    cfg_vel = np.zeros_like(cfg_pos)

    cfg_idx = name_indices(joint_order, DOF_NAMES)
    if len(cfg_idx) != len(joint_order):
        raise RuntimeError("name_indices returned unexpected length for joint_order.")
    cfg_pos[cfg_idx] = joint_pos
    cfg_vel[cfg_idx] = joint_vel
    return cfg_pos, cfg_vel


def _get_first_present(d: dict, keys: list[str]):
    for k in keys:
        if k in d:
            return k, d[k]
    return None, None

def _extract_debug_motion_timestep(entry: dict, robot_state: dict) -> int | None:
    if "motion_timestep" in entry:
        return int(entry["motion_timestep"])
    if "motion_timestep" in robot_state:
        return int(robot_state["motion_timestep"])
    return None


def _apply_debug_robot_state_to_mujoco(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_state: dict,
    *,
    dof_qpos_addrs: np.ndarray,
    dof_qvel_addrs: np.ndarray,
    robot_qpos_addr: int,
    robot_qvel_addr: int,
    obj_qpos_addr: int,
    obj_qvel_addr: int,
) -> None:
    data.qvel[:] = 0.0
    if "table_pose_wxyz" in robot_state:
        tpos, tquat = _split_pose_wxyz(robot_state["table_pose_wxyz"])
        set_table_pose(model, tpos, tquat)

    rpos, rquat = _split_pose_wxyz(robot_state["root_pose_wxyz"])
    data.qpos[robot_qpos_addr:robot_qpos_addr + 3] = rpos
    data.qpos[robot_qpos_addr + 3:robot_qpos_addr + 7] = rquat

    cfg_pos, cfg_vel = _reorder_joints_to_config(
        robot_state["joint_pos"],
        robot_state["joint_vel"],
        list(robot_state["joint_order"]),
    )
    data.qpos[dof_qpos_addrs] = cfg_pos
    data.qvel[dof_qvel_addrs] = cfg_vel

    data.qvel[robot_qvel_addr:robot_qvel_addr + 6] = 0.0
    data.qvel[obj_qvel_addr:obj_qvel_addr + 6] = 0.0
    if robot_qvel_addr != robot_qpos_addr:
        data.qvel[robot_qpos_addr:robot_qpos_addr + 6] = 0.0

    if "root_lin_vel" in robot_state:
        root_lin_vel = _as_numpy(robot_state["root_lin_vel"]).astype(np.float32).reshape(3)
        data.qvel[robot_qvel_addr:robot_qvel_addr + 3] = root_lin_vel
    if "root_ang_vel" in robot_state:
        root_ang_vel = _as_numpy(robot_state["root_ang_vel"]).astype(np.float32).reshape(3)
        data.qvel[robot_qvel_addr + 3:robot_qvel_addr + 6] = root_ang_vel
        if robot_qvel_addr != robot_qpos_addr:
            data.qvel[robot_qpos_addr + 3:robot_qpos_addr + 6] = root_ang_vel

    obj_pose_key, obj_pose_val = _get_first_present(robot_state, ["object_pose_wxyz", "obj_pose_wxyz"])
    if obj_pose_key is not None:
        opos, oquat = _split_pose_wxyz(obj_pose_val)
        set_freejoint_pose(data, obj_qpos_addr, opos, oquat)

    obj_lin_key, obj_lin_val = _get_first_present(robot_state, ["object_lin_vel", "obj_vel_lin"])
    if obj_lin_key is not None:
        o_lin = _as_numpy(obj_lin_val).astype(np.float32).reshape(3)
        data.qvel[obj_qvel_addr:obj_qvel_addr + 3] = o_lin

    obj_ang_key, obj_ang_val = _get_first_present(robot_state, ["object_ang_vel", "obj_vel_ang"])
    if obj_ang_key is not None:
        o_ang = _as_numpy(obj_ang_val).astype(np.float32).reshape(3)
        data.qvel[obj_qvel_addr + 3:obj_qvel_addr + 6] = o_ang

    mujoco.mj_forward(model, data)

def _print_obs_error(label: str, ref: np.ndarray, got: np.ndarray) -> None:
    ref = _as_numpy(ref).astype(np.float32).reshape(-1)
    got = _as_numpy(got).astype(np.float32).reshape(-1)
    if ref.shape != got.shape:
        print(f"[{label}] shape mismatch: ref={ref.shape} got={got.shape}")
        return
    diff = got - ref
    absdiff = np.abs(diff)
    max_idx = int(absdiff.argmax()) if absdiff.size else 0
    print(
        f"[{label}] abs_err: mean={float(absdiff.mean()):.6g} "
        f"max={float(absdiff.max()):.6g} @idx={max_idx} "
        f"(ref={float(ref[max_idx]):.6g}, got={float(got[max_idx]):.6g})"
    )


def _print_obs_error_blocks(label: str, ref: np.ndarray, got: np.ndarray) -> None:
    ref = _as_numpy(ref).astype(np.float32).reshape(-1)
    got = _as_numpy(got).astype(np.float32).reshape(-1)
    if ref.size != 1152 or got.size != 1152:
        _print_obs_error(label, ref, got)
        return

    # build_ott_obs concatenation order:
    # task(45) + hand(140) + body(320) + command(647) = 1152
    # command(647): bps(512) + short_pos(30) + short_ori(60) + long_pos(15) + long_ori(30)
    CMD0 = 45 + 140 + 320   # 505
    blocks = [
        ("task",           0,       45),
        ("hand",          45,      185),
        ("body",         185,      505),
        ("cmd",          505,     1152),
        ("cmd/bps",      505,     1017),   # 512
        ("cmd/short_pos",1017,    1047),   # 10×3=30
        ("cmd/short_ori",1047,    1107),   # 10×6=60
        ("cmd/long_pos", 1107,    1122),   # 5×3=15
        ("cmd/long_ori", 1122,    1152),   # 5×6=30
    ]
    _print_obs_error(label, ref, got)
    for bname, lo, hi in blocks:
        _print_obs_error(f"{label}/{bname}", ref[lo:hi], got[lo:hi])


# ═══════════════════════════════════════════════════════════════════════════
# Observation builder
# ═══════════════════════════════════════════════════════════════════════════

def build_ott_obs(
    data: mujoco.MjData,
    dof_qpos_addrs: np.ndarray,
    dof_qvel_addrs: np.ndarray,
    robot_qpos_addr: int,
    obj_qpos_addr: int,
    torso_body_id: int,
    motion: dict,
    motion_timestep: int,
    obs_buf: OTTObsBuffer,
    obj_bps: np.ndarray,
    obj_pos_seq: np.ndarray | None = None,       # (10, 3) world-frame short-horizon, overrides motion["obj_pos"]
    obj_rot_seq: np.ndarray | None = None,       # (10, 4) wxyz short-horizon
    obj_pos_seq_long: np.ndarray | None = None,  # (5, 3) world-frame long-horizon
    obj_rot_seq_long: np.ndarray | None = None,  # (5, 4) wxyz long-horizon
) -> np.ndarray:
    """Build the OTT policy observation vector.

    Layout (total = 45 + 140 + 320 + 647 = 1152 dims):
        Task        ( 45): obj_pos_hist(5×3=15) + obj_rot_6d_hist(5×6=30)
        Proprio_hand(140): hand_jpos_hist(5×14=70) + hand_jvel_hist(5×14=70)
        Proprio_body(320): body_jpos_hist(5×29=145) + body_jvel_hist(5×29=145)
                           + body_angvel_hist(5×3=15) + body_pgrav_hist(5×3=15)
        Command     (647): obj_bps(512)
                           + motion_obj_pos(10×3=30) + motion_obj_ori(10×6=60)
                           + motion_obj_pos_long(5×3=15) + motion_obj_ori_long(5×6=30)

    All object positions / orientations are expressed in the torso heading frame
    (world frame rotated only by the robot's yaw angle).

    Returns:
        np.ndarray [1, obs_dim]
    """
    motion_length = motion["motion_length"]

    # ── Robot state ────────────────────────────────────────────────────────
    joint_pos = data.qpos[dof_qpos_addrs].astype(np.float32)       # (43,) config order
    joint_vel = data.qvel[dof_qvel_addrs].astype(np.float32)       # (43,)
    base_quat = data.qpos[robot_qpos_addr + 3:robot_qpos_addr + 7].astype(np.float32)  # wxyz
    base_ang_vel = data.qvel[robot_qpos_addr + 3:robot_qpos_addr + 6].astype(np.float32)

    torso_pos  = data.xpos[torso_body_id].astype(np.float32)   # world-frame pos from mj_forward
    torso_quat = data.xquat[torso_body_id].astype(np.float32)  # world-frame wxyz from mj_forward
    hquat = heading_quat(torso_quat)

    # Angular velocity in body frame (consistent with loco policy)
    # base_ang_vel = quat_rotate_inverse(base_quat, base_ang_vel_world)
    # Projected gravity in body frame
    proj_grav = quat_rotate_inverse(base_quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))

    # ── Object state ───────────────────────────────────────────────────────
    obj_pos_world  = data.qpos[obj_qpos_addr:obj_qpos_addr + 3].astype(np.float32)
    obj_quat_world = data.qpos[obj_qpos_addr + 3:obj_qpos_addr + 7].astype(np.float32)  # wxyz

    cur_obj_pos    = obj_pos_rel_heading(obj_pos_world, torso_pos, hquat)
    cur_obj_rot_6d = obj_rot_rel_heading(obj_quat_world, hquat)

    # ── Proprio features ───────────────────────────────────────────────────
    joint_pos_rel = joint_pos - DEFAULT_DOF_ANGLES
    body_jpos = joint_pos_rel[BODY_INDICES]   # (29,)
    body_jvel = joint_vel[BODY_INDICES]        # (29,)
    hand_jpos = joint_pos_rel[HAND_INDICES]    # (14,)
    hand_jvel = joint_vel[HAND_INDICES]        # (14,)

    # ── Push to history buffer (oldest at index 0, newest at index -1) ─────
    obs_buf.push(
        cur_obj_pos, cur_obj_rot_6d,
        hand_jpos, hand_jvel,
        body_jpos, body_jvel, base_ang_vel, proj_grav,
    )

    # ── Command: future object trajectory in heading frame ─────────────────
    if obj_pos_seq is not None and obj_rot_seq is not None:
        # Use pre-computed sequences from robot_state (world-frame → heading frame)
        mo_pos = np.array([
            obj_pos_rel_heading(obj_pos_seq[i], torso_pos, hquat)
            for i in range(len(obj_pos_seq))
        ])  # (10, 3)
        mo_ori = np.array([
            obj_rot_rel_heading(obj_rot_seq[i], hquat)
            for i in range(len(obj_rot_seq))
        ])  # (10, 6)
    else:
        # Short-horizon: t+1 .. t+10
        short_idx = np.clip(motion_timestep + np.arange(1, 11), 0, motion_length - 1)
        mo_pos = np.array([
            obj_pos_rel_heading(motion["obj_pos"][i], torso_pos, hquat)
            for i in short_idx
        ])  # (10, 3)
        mo_ori = np.array([
            obj_rot_rel_heading(motion["obj_rot"][i], hquat)
            for i in short_idx
        ])  # (10, 6)

    if obj_pos_seq_long is not None and obj_rot_seq_long is not None:
        # Use pre-computed long-horizon sequences from robot_state
        mo_pos_long = np.array([
            obj_pos_rel_heading(obj_pos_seq_long[i], torso_pos, hquat)
            for i in range(len(obj_pos_seq_long))
        ])  # (5, 3)
        mo_ori_long = np.array([
            obj_rot_rel_heading(obj_rot_seq_long[i], hquat)
            for i in range(len(obj_rot_seq_long))
        ])  # (5, 6)
    else:
        # Long-horizon: t+20, t+40, t+60, t+80, t+100
        long_idx = np.clip(motion_timestep + np.array([20, 40, 60, 80, 100]), 0, motion_length - 1)
        mo_pos_long = np.array([
            obj_pos_rel_heading(motion["obj_pos"][i], torso_pos, hquat)
            for i in long_idx
        ])  # (5, 3)
        mo_ori_long = np.array([
            obj_rot_rel_heading(motion["obj_rot"][i], hquat)
            for i in long_idx
        ])  # (5, 6)

    # ── Assemble final observation ─────────────────────────────────────────
    obs = np.concatenate([
        # Task (45)
        obs_buf.obj_pos.flatten(),       # 5×3  = 15
        obs_buf.obj_rot_6d.flatten(),    # 5×6  = 30
        # Proprio_hand (140)
        obs_buf.hand_jpos.flatten(),     # 5×14 = 70
        obs_buf.hand_jvel.flatten(),     # 5×14 = 70
        # Proprio_body (320)
        obs_buf.body_jpos.flatten(),     # 5×29 = 145
        obs_buf.body_jvel.flatten(),     # 5×29 = 145
        obs_buf.body_angvel.flatten(),   # 5×3  = 15
        obs_buf.body_pgrav.flatten(),    # 5×3  = 15
        # Command (647)
        obj_bps,                         # 512
        mo_pos.flatten(),                # 10×3 = 30
        mo_ori.flatten(),                # 10×6 = 60
        mo_pos_long.flatten(),           # 5×3  = 15
        mo_ori_long.flatten(),           # 5×6  = 30
    ])

    return obs.reshape(1, -1).astype(np.float32)  # (1, 1152)


def build_stabilization_motion_cmd_seq(
    motion_dof_pos_29: np.ndarray,
    motion_dof_vel_29: np.ndarray,
    timestep: int,
    motion_length: int,
) -> np.ndarray:
    """Build master-policy motion command sequence [1, 580] at the given timestep."""
    indices = np.clip(np.arange(timestep, timestep + 10), 0, motion_length - 1)
    pos_seq = motion_dof_pos_29[indices]  # [10, 29]
    vel_seq = motion_dof_vel_29[indices]  # [10, 29]
    cmd_seq = np.concatenate([pos_seq, vel_seq], axis=1)  # [10, 58]
    return cmd_seq.reshape(1, -1).astype(np.float32)


def build_stabilization_obs(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    base_quat_wxyz: np.ndarray,
    base_ang_vel: np.ndarray,
    last_action_29: np.ndarray,
    motion_cmd_seq: np.ndarray,
    obs_idx: np.ndarray,
) -> np.ndarray:
    """Build the 673-dim master-policy observation used in test_wbt_mw.py."""
    dof_pos_rel = joint_pos[obs_idx] - DEFAULT_DOF_ANGLES[obs_idx]
    dof_vel = joint_vel[obs_idx]
    proj_grav = quat_rotate_inverse(base_quat_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float32))
    obs = np.concatenate([dof_pos_rel, dof_vel, base_ang_vel, proj_grav, last_action_29, motion_cmd_seq.flatten()])
    return obs.reshape(1, -1).astype(np.float32)


def save_debug_stabilie_snapshot(
    save_path: str,
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    dof_qpos_addrs: np.ndarray,
    dof_qvel_addrs: np.ndarray,
    robot_qpos_addr: int,
    robot_qvel_addr: int,
    obj_qpos_addr: int,
    source_debug_file: str | None,
) -> Path:
    """Save initialization state for test_wbt.py --debug_stabilie_file."""
    out_path = Path(save_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    root_pose_wxyz = data.qpos[robot_qpos_addr:robot_qpos_addr + 7].astype(np.float32).copy()
    root_lin_vel = data.qvel[robot_qvel_addr:robot_qvel_addr + 3].astype(np.float32).copy()
    root_ang_vel = data.qvel[robot_qvel_addr + 3:robot_qvel_addr + 6].astype(np.float32).copy()
    joint_pos = data.qpos[dof_qpos_addrs].astype(np.float32).copy()
    joint_vel = data.qvel[dof_qvel_addrs].astype(np.float32).copy()
    object_pose_wxyz = data.qpos[obj_qpos_addr:obj_qpos_addr + 7].astype(np.float32).copy()

    table_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    table_pose_wxyz = None
    if table_body_id >= 0:
        table_pose_wxyz = np.concatenate([
            model.body_pos[table_body_id].astype(np.float32),
            model.body_quat[table_body_id].astype(np.float32),
        ])
    else:
        table_pose_wxyz = np.zeros(7, dtype=np.float32)

    np.savez_compressed(
        str(out_path),
        root_pose_wxyz=root_pose_wxyz,
        root_lin_vel=root_lin_vel,
        root_ang_vel=root_ang_vel,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        joint_order=np.asarray(DOF_NAMES),
        object_pose_wxyz=object_pose_wxyz,
        table_pose_wxyz=table_pose_wxyz,
        source_debug_file=np.asarray("" if source_debug_file is None else str(source_debug_file)),
    )
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Main simulation
# ═══════════════════════════════════════════════════════════════════════════

def run(args):
    root = Path(__file__).parent
    pkl_path = args.pkl

    # ── Load motion clip ───────────────────────────────────────────────────
    motion        = load_motion_clip(pkl_path, args.clip_key)
    motion_fps    = motion["motion_fps"]
    motion_length = motion["motion_length"]

    # ── Resolve XML from obj_name ──────────────────────────────────────────
    xml_dir  = root / "src/holosoma/holosoma/data/robots/g1/g1_object"
    if args.debug_file is not None:
        motion['obj_name'] = 'apple'
    xml_path = str(xml_dir / f"g1_43dof_{motion['obj_name']}.xml")
    if not Path(xml_path).exists():
        raise FileNotFoundError(
            f"No XML found for obj_name='{motion['obj_name']}': {xml_path}\n"
            f"Available XMLs: {sorted(p.name for p in xml_dir.glob('g1_43dof_*.xml'))}"
        )
    print(f"XML: {xml_path}")

    policy_path = args.policy or str(
        root / "src/holosoma_inference/holosoma_inference/models/wbt/object/bps_policy.onnx"
    )

    # ── Load ONNX policy ───────────────────────────────────────────────────
    print(f"Loading policy: {policy_path}")
    session = onnxruntime.InferenceSession(policy_path)
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    print(f"  Input:  {inp.name} {inp.shape}")
    print(f"  Output: {out.name} {out.shape}")
    obs_dim = int(inp.shape[-1]) if inp.shape and inp.shape[-1] is not None else 1

    # ── Load MuJoCo model ──────────────────────────────────────────────────
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    model.opt.timestep = 1.0 / args.sim_hz
    sim_dt           = float(model.opt.timestep)
    steps_per_policy = args.sim_hz // args.policy_hz
    policy_dt        = sim_dt * steps_per_policy
    print(f"Model: {model.nq} qpos, {model.nv} qvel, {model.nu} actuators, dt={sim_dt}s")
    print(f"Physics: {1/sim_dt:.0f}Hz | Policy: {1/policy_dt:.0f}Hz ({steps_per_policy} substeps)")

    dof_qpos_addrs, dof_qvel_addrs = mj_hinge_addrs(model, DOF_NAMES)
    actuator_ids    = mj_actuator_ids(model, DOF_NAMES)
    obj_qpos_addr   = get_freejoint_qpos_addr(model, "object")
    robot_qpos_addr = get_freejoint_qpos_addr(model, "pelvis")   # robot root body
    obj_qvel_addr   = get_freejoint_qvel_addr(model, "object")
    robot_qvel_addr = get_freejoint_qvel_addr(model, "pelvis")
    torso_body_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    if torso_body_id < 0:
        raise KeyError("Body 'torso_link' not found in model.")
    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    obj_mass = model.body_mass[obj_body_id] if obj_body_id >= 0 else float("nan")
    print(f"  Robot freejoint qpos addr : {robot_qpos_addr}")
    print(f"  Object freejoint qpos addr: {obj_qpos_addr}")
    print(f"  Torso body id             : {torso_body_id}")
    print(f"  Object mass               : {obj_mass:.4f} kg")

    # Action index map: ACTION_ORDER_43DOF → DOF_NAMES (config order)
    act_idx = name_indices(ACTION_ORDER_43DOF, DOF_NAMES)

    # ── Optional stabilization resources (same policy+obs as wbt_mw) ─────
    stabilize_duration = float(getattr(args, "stabilize_sec", 0.0))
    if stabilize_duration < 0.0:
        raise ValueError(f"--stabilize-sec must be >= 0, got {stabilize_duration}")
    stabilize_enabled = stabilize_duration > 0.0
    stabilize_session = None
    stabilize_input_name = None
    stabilize_obs_idx = None
    stabilize_act_idx = None
    stabilize_act_scale_29 = None
    stabilize_act_offset_29 = None
    stabilize_act_clip_min_29 = None
    stabilize_act_clip_max_29 = None
    stabilize_hand_idx = None
    stabilize_motion_dof_pos_29 = None
    stabilize_motion_dof_vel_29 = None
    stabilize_hand_target_q = None
    if stabilize_enabled:
        stabilize_policy_path = args.stabilize_policy or str(
            root / "src/holosoma_inference/holosoma_inference/models/wbt/base/master_policy.onnx"
        )
        print(f"Loading stabilization policy: {stabilize_policy_path}")
        stabilize_session = onnxruntime.InferenceSession(stabilize_policy_path)
        stabilize_inp = stabilize_session.get_inputs()[0]
        stabilize_out = stabilize_session.get_outputs()[0]
        stabilize_input_name = stabilize_inp.name
        print(f"  Stabilization input:  {stabilize_inp.name} {stabilize_inp.shape}")
        print(f"  Stabilization output: {stabilize_out.name} {stabilize_out.shape}")

        stabilize_obs_idx = name_indices(STABILIZATION_OBS_ORDER_29DOF, DOF_NAMES)
        stabilize_act_idx = name_indices(STABILIZATION_ACTION_ORDER_29DOF, DOF_NAMES)
        stab_in_43 = name_indices(STABILIZATION_ACTION_ORDER_29DOF, ACTION_ORDER_43DOF)
        stabilize_act_scale_29 = ACTION_SCALE[stab_in_43]
        stabilize_act_offset_29 = ACTION_OFFSET[stab_in_43]
        stabilize_act_clip_min_29 = ACTION_CLIP_MIN[stab_in_43]
        stabilize_act_clip_max_29 = ACTION_CLIP_MAX[stab_in_43]
        stabilize_hand_idx = name_indices(HAND_JOINT_NAMES, DOF_NAMES)

        body_in_motion = name_indices(G1_MOTION_JOINT_NAMES_29, G1_DEX3_JOINT_NAMES)
        stabilize_motion_dof_pos_29 = motion["dof_pos"][:, body_in_motion]
        dof_pos_next = np.roll(stabilize_motion_dof_pos_29, -1, axis=0)
        dof_pos_next[-1] = stabilize_motion_dof_pos_29[-1]
        stabilize_motion_dof_vel_29 = (dof_pos_next - stabilize_motion_dof_pos_29) * motion_fps
        stabilize_motion_dof_vel_29[-1] = 0.0

        hand_in_motion = name_indices(HAND_JOINT_NAMES, G1_DEX3_JOINT_NAMES)
        stabilize_hand_target_q = motion["dof_pos"][:, hand_in_motion]

    # ── Load object BPS code ───────────────────────────────────────────────
    bps_dir  = root / "src/holosoma/holosoma/data/objects_new/objects_new"
    bps_path = bps_dir / motion["obj_name"] / f"{motion['obj_name']}_bps.pkl"
    if not bps_path.exists():
        raise FileNotFoundError(f"BPS file not found: {bps_path}")
    bps_data = joblib.load(str(bps_path))
    # bps_code is a torch.Tensor of shape [1, 512]
    obj_bps_code = bps_data["bps_code"]
    if hasattr(obj_bps_code, "numpy"):
        obj_bps_code = obj_bps_code.numpy()
    obj_bps = obj_bps_code.flatten().astype(np.float32)  # (512,)
    print(f"Loaded BPS code: {obj_bps.shape} for '{motion['obj_name']}'")

    # ── Observation history buffer ─────────────────────────────────────────
    obs_buf = OTTObsBuffer()

    # ── Always load debug file (for init and trajectory sequences) ─────────
    debug_path = Path(args.debug_file).expanduser() if args.debug_file else None
    debug_entries = None
    if debug_path and debug_path.exists():
        _dbg_raw = joblib.load(str(debug_path))
        if isinstance(_dbg_raw, dict) and "summary" in _dbg_raw and isinstance(_dbg_raw["summary"], list):
            debug_entries = _dbg_raw["summary"]
        elif isinstance(_dbg_raw, list):
            debug_entries = _dbg_raw
        else:
            raise TypeError(
                f"--debug-file must be a list of entries or a dict with key 'summary' (list). Got: {type(_dbg_raw)}"
            )
        del _dbg_raw
        print(f"Loaded debug_file: {debug_path} ({len(debug_entries)} entries)")
    elif args.debug_compare:
        raise FileNotFoundError(f"--debug-compare set but --debug-file not found: {debug_path}")

    if args.debug_compare:

        max_steps = args.debug_max_steps
        start_step = args.debug_start_step
        if start_step < 0:
            raise ValueError("--debug-start-step must be >= 0")
        if max_steps is not None and max_steps <= 0:
            raise ValueError("--debug-max-steps must be > 0 when provided")

        # Reset obs history so we replay entries sequentially.
        obs_buf.reset()
        if start_step != 0:
            print("[Debug] Warning: start_step != 0, history buffer won't match the saved obs unless you replay from 0.")

        # Precompute policy_dt (needed to infer motion_timestep when not saved).
        steps_per_policy = args.sim_hz // args.policy_hz
        policy_dt = (1.0 / args.sim_hz) * steps_per_policy
        act_idx = name_indices(ACTION_ORDER_43DOF, DOF_NAMES)

        print(f"\n[Debug] Comparing observations using --debug-file={debug_path}")
        print(f"[Debug] Entries: {len(debug_entries)}  start={start_step}  max={max_steps or 'ALL'}")

        video_recorder = None
        if args.record or args.offscreen:
            video_recorder = SimpleVideoRecorder(
                model,
                name="test_ott_mw_debug_compare",
                camera_pos=tuple(args.camera_pos),
                camera_target=tuple(args.camera_target),
                width=args.video_width,
                height=args.video_height,
                save_dir=args.video_dir,
                output_format=args.video_format,
            )
            print(f"[Debug] Video capture enabled: {args.video_width}x{args.video_height} (fps={args.policy_hz})")

        agg_abs_sum = 0.0
        agg_count = 0
        agg_max = 0.0
        agg_max_step = -1
        agg_max_idx = -1

        try:
            end_step = len(debug_entries) if max_steps is None else min(len(debug_entries), start_step + max_steps)
            for step in range(start_step, end_step):
                entry = debug_entries[step]
                if not isinstance(entry, dict):
                    raise TypeError(f"Debug entry at index {step} must be a dict, got {type(entry)}")

                ref_obs = entry["observations"]
                robot_state = entry["robot_state"]

                _apply_debug_robot_state_to_mujoco(
                    model,
                    data,
                    robot_state,
                    dof_qpos_addrs=dof_qpos_addrs,
                    dof_qvel_addrs=dof_qvel_addrs,
                    robot_qpos_addr=robot_qpos_addr,
                    robot_qvel_addr=robot_qvel_addr,
                    obj_qpos_addr=obj_qpos_addr,
                    obj_qvel_addr=obj_qvel_addr,
                )
                if video_recorder is not None:
                    video_recorder.capture_frame(data)

                motion_timestep = _extract_debug_motion_timestep(entry, robot_state)
                if motion_timestep is None:
                    t = float(step) * policy_dt
                    motion_timestep = min(int(t * motion_fps), motion_length - 1)

                # Extract trajectory sequences from robot_state (preferred command source)
                _pos_seq = robot_state["motion_obj_pos_seq"]
                _pos_long = robot_state["motion_obj_pos_seq_long"]
                _rot_seq = robot_state["motion_obj_rot_seq"]
                _rot_long = robot_state["motion_obj_rot_seq_long"]


                if step == start_step and any(x is None for x in (_pos_seq, _rot_seq, _pos_long, _rot_long)):
                    print(
                        "[Debug] Warning: motion_obj_*_seq keys missing in robot_state; "
                        "falling back to motion PKL trajectories for command features."
                    )

                if step == start_step and args.debug_bootstrap_hist:
                    _ = build_ott_obs(
                        data, dof_qpos_addrs, dof_qvel_addrs,
                        robot_qpos_addr, obj_qpos_addr, torso_body_id,
                        motion, motion_timestep,
                        obs_buf, obj_bps,
                        obj_pos_seq=_pos_seq, obj_rot_seq=_rot_seq,
                        obj_pos_seq_long=_pos_long, obj_rot_seq_long=_rot_long,
                    )
                    for buf in (
                        obs_buf.obj_pos, obs_buf.obj_rot_6d,
                        obs_buf.hand_jpos, obs_buf.hand_jvel,
                        obs_buf.body_jpos, obs_buf.body_jvel,
                        obs_buf.body_angvel, obs_buf.body_pgrav,
                    ):
                        buf[:] = buf[-1]

                got_obs = build_ott_obs(
                    data, dof_qpos_addrs, dof_qvel_addrs,
                    robot_qpos_addr, obj_qpos_addr, torso_body_id,
                    motion, motion_timestep,
                    obs_buf, obj_bps,
                    obj_pos_seq=_pos_seq, obj_rot_seq=_rot_seq,
                    obj_pos_seq_long=_pos_long, obj_rot_seq_long=_rot_long,
                )[0]

                _print_obs_error_blocks(f"step={step} motion_t={motion_timestep}", ref_obs, got_obs)

                # ── Action comparison: recorded vs policy re-inference ──────
                ref_raw_action = entry.get("actions", None)          # (43,) raw policy output
                ref_proc_action = entry.get("processed_actions", None)  # (1,43) scaled+clipped

                # Re-infer with got_obs (what our sim would produce)
                got_raw_action = session.run(None, {inp.name: got_obs.reshape(1, -1)})[0][0]  # (43,)
                got_proc_action = np.clip(
                    got_raw_action * ACTION_SCALE + ACTION_OFFSET,
                    ACTION_CLIP_MIN, ACTION_CLIP_MAX,
                )  # (43,)

                if ref_raw_action is not None:
                    _print_obs_error(
                        f"step={step} action/raw",
                        _as_numpy(ref_raw_action).flatten(),
                        got_raw_action,
                    )
                if ref_proc_action is not None:
                    _print_obs_error(
                        f"step={step} action/processed",
                        _as_numpy(ref_proc_action).flatten(),
                        got_proc_action,
                    )

                # ── Transition dynamics comparison ──────────────────────────
                # Apply the re-inferred action from this step, run physics,
                # then compare resulting joint pos/vel with next entry's state.
                next_step = step + 1
                if next_step < end_step:
                    next_rs = debug_entries[next_step].get("robot_state", {})
                    if "joint_pos" in next_rs and "joint_vel" in next_rs and "joint_order" in next_rs:
                        # Build target_q: start from current qpos, apply action in config order
                        dyn_target_q = data.qpos[dof_qpos_addrs].astype(np.float32).copy()
                        dyn_target_q[act_idx] = got_proc_action
                        # Run one policy period of physics substeps
                        for _ in range(steps_per_policy):
                            apply_pd_control(data, dyn_target_q, dof_qpos_addrs, dof_qvel_addrs, actuator_ids)
                            mujoco.mj_step(model, data)
                        sim_jpos = data.qpos[dof_qpos_addrs].astype(np.float32)  # (43,) config order
                        sim_jvel = data.qvel[dof_qvel_addrs].astype(np.float32)  # (43,)
                        # Get next entry's expected joint state in config order
                        next_cfg_pos, next_cfg_vel = _reorder_joints_to_config(
                            next_rs["joint_pos"], next_rs["joint_vel"], list(next_rs["joint_order"])
                        )
                        _print_obs_error(f"step={step} dyn/joint_pos", next_cfg_pos, sim_jpos)
                        _print_obs_error(f"step={step} dyn/joint_vel", next_cfg_vel, sim_jvel)

                if args.debug_break:
                    breakpoint()
                if ref_obs.shape == got_obs.shape:
                    absdiff = np.abs(got_obs - ref_obs)
                    agg_abs_sum += float(absdiff.sum())
                    agg_count += int(absdiff.size)
                    step_max = float(absdiff.max()) if absdiff.size else 0.0
                    if step_max > agg_max:
                        agg_max = step_max
                        agg_max_step = step
                        agg_max_idx = int(absdiff.argmax()) if absdiff.size else -1

            if agg_count > 0:
                print(
                    f"\n[Debug] Overall: mean_abs_err={agg_abs_sum / agg_count:.6g} "
                    f"max_abs_err={agg_max:.6g} (step={agg_max_step}, idx={agg_max_idx})"
                )
            print("[Debug] Done.\n")
        finally:
            if video_recorder is not None:
                video_recorder.save(fps=float(args.policy_hz), tag="debug_compare")
                video_recorder.cleanup()
        return

    # ── Resolve init timestep ──────────────────────────────────────────────
    init_t = int(getattr(args, "init_timestep", 0))
    if init_t < 0 or init_t >= motion_length:
        raise ValueError(
            f"--init-timestep {init_t} is out of range [0, {motion_length - 1}]"
        )
    if init_t != 0:
        print(f"Init timestep: {init_t} (overriding default frame 0)")

    # Pre-compute joint config for init_t
    def _init_q_at(t: int) -> np.ndarray:
        q = DEFAULT_DOF_ANGLES.copy()
        q[motion["config_idx"]] = motion["dof_pos"][t]
        return q

    # ── Initialize scene ───────────────────────────────────────────────────
    use_debug_init = debug_entries is not None and len(debug_entries) > 0
    if use_debug_init:
        _init_entry_idx = min(init_t, len(debug_entries) - 1)
        entry0 = debug_entries[_init_entry_idx]
        if not isinstance(entry0, dict) or "robot_state" not in entry0:
            raise TypeError(f"Debug entry at index {_init_entry_idx} must be a dict with key 'robot_state'.")
        _apply_debug_robot_state_to_mujoco(
            model,
            data,
            entry0["robot_state"],
            dof_qpos_addrs=dof_qpos_addrs,
            dof_qvel_addrs=dof_qvel_addrs,
            robot_qpos_addr=robot_qpos_addr,
            robot_qvel_addr=robot_qvel_addr,
            obj_qpos_addr=obj_qpos_addr,
            obj_qvel_addr=obj_qvel_addr,
        )
        print(f"Scene initialized from --debug-file entry {_init_entry_idx}.")
    else:
        # Initialize from motion clip at frame init_t
        set_table_pose(model, motion["table_pos"], motion["table_rot"])
        data.qpos[robot_qpos_addr:robot_qpos_addr + 3] = motion["root_pos_all"][init_t]
        data.qpos[robot_qpos_addr + 3:robot_qpos_addr + 7] = motion["root_quat_all"][init_t]
        data.qpos[dof_qpos_addrs] = _init_q_at(init_t)
        data.qvel[:] = 0.0
        set_freejoint_pose(data, obj_qpos_addr, motion["obj_pos"][init_t], motion["obj_rot"][init_t])
        mujoco.mj_forward(model, data)
        print(f"Scene initialized from motion clip frame {init_t}.")

    if args.save_debug_stabilie_file:
        saved = save_debug_stabilie_snapshot(
            args.save_debug_stabilie_file,
            model=model,
            data=data,
            dof_qpos_addrs=dof_qpos_addrs,
            dof_qvel_addrs=dof_qvel_addrs,
            robot_qpos_addr=robot_qpos_addr,
            robot_qvel_addr=robot_qvel_addr,
            obj_qpos_addr=obj_qpos_addr,
            source_debug_file=None if debug_path is None else str(debug_path),
        )
        print(f"Saved debug_stabilie snapshot: {saved.resolve()}", flush=True)

    if args.bootstrap_hist:
        seq_pos = seq_rot = seq_pos_long = seq_rot_long = None
        mt_for_boot = init_t
        if use_debug_init:
            seq_pos = entry0["robot_state"]["motion_obj_pos_seq"]
            seq_pos_long = entry0["robot_state"]["motion_obj_pos_seq_long"]
            seq_rot = entry0["robot_state"]["motion_obj_rot_seq"]
            seq_rot_long = entry0["robot_state"]["motion_obj_rot_seq_long"]
            mt_dbg = _extract_debug_motion_timestep(entry0, entry0["robot_state"])
            if mt_dbg is not None:
                mt_for_boot = mt_dbg
        _ = build_ott_obs(
            data, dof_qpos_addrs, dof_qvel_addrs,
            robot_qpos_addr, obj_qpos_addr, torso_body_id,
            motion, motion_timestep=mt_for_boot,
            obs_buf=obs_buf, obj_bps=obj_bps,
            obj_pos_seq=seq_pos,
            obj_rot_seq=seq_rot,
            obj_pos_seq_long=seq_pos_long,
            obj_rot_seq_long=seq_rot_long,
        )
        for buf in (
            obs_buf.obj_pos, obs_buf.obj_rot_6d,
            obs_buf.hand_jpos, obs_buf.hand_jvel,
            obs_buf.body_jpos, obs_buf.body_jvel,
            obs_buf.body_angvel, obs_buf.body_pgrav,
        ):
            buf[:] = buf[-1]
        # Re-assemble obs after fill so all history slots are reflected
        boot_obs_vec = build_ott_obs(
            data, dof_qpos_addrs, dof_qvel_addrs,
            robot_qpos_addr, obj_qpos_addr, torso_body_id,
            motion, motion_timestep=mt_for_boot,
            obs_buf=obs_buf, obj_bps=obj_bps,
            obj_pos_seq=seq_pos,
            obj_rot_seq=seq_rot,
            obj_pos_seq_long=seq_pos_long,
            obj_rot_seq_long=seq_rot_long,
        )[0]

        # ── Verify bootstrapped obs against saved reference ────────────────
        if use_debug_init and entry0 is not None and "observations" in entry0:
            ref_obs_boot = _as_numpy(entry0["observations"]).astype(np.float32).reshape(-1)
            print(f"\n  [Init/bootstrap_hist] Comparing built obs vs saved entry[{min(init_t, len(debug_entries) - 1)}]['observations']:")
            _print_obs_error_blocks(f"bootstrap init_t={init_t}", ref_obs_boot, boot_obs_vec)
        else:
            print("\n  [Init/bootstrap_hist] No saved reference obs — skipping verification.")

    # ── Video recorder ────────────────────────────────────────────────────
    video_recorder = None
    if args.record or args.offscreen:
        video_recorder = SimpleVideoRecorder(
            model,
            name="test_ott_mw",
            camera_pos=tuple(args.camera_pos),
            camera_target=tuple(args.camera_target),
            width=args.video_width,
            height=args.video_height,
            save_dir=args.video_dir,
            output_format=args.video_format,
        )
        print(f"Video recording enabled: {args.video_width}x{args.video_height}")

    # ── Simulation state ───────────────────────────────────────────────────
    last_raw_action = np.zeros(len(ACTION_ORDER_43DOF), dtype=np.float32)
    target_q        = data.qpos[dof_qpos_addrs].astype(np.float32).copy()
    policy_step     = 0
    motion_timestep = init_t
    motion_time_acc = init_t / motion_fps
    stabilize_cmd_timestep = init_t
    stabilize_cmd_time_acc = init_t / motion_fps
    stabilize_last_action_29 = np.zeros(29, dtype=np.float32)
    stabilize_active = stabilize_enabled
    stabilize_start_sim_time = float(data.time)
    if stabilize_enabled:
        print(
            f"Stabilization stage enabled: running wbt_mw master_policy/obs for {stabilize_duration:.1f}s "
            f"(--stabilize-sec) before OTT inference."
        )

    # ── Trajectory visualization state (world-frame positions) ────────────
    _viz_short_pos: list = []   # (10, 3) world-frame short-horizon points
    _viz_long_pos:  list = []   # (5,  3) world-frame long-horizon points

    # ── Object initial height for table-hide logic ─────────────────────────
    obj_init_z   = float(data.qpos[obj_qpos_addr + 2])
    table_hidden = False
    _TABLE_HIDDEN_POS  = np.array([0.0, 0.0, -10.0], dtype=np.float64)
    _table_orig_pos    = motion["table_pos"].copy()
    _table_orig_quat   = motion["table_rot"].copy()
    print(f"  Object init height (z)    : {obj_init_z:.4f} m  (table hides when obj_z > {obj_init_z + 0.1:.4f} m)")

    print(f"\nStarting OTT policy inference.")
    print(f"Motion: {motion_length} frames @ {motion_fps}Hz = {motion_length / motion_fps:.2f}s")
    print(f"obs_dim={obs_dim}  (expected 1152: task=45 + hand=140 + body=320 + cmd=647)")

    def _on_policy_step():
        nonlocal last_raw_action, target_q, motion_time_acc, motion_timestep, policy_step, table_hidden
        nonlocal stabilize_active, stabilize_last_action_29, stabilize_cmd_timestep, stabilize_cmd_time_acc

        # Optional stabilization stage: run same master-policy path as wbt_mw for 1s.
        if stabilize_active:
            joint_pos = data.qpos[dof_qpos_addrs].astype(np.float32)
            joint_vel = data.qvel[dof_qvel_addrs].astype(np.float32)
            base_quat_wxyz = data.qpos[robot_qpos_addr + 3:robot_qpos_addr + 7].astype(np.float32)
            base_ang_vel = data.qvel[robot_qvel_addr + 3:robot_qvel_addr + 6].astype(np.float32)

            motion_cmd_seq = build_stabilization_motion_cmd_seq(
                stabilize_motion_dof_pos_29, stabilize_motion_dof_vel_29,
                stabilize_cmd_timestep, motion_length,
            )
            obs_stab = build_stabilization_obs(
                joint_pos, joint_vel, base_quat_wxyz, base_ang_vel,
                stabilize_last_action_29, motion_cmd_seq, stabilize_obs_idx,
            )
            raw_action_29 = stabilize_session.run(None, {stabilize_input_name: obs_stab})[0][0]
            stabilize_last_action_29 = raw_action_29.copy()

            scaled_29 = np.clip(
                raw_action_29 * stabilize_act_scale_29 + stabilize_act_offset_29,
                stabilize_act_clip_min_29, stabilize_act_clip_max_29,
            )
            target_q = joint_pos.copy()
            target_q[stabilize_act_idx] = scaled_29
            t_clamp = min(stabilize_cmd_timestep, motion_length - 1)
            target_q[stabilize_hand_idx] = stabilize_hand_target_q[t_clamp]

            # Match wbt_mw-style command progression during stabilization stage.
            stabilize_cmd_time_acc += policy_dt
            stabilize_cmd_timestep = min(int(stabilize_cmd_time_acc * motion_fps), motion_length - 1)

            stabilize_elapsed = float(data.time) - stabilize_start_sim_time
            if stabilize_elapsed >= stabilize_duration:
                stabilize_active = False
                print(
                    f"  [t={float(data.time):.2f}s] Stabilization finished ({stabilize_duration:.1f}s) — "
                    "starting OTT policy inference.",
                    flush=True,
                )
            return

        # ── Observation ────────────────────────────────────────────────────
        seq_pos = seq_rot = seq_pos_long = seq_rot_long = None
        motion_t_for_obs = motion_timestep
        if debug_entries is not None:
            dbg_step = min(policy_step, len(debug_entries) - 1)
            dbg_entry = debug_entries[dbg_step]
            if isinstance(dbg_entry, dict) and "robot_state" in dbg_entry:
                rs = dbg_entry["robot_state"]
                seq_pos = rs["motion_obj_pos_seq"]
                seq_pos_long = rs["motion_obj_pos_seq_long"]
                seq_rot = rs["motion_obj_rot_seq"]
                seq_rot_long = rs["motion_obj_rot_seq_long"]
                mt_dbg = _extract_debug_motion_timestep(dbg_entry, rs)
                if mt_dbg is not None:
                    motion_t_for_obs = mt_dbg

        # ── Update trajectory viz state ────────────────────────────────────
        if seq_pos is not None:
            _viz_short_pos[:] = list(np.asarray(seq_pos, dtype=np.float32))
        else:
            short_idx = np.clip(motion_t_for_obs + np.arange(1, 11), 0, motion_length - 1)
            _viz_short_pos[:] = list(motion["obj_pos"][short_idx])
        if seq_pos_long is not None:
            _viz_long_pos[:] = list(np.asarray(seq_pos_long, dtype=np.float32))
        else:
            long_idx = np.clip(motion_t_for_obs + np.array([20, 40, 60, 80, 100]), 0, motion_length - 1)
            _viz_long_pos[:] = list(motion["obj_pos"][long_idx])

        obs = build_ott_obs(
            data, dof_qpos_addrs, dof_qvel_addrs,
            robot_qpos_addr, obj_qpos_addr, torso_body_id,
            motion, motion_t_for_obs,
            obs_buf, obj_bps,
            obj_pos_seq=seq_pos,
            obj_rot_seq=seq_rot,
            obj_pos_seq_long=seq_pos_long,
            obj_rot_seq_long=seq_rot_long,
        )

        # ── Policy inference ───────────────────────────────────────────────
        raw_action = session.run(None, {inp.name: obs})[0][0]   # [43]
        last_raw_action = raw_action.copy()

        # Scale + offset + clip (in ACTION_ORDER_43DOF), then reindex to config order
        scaled = np.clip(
            raw_action * ACTION_SCALE + ACTION_OFFSET,
            ACTION_CLIP_MIN, ACTION_CLIP_MAX,
        ) 
        target_q = data.qpos[dof_qpos_addrs].astype(np.float32).copy()
        target_q[act_idx] = scaled

        # Advance motion clock
        motion_time_acc += policy_dt
        motion_timestep  = min(int(motion_time_acc * motion_fps), motion_length - 1)
        policy_step     += 1

        # ── Table hide/show based on object height ─────────────────────────
        obj_z = float(data.qpos[obj_qpos_addr + 2])
        if not table_hidden and obj_z > obj_init_z + 0.1:
            set_table_pose(model, _TABLE_HIDDEN_POS, _table_orig_quat)
            table_hidden = True
            print(f"  [t={motion_time_acc:.2f}s] Object lifted {obj_z - obj_init_z:.3f}m — table hidden.", flush=True)

        if policy_step % 50 == 0:
            base_z = float(data.qpos[robot_qpos_addr + 2])
            print(
                f"  [t={motion_time_acc:6.2f}s] policy_step={policy_step} "
                f"motion_t={motion_timestep}/{motion_length}  "
                f"base_z={base_z:.3f}m  obj_z={obj_z:.3f}m",
                flush=True,
            )

    def _apply_ctrl(d):
        apply_pd_control(d, target_q, dof_qpos_addrs, dof_qvel_addrs, actuator_ids)

    def _on_reset():
        nonlocal last_raw_action, target_q, motion_timestep, motion_time_acc, policy_step, table_hidden
        nonlocal stabilize_last_action_29, stabilize_active, stabilize_start_sim_time
        nonlocal stabilize_cmd_timestep, stabilize_cmd_time_acc
        last_raw_action = np.zeros(len(ACTION_ORDER_43DOF), dtype=np.float32)
        motion_timestep = init_t
        motion_time_acc = init_t / motion_fps
        policy_step     = 0
        table_hidden    = False
        obs_buf.reset()
        set_table_pose(model, _table_orig_pos, _table_orig_quat)

        if use_debug_init:
            _reset_entry_idx = min(init_t, len(debug_entries) - 1)
            entry0 = debug_entries[_reset_entry_idx]
            _apply_debug_robot_state_to_mujoco(
                model,
                data,
                entry0["robot_state"],
                dof_qpos_addrs=dof_qpos_addrs,
                dof_qvel_addrs=dof_qvel_addrs,
                robot_qpos_addr=robot_qpos_addr,
                robot_qvel_addr=robot_qvel_addr,
                obj_qpos_addr=obj_qpos_addr,
                obj_qvel_addr=obj_qvel_addr,
            )
        else:
            data.qpos[robot_qpos_addr:robot_qpos_addr + 3] = motion["root_pos_all"][init_t]
            data.qpos[robot_qpos_addr + 3:robot_qpos_addr + 7] = motion["root_quat_all"][init_t]
            data.qpos[dof_qpos_addrs] = _init_q_at(init_t)
            data.qvel[:] = 0.0
            set_freejoint_pose(data, obj_qpos_addr, motion["obj_pos"][init_t], motion["obj_rot"][init_t])
            mujoco.mj_forward(model, data)

        target_q = data.qpos[dof_qpos_addrs].astype(np.float32).copy()
        stabilize_last_action_29[:] = 0.0
        stabilize_cmd_timestep = init_t
        stabilize_cmd_time_acc = init_t / motion_fps
        stabilize_active = stabilize_enabled
        stabilize_start_sim_time = float(data.time)
        if args.bootstrap_hist:
            seq_pos = seq_rot = seq_pos_long = seq_rot_long = None
            mt_for_boot = init_t
            if use_debug_init:
                rs0 = entry0["robot_state"]
                seq_pos = rs0["motion_obj_pos_seq"]
                seq_pos_long = rs0["motion_obj_pos_seq_long"]
                seq_rot = rs0["motion_obj_rot_seq"]
                seq_rot_long = rs0["motion_obj_rot_seq_long"]
                mt_dbg = _extract_debug_motion_timestep(entry0, rs0)
                if mt_dbg is not None:
                    mt_for_boot = mt_dbg
            _ = build_ott_obs(
                data, dof_qpos_addrs, dof_qvel_addrs,
                robot_qpos_addr, obj_qpos_addr, torso_body_id,
                motion, motion_timestep=mt_for_boot,
                obs_buf=obs_buf, obj_bps=obj_bps,
                obj_pos_seq=seq_pos,
                obj_rot_seq=seq_rot,
                obj_pos_seq_long=seq_pos_long,
                obj_rot_seq_long=seq_rot_long,
            )
            for buf in (
                obs_buf.obj_pos, obs_buf.obj_rot_6d,
                obs_buf.hand_jpos, obs_buf.hand_jvel,
                obs_buf.body_jpos, obs_buf.body_jvel,
                obs_buf.body_angvel, obs_buf.body_pgrav,
            ):
                buf[:] = buf[-1]
            # Re-assemble obs after fill so all history slots are reflected
            boot_obs_vec_r = build_ott_obs(
                data, dof_qpos_addrs, dof_qvel_addrs,
                robot_qpos_addr, obj_qpos_addr, torso_body_id,
                motion, motion_timestep=mt_for_boot,
                obs_buf=obs_buf, obj_bps=obj_bps,
                obj_pos_seq=seq_pos,
                obj_rot_seq=seq_rot,
                obj_pos_seq_long=seq_pos_long,
                obj_rot_seq_long=seq_rot_long,
            )[0]

            # ── Verify bootstrapped obs against saved reference ────────────
            if use_debug_init and entry0 is not None and "observations" in entry0:
                ref_obs_boot_r = _as_numpy(entry0["observations"]).astype(np.float32).reshape(-1)
                print(f"\n  [Reset/bootstrap_hist] Comparing built obs vs saved entry[{_reset_entry_idx}]['observations']:")
                _print_obs_error_blocks(f"reset bootstrap init_t={init_t}", ref_obs_boot_r, boot_obs_vec_r)
            else:
                print("\n  [Reset/bootstrap_hist] No saved reference obs — skipping verification.")
        if use_debug_init:
            print(f"\n  [Reset] Restarted from --debug-file entry {_reset_entry_idx}", flush=True)
        else:
            print(f"\n  [Reset] Restarted from motion frame {init_t}", flush=True)
        if stabilize_enabled:
            print(
                f"  [Reset] Stabilization stage re-armed for {stabilize_duration:.1f}s.",
                flush=True,
            )

    def _should_stop():
        if stabilize_active:
            return False
        if debug_entries is not None:
            return policy_step >= len(debug_entries) - 1
        return motion_timestep >= motion_length - 1

    _IDENTITY_MAT = np.eye(3).flatten().astype(np.float64)
    _SHORT_RGBA   = np.array([0.2, 0.9, 0.2, 0.8], dtype=np.float32)   # green
    _LONG_RGBA    = np.array([0.9, 0.2, 0.2, 0.8], dtype=np.float32)   # red
    _SHORT_SIZE   = np.array([0.015, 0.0, 0.0], dtype=np.float64)
    _LONG_SIZE    = np.array([0.015, 0.0, 0.0], dtype=np.float64)

    def _add_traj_markers(scn) -> None:
        max_g = scn.maxgeom
        for pts, size, rgba in (
            (_viz_short_pos, _SHORT_SIZE, _SHORT_RGBA),
            (_viz_long_pos,  _LONG_SIZE,  _LONG_RGBA),
        ):
            for pos in pts:
                if scn.ngeom >= max_g:
                    break
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    size,
                    np.asarray(pos, dtype=np.float64),
                    _IDENTITY_MAT,
                    rgba,
                )
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
        description="OTT (object trajectory tracking) policy inference in MuJoCo"
    )
    parser.add_argument(
        "--pkl", required=True,
        help="Path to motion clip PKL (dict with {fps, dof_pos, global_translation_extend, ...})",
    )
    parser.add_argument(
        "--clip-key", default=None,
        help="Clip key within the PKL (default: first key)",
    )
    parser.add_argument(
        "--policy", default=None,
        help="Path to OTT ONNX policy (default: src/holosoma_inference/holosoma_inference/models/wbt/object/bps_policy.onnx)",
    )
    parser.add_argument("--sim-hz",    type=int, default=200, help="Simulation frequency in Hz (default: 200)")
    parser.add_argument("--policy-hz", type=int, default=50,  help="Policy frequency in Hz (default: 50)")
    parser.add_argument(
        "--stabilize-sec", "--stabilize_sec",
        "--stablilize-sec", "--stablilize_sec",
        dest="stabilize_sec",
        type=float,
        default=0.0,
        help="Run wbt_mw-equivalent stabilization (master_policy + 673-dim obs) for this many seconds before OTT inference (default: 0).",
    )
    parser.add_argument(
        "--stabilize-policy",
        default=None,
        help="Path to stabilization master_policy.onnx (default: models/wbt/base/master_policy.onnx).",
    )
    parser.add_argument(
        "--init-timestep", dest="init_timestep", type=int, default=0,
        help="Motion clip frame index to initialize the scene from (default: 0)",
    )
    parser.add_argument(
        "--no-bootstrap-hist",
        dest="bootstrap_hist",
        action="store_false",
        help="Disable bootstrapping the 5-step history buffer on reset (default: enabled).",
    )
    parser.set_defaults(bootstrap_hist=True)
    parser.add_argument(
        "--debug-compare",
        action="store_true",
        help="Replay states from --debug-file, rebuild observation, and print errors (then exit).",
    )
    parser.add_argument(
        "--debug-file",
        type=str,
        # default="~/Desktop/codebase/unitreeG1_mw/holosoma/src/holosoma/holosoma/data/motions/g1_43dof/prior/object_o_a_full_state.pkl",
        default=None,
        help="If this PKL exists, replay saved robot/object state(s), rebuild obs, and print observation errors.",
    )
    parser.add_argument(
        "--save-debug-stabilie-file", "--save_debug_stabilie_file",
        dest="save_debug_stabilie_file",
        type=str,
        default=None,
        help="Save current init state (root/joint/object/table) to npz for test_wbt.py --debug_stabilie_file.",
    )
    parser.add_argument("--debug-start-step", type=int, default=0, help="Start index in debug-file entries (default: 0)")
    parser.add_argument("--debug-max-steps", type=int, default=None, help="Max number of debug steps to compare (default: all)")
    parser.add_argument(
        "--no-debug-bootstrap-hist",
        dest="debug_bootstrap_hist",
        action="store_false",
        help="Disable bootstrapping the 5-step history at the first debug step.",
    )
    parser.set_defaults(debug_bootstrap_hist=True)
    parser.add_argument(
        "--debug-break",
        action="store_true",
        help="Enter breakpoint() on each debug compare step (default: disabled).",
    )
    parser.add_argument("--record",    action="store_true", help="Record video while showing viewer.")
    parser.add_argument("--offscreen", action="store_true", help="Run headless, record video, stop when motion ends.")
    parser.add_argument("--video-dir",    type=str,   default="logs/videos")
    parser.add_argument("--video-width",  type=int,   default=1920)
    parser.add_argument("--video-height", type=int,   default=1080)
    parser.add_argument("--video-format", type=str,   default="h264", choices=["h264", "mp4"])
    parser.add_argument("--camera-pos",    type=float, nargs=3, default=[-2.0, 0.0, 1.5],
                        help="Camera position [x y z]")
    parser.add_argument("--camera-target", type=float, nargs=3, default=[1.0, 0.0, 1.0],
                        help="Camera look-at target [x y z]")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

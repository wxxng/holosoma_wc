"""
Shared constants, robot model, and actuator dynamics for G1-43DOF MuJoCo test scripts.

Imported by test_loco_mw.py, test_wbt_mw.py, etc.
"""

import subprocess
import time
from datetime import datetime
from pathlib import Path

import re

import mujoco
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# Joint naming and config order
# ═══════════════════════════════════════════════════════════════════════════

# 43-DOF config order = MuJoCo XML hinge-joint order
DOF_NAMES: list[str] = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
]

# Policy action order (same for locomotion and master/hotdex policies)
ACTION_ORDER_43DOF: list[str] = [
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
    "left_hand_index_0_joint", "left_hand_middle_0_joint", "left_hand_thumb_0_joint",
    "right_hand_index_0_joint", "right_hand_middle_0_joint", "right_hand_thumb_0_joint",
    "left_hand_index_1_joint", "left_hand_middle_1_joint", "left_hand_thumb_1_joint",
    "right_hand_index_1_joint", "right_hand_middle_1_joint", "right_hand_thumb_1_joint",
    "left_hand_thumb_2_joint", "right_hand_thumb_2_joint",
]

BODY_JOINT_NAMES: list[str] = [
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

HAND_JOINT_NAMES: list[str] = [
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
]

# ═══════════════════════════════════════════════════════════════════════════
# Motor constants  (from G1_DEX3_CFG actuator dynamics)
# ═══════════════════════════════════════════════════════════════════════════

ARMATURE_5020    = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010    = 0.00425

_NATURAL_FREQ  = 10 * 2.0 * np.pi   # 10 Hz
_DAMPING_RATIO = 2.0

STIFFNESS_5020    = ARMATURE_5020    * _NATURAL_FREQ ** 2
STIFFNESS_7520_14 = ARMATURE_7520_14 * _NATURAL_FREQ ** 2
STIFFNESS_7520_22 = ARMATURE_7520_22 * _NATURAL_FREQ ** 2
STIFFNESS_4010    = ARMATURE_4010    * _NATURAL_FREQ ** 2

DAMPING_5020    = 2.0 * _DAMPING_RATIO * ARMATURE_5020    * _NATURAL_FREQ
DAMPING_7520_14 = 2.0 * _DAMPING_RATIO * ARMATURE_7520_14 * _NATURAL_FREQ
DAMPING_7520_22 = 2.0 * _DAMPING_RATIO * ARMATURE_7520_22 * _NATURAL_FREQ
DAMPING_4010    = 2.0 * _DAMPING_RATIO * ARMATURE_4010    * _NATURAL_FREQ


def _resolve(patterns: dict, names: list[str], default: float = 0.0) -> np.ndarray:
    """Resolve a {regex_pattern: value} dict against joint names (first match wins)."""
    result = np.full(len(names), default, dtype=np.float32)
    for i, name in enumerate(names):
        for pattern, value in patterns.items():
            if re.fullmatch(pattern, name):
                result[i] = value
                break
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Robot defaults
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_DOF_ANGLES = _resolve({
    ".*_hip_pitch_joint":        -0.312,
    ".*_knee_joint":              0.669,
    ".*_ankle_pitch_joint":      -0.363,
    ".*_elbow_joint":             0.6,
    "left_shoulder_pitch_joint":  0.2,
    "left_shoulder_roll_joint":   0.2,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
}, DOF_NAMES)

# ═══════════════════════════════════════════════════════════════════════════
# Actuator dynamics  (all arrays in DOF_NAMES / config order)
# ═══════════════════════════════════════════════════════════════════════════

MOTOR_KP = _resolve({
    ".*_hip_pitch_joint":      STIFFNESS_7520_14,
    ".*_hip_yaw_joint":        STIFFNESS_7520_14,
    ".*_hip_roll_joint":       STIFFNESS_7520_22,
    ".*_knee_joint":           STIFFNESS_7520_22,
    ".*_ankle_.*_joint":       2*STIFFNESS_5020,
    "waist_yaw_joint":         STIFFNESS_7520_14,
    "waist_roll_joint":        2*STIFFNESS_5020,
    "waist_pitch_joint":       2*STIFFNESS_5020,
    ".*_shoulder_pitch_joint": STIFFNESS_5020,
    ".*_shoulder_roll_joint":  STIFFNESS_5020,
    ".*_shoulder_yaw_joint":   STIFFNESS_5020,
    ".*_elbow_joint":          STIFFNESS_5020,
    ".*_wrist_roll_joint":     STIFFNESS_5020,
    ".*_wrist_pitch_joint":    STIFFNESS_4010,
    ".*_wrist_yaw_joint":      STIFFNESS_4010,
    ".*_hand_thumb_0_joint":   2.0,
    ".*_hand_.*_0_joint":      0.5,
    ".*_hand_.*":              0.5,
}, DOF_NAMES)

MOTOR_KD = _resolve({
    ".*_hip_pitch_joint":      DAMPING_7520_14,
    ".*_hip_yaw_joint":        DAMPING_7520_14,
    ".*_hip_roll_joint":       DAMPING_7520_22,
    ".*_knee_joint":           DAMPING_7520_22,
    ".*_ankle_.*_joint":       2*DAMPING_5020,
    "waist_yaw_joint":         DAMPING_7520_14,
    "waist_roll_joint":        2*DAMPING_5020,
    "waist_pitch_joint":       2*DAMPING_5020,
    ".*_shoulder_pitch_joint": DAMPING_5020,
    ".*_shoulder_roll_joint":  DAMPING_5020,
    ".*_shoulder_yaw_joint":   DAMPING_5020,
    ".*_elbow_joint":          DAMPING_5020,
    ".*_wrist_roll_joint":     DAMPING_5020,
    ".*_wrist_pitch_joint":    DAMPING_4010,
    ".*_wrist_yaw_joint":      DAMPING_4010,
    ".*_hand_.*":              0.1,
}, DOF_NAMES)

# MOTOR_KP = np.array([
#     40.179, 99.098, 40.179, 99.098, 28.501, 28.501,  # left leg
#     40.179, 99.098, 40.179, 99.098, 28.501, 28.501,  # right leg
#     40.179, 28.501, 28.501,                           # waist
#     14.251, 14.251, 14.251, 14.251, 14.251, 16.778, 16.778,  # left arm
#      2.000,  0.500, 0.500, 0.500, 0.500, 0.500, 0.500,  # left hand
#     14.251, 14.251, 14.251, 14.251, 14.251, 16.778, 16.778,  # right arm
#      2.000,  0.500, 0.500, 0.500, 0.500, 0.500, 0.500,  # right hand
# ], dtype=np.float32)

# MOTOR_KD = np.array([
#     1.814, 1.814, 1.814, 1.814, 1.814, 1.814,  # left leg
#     1.814, 1.814, 1.814, 1.814, 1.814, 1.814,  # right leg
#     2.558, 1.814, 1.814,                        # waist
#     0.907, 0.907, 0.907, 0.907, 0.907, 1.068, 1.068,  # left arm
#     0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100,  # left hand
#     0.907, 0.907, 0.907, 0.907, 0.907, 1.068, 1.068,  # right arm
#     0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100,  # right hand
# ], dtype=np.float32)


# effort_limit_sim (G1_DEX3_CFG)
TORQUE_LIMITS = _resolve({
    ".*_hip_yaw_joint":        88.0,
    ".*_hip_pitch_joint":      88.0,
    ".*_hip_roll_joint":      139.0,
    ".*_knee_joint":          139.0,
    ".*_ankle_.*_joint":       50.0,
    "waist_yaw_joint":         88.0,
    "waist_roll_joint":        50.0,
    "waist_pitch_joint":       50.0,
    ".*_shoulder_pitch_joint": 25.0,
    ".*_shoulder_roll_joint":  25.0,
    ".*_shoulder_yaw_joint":   25.0,
    ".*_elbow_joint":          25.0,
    ".*_wrist_roll_joint":     25.0,
    ".*_wrist_pitch_joint":     5.0,
    ".*_wrist_yaw_joint":       5.0,
    ".*_hand_thumb_0_joint":   2.45,
    ".*_hand_.*":               1.4,
}, DOF_NAMES)

# velocity_limit_sim (G1_DEX3_CFG) — used for DC motor back-EMF model
VELOCITY_LIMITS = _resolve({
    ".*_hip_yaw_joint":        32.0,
    ".*_hip_pitch_joint":      32.0,
    ".*_hip_roll_joint":       20.0,
    ".*_knee_joint":           20.0,
    ".*_ankle_.*_joint":       37.0,
    "waist_yaw_joint":         32.0,
    "waist_roll_joint":        37.0,
    "waist_pitch_joint":       37.0,
    ".*_shoulder_pitch_joint": 37.0,
    ".*_shoulder_roll_joint":  37.0,
    ".*_shoulder_yaw_joint":   37.0,
    ".*_elbow_joint":          37.0,
    ".*_wrist_roll_joint":     37.0,
    ".*_wrist_pitch_joint":    22.0,
    ".*_wrist_yaw_joint":      22.0,
    ".*_hand_thumb_0_joint":   3.14,
    ".*_hand_.*":              12.0,
}, DOF_NAMES)

# ═══════════════════════════════════════════════════════════════════════════
# Policy action scaling  (in ACTION_ORDER_43DOF)
# ═══════════════════════════════════════════════════════════════════════════

_dof_idx = {n: i for i, n in enumerate(DOF_NAMES)}
_ao_idx  = np.array([_dof_idx[n] for n in ACTION_ORDER_43DOF], dtype=np.int64)

ACTION_SCALE    = (0.25 * TORQUE_LIMITS / MOTOR_KP)[_ao_idx]
ACTION_OFFSET   = DEFAULT_DOF_ANGLES[_ao_idx]
ACTION_CLIP_MIN = -(5.0 * TORQUE_LIMITS / MOTOR_KP)[_ao_idx]
ACTION_CLIP_MAX =  (5.0 * TORQUE_LIMITS / MOTOR_KP)[_ao_idx]

# ═══════════════════════════════════════════════════════════════════════════
# Index utilities
# ═══════════════════════════════════════════════════════════════════════════

def name_indices(names: list[str], reference: list[str]) -> np.ndarray:
    """Return the index of each name in *names* within *reference*.

    Example: name_indices(["b", "a"], ["a", "b", "c"]) → [1, 0]
    """
    lookup = {n: i for i, n in enumerate(reference)}
    return np.array([lookup[n] for n in names], dtype=np.int64)


# ═══════════════════════════════════════════════════════════════════════════
# MuJoCo address helpers
# ═══════════════════════════════════════════════════════════════════════════

def mj_hinge_addrs(
    model: mujoco.MjModel,
    joint_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (qpos_addrs, qvel_addrs) for hinge joints in *joint_names* order."""
    qpos = np.zeros(len(joint_names), dtype=np.int64)
    qvel = np.zeros(len(joint_names), dtype=np.int64)
    for i, name in enumerate(joint_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"Joint '{name}' not found in MJCF.")
        qpos[i] = int(model.jnt_qposadr[jid])
        qvel[i] = int(model.jnt_dofadr[jid])
    return qpos, qvel


def mj_actuator_ids(
    model: mujoco.MjModel,
    actuator_names: list[str],
) -> np.ndarray:
    """Return actuator IDs for *actuator_names*."""
    ids = np.zeros(len(actuator_names), dtype=np.int64)
    for i, name in enumerate(actuator_names):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise KeyError(f"Actuator '{name}' not found in MJCF.")
        ids[i] = int(aid)
    return ids


# ═══════════════════════════════════════════════════════════════════════════
# Actuator dynamics
# ═══════════════════════════════════════════════════════════════════════════

def apply_pd_control(
    data: mujoco.MjData,
    target_q: np.ndarray,
    dof_qpos_addrs: np.ndarray,
    dof_qvel_addrs: np.ndarray,
    actuator_ids: np.ndarray,
    *,
    verbose_vel_limit: bool = False,
) -> None:
    """Compute PD torques with DC motor back-EMF model and write to data.ctrl.

    tau_limit(vel) = TORQUE_LIMIT * max(0, 1 - |vel| / VEL_LIMIT)

    Args:
        verbose_vel_limit: if True, print a warning whenever a joint exceeds
                           its velocity limit.
    """
    curr_q = data.qpos[dof_qpos_addrs].astype(np.float32)
    curr_dq = data.qvel[dof_qvel_addrs].astype(np.float32)
    tau = MOTOR_KP * (target_q - curr_q) + MOTOR_KD * (0.0 - curr_dq)

    # vel_ratio = np.abs(curr_dq) / VELOCITY_LIMITS
    # if verbose_vel_limit:
    #     exceeded = vel_ratio > 1.0
    #     if exceeded.any():
    #         for i in np.where(exceeded)[0]:
    #             print(
    #                 f"[vel_limit] {DOF_NAMES[i]}: "
    #                 f"|vel|={abs(curr_dq[i]):.3f} > limit={VELOCITY_LIMITS[i]:.3f}",
    #                 flush=True,
    #             )

    # tau_limit = TORQUE_LIMITS * np.clip(1.0 - vel_ratio, 0.0, 1.0)
    # tau = np.clip(tau, -tau_limit, tau_limit)
    # tau = np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)
    data.ctrl[:] = 0.0
    data.ctrl[actuator_ids] = tau


# ═══════════════════════════════════════════════════════════════════════════
# Math utilities
# ═══════════════════════════════════════════════════════════════════════════

def quat_rotate(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector *v* by unit quaternion *q* (wxyz)."""
    w = q_wxyz[0]
    xyz = q_wxyz[1:4]
    t = 2.0 * np.cross(xyz, v)
    return (v + w * t + np.cross(xyz, t)).astype(np.float32)


def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector *v* by the inverse of unit quaternion *q* (wxyz).

    Equivalent to transforming a world-frame vector into the body frame.
    """
    w = q_wxyz[0]
    xyz = q_wxyz[1:4]
    t = 2.0 * np.cross(xyz, v)
    return (v - w * t + np.cross(xyz, t)).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Video recorder
# ═══════════════════════════════════════════════════════════════════════════

class SimpleVideoRecorder:
    """Lightweight offscreen MuJoCo video recorder.

    Args:
        name: base name used in the output filename (e.g. "test_loco_mw").
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        name: str = "test",
        width: int = 640,
        height: int = 360,
        camera_pos: tuple[float, float, float] = (3.0, 0.0, 2.0),
        camera_target: tuple[float, float, float] = (1.0, 0.0, 1.0),
        vertical_fov: float = 45.0,
        save_dir: str = "logs/videos",
        output_format: str = "h264",
    ) -> None:
        self.name = name
        self.model = model
        self.width = width
        self.height = height
        self.save_dir = Path(save_dir)
        self.output_format = output_format
        self.frames: list[np.ndarray] = []

        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self._camera)
        self._set_fixed_camera(camera_pos, camera_target)

        if hasattr(model, "vis") and hasattr(model.vis, "global_"):
            model.vis.global_.fovy = vertical_fov
        if hasattr(model, "vis") and hasattr(model.vis, "map"):
            model.vis.map.znear = 0.01 / model.stat.extent

    def _set_fixed_camera(self, position, target) -> None:
        offset = np.array(position) - np.array(target)
        distance = float(np.linalg.norm(offset))
        azimuth = float(np.degrees(np.arctan2(offset[1], offset[0])))
        elevation = float(np.degrees(np.arcsin(offset[2] / distance))) if distance > 0 else 0.0
        self._camera.lookat[:] = target
        self._camera.distance = distance
        self._camera.azimuth = azimuth
        self._camera.elevation = -elevation

    def capture_frame(self, data: mujoco.MjData, scene_fn=None, follow_robot: bool = True) -> None:
        if follow_robot:
            self._camera.lookat[:] = data.qpos[:3]  # follow robot base
        self._renderer.update_scene(data, camera=self._camera)
        if scene_fn is not None:
            scene_fn(self._renderer.scene)
        frame = self._renderer.render()
        if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 3:
            self.frames.append(frame.copy())

    def save(self, fps: float, tag: str = "") -> Path | None:
        import cv2

        if not self.frames:
            print("[video] No frames captured, skipping save.", flush=True)
            return None

        self.save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag_str = f"_{tag}" if tag else ""
        raw_path = self.save_dir / f"{self.name}{tag_str}_{timestamp}_raw.mp4"
        final_path = self.save_dir / f"{self.name}{tag_str}_{timestamp}.mp4"

        h, w = self.frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(raw_path), fourcc, fps, (w, h))
        for frame in self.frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"[video] Wrote {len(self.frames)} frames ({len(self.frames)/fps:.1f}s) to raw mp4", flush=True)

        if self.output_format == "h264":
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", str(raw_path),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "23", "-preset", "medium",
                        str(final_path),
                    ],
                    capture_output=True, text=True, check=True,
                )
                raw_path.unlink(missing_ok=True)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"[video] ffmpeg H.264 re-encode failed ({e}), keeping mp4v", flush=True)
                final_path = raw_path
        else:
            raw_path.rename(final_path)

        print(f"[video] Saved: {final_path.resolve()}", flush=True)
        return final_path

    def cleanup(self) -> None:
        self._renderer = None
        self._camera = None
        self.frames.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Simulation loop
# ═══════════════════════════════════════════════════════════════════════════

def run_mujoco_loop(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    sim_dt: float,
    steps_per_policy: int,
    on_policy_step,
    apply_ctrl,
    on_reset=None,
    should_stop=None,
    video_recorder=None,
    offscreen: bool = False,
    on_render=None,
    record_stride: int = 1,
) -> None:
    """Run a MuJoCo simulation loop, handling viewer/offscreen, real-time sync,
    video capture, key callbacks (Q=quit, R=reset), and physics stepping.

    Args:
        on_policy_step: callable() -> None
            Called every ``steps_per_policy`` physics steps, before apply_ctrl.
        apply_ctrl: callable(data) -> None
            Called every physics step to write data.ctrl (e.g. apply_pd_control).
        on_reset: callable() -> None | None
            Called when R key is pressed (viewer mode). Should reset MuJoCo state.
        should_stop: callable() -> bool | None
            When provided, the loop stops early when this returns True.
        video_recorder: SimpleVideoRecorder | None
            Frame captured every ``steps_per_policy * record_stride`` physics steps.
        offscreen: bool
            If True, run headless without a viewer window.
        record_stride: int
            Capture one video frame every this many policy steps (default 1).
    """
    physics_step = 0
    policy_frame = 0  # counts policy steps, used for record_stride
    done = False
    reset_requested = False

    def _key_callback(keycode):
        nonlocal done, reset_requested
        if keycode == ord("Q") or keycode == 256:
            done = True
        elif keycode == ord("R") and on_reset is not None:
            reset_requested = True

    try:
        if offscreen:
            print("Running headless (offscreen).\n")
            while not done:
                if should_stop is not None and should_stop():
                    break
                if physics_step % steps_per_policy == 0:
                    on_policy_step()
                apply_ctrl(data)
                mujoco.mj_step(model, data)
                physics_step += 1
                if physics_step % steps_per_policy == 0:
                    if video_recorder is not None and policy_frame % record_stride == 0:
                        video_recorder.capture_frame(data, scene_fn=on_render)
                    policy_frame += 1
        else:
            hint = "Press Q or ESC to quit"
            if on_reset is not None:
                hint += ", R to reset"
            print(hint + ".\n")
            import mujoco.viewer as _mj_viewer
            wall_next = time.perf_counter()
            with _mj_viewer.launch_passive(model, data, key_callback=_key_callback) as viewer:
                while viewer.is_running() and not done:
                    if reset_requested:
                        reset_requested = False
                        on_reset()
                        wall_next = time.perf_counter()
                        continue
                    if should_stop is not None and should_stop():
                        break
                    if physics_step % steps_per_policy == 0:
                        on_policy_step()
                    apply_ctrl(data)
                    mujoco.mj_step(model, data)
                    physics_step += 1
                    if physics_step % steps_per_policy == 0:
                        if on_render is not None:
                            viewer.user_scn.ngeom = 0
                            on_render(viewer.user_scn)
                        if video_recorder is not None and policy_frame % record_stride == 0:
                            video_recorder.capture_frame(data)
                        policy_frame += 1
                        viewer.sync()
                    wall_next += sim_dt
                    slack = wall_next - time.perf_counter()
                    if slack > 0:
                        time.sleep(slack)
                    else:
                        wall_next = time.perf_counter()
    except KeyboardInterrupt:
        print("\n  [Interrupted] Ctrl+C detected.")

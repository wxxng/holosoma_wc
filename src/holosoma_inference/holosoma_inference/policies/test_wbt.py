#!/usr/bin/env python3
"""
MuJoCo + ONNX WBT test with stabilization and hotdex motion tracking.

Loads the G1 robot in MuJoCo:
  - Runs master_policy.onnx at 50Hz for stabilization
  - Press 's' to switch to g1-43dof-hotdex policy for motion tracking
  - Press 's' again (while running) to toggle object-state freeze (matches wbt.py hotdex behavior)

Usage:
    python test_wbt.py
"""

import argparse
import hashlib
import math
import os
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import joblib
import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime

# Ensure repo `src/` is importable when running as `python test_wbt.py`.
_SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

try:
    from holosoma_inference.policies.wbt import (
        ACTION_ORDER_43DOF,
        G1_DEX3_JOINT_NAMES,
        G1_MOTION_JOINT_NAMES_29,
        MASTER_POLICY_ACTION_CLIP_MAX_43,
        MASTER_POLICY_ACTION_CLIP_MIN_43,
        MASTER_POLICY_ACTION_OFFSET_43,
        MASTER_POLICY_ACTION_SCALE_43,
        STABILIZATION_ACTION_ORDER_29DOF,
        STABILIZATION_OBS_ORDER_29DOF,
    )
except Exception as exc:  # noqa: BLE001
    raise RuntimeError(
        "Failed to import `holosoma_inference.policies.wbt`.\n"
        "Run from the repo root, or install the package, and ensure dependencies "
        "(e.g. pinocchio/onnxruntime) are available."
    ) from exc

# ═══════════════════════════════════════════════════════════════════════════
# Standalone video recorder (no holosoma simulator dependency)
# ═══════════════════════════════════════════════════════════════════════════

class SimpleVideoRecorder:
    """Lightweight MuJoCo video recorder for standalone test scripts.

    Uses a fixed camera position/target, renders offscreen via mujoco.Renderer,
    and saves to MP4 via OpenCV + optional ffmpeg H.264 re-encode.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        width: int = 640,
        height: int = 360,
        camera_pos: tuple[float, float, float] = (3.0, 0.0, 2.0),
        camera_target: tuple[float, float, float] = (1.0, 0.0, 1.0),
        vertical_fov: float = 45.0,
        save_dir: str = "logs/videos",
        output_format: str = "h264",
    ) -> None:
        self.model = model
        self.width = width
        self.height = height
        self.camera_pos = camera_pos
        self.camera_target = camera_target
        self.save_dir = Path(save_dir)
        self.output_format = output_format
        self.frames: list[np.ndarray] = []

        # Renderer
        self._renderer = mujoco.Renderer(model, height=height, width=width)

        # Camera (spherical coords derived from position + target)
        self._camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self._camera)
        self._set_fixed_camera(camera_pos, camera_target)

        # Override model FOV
        if hasattr(model, "vis") and hasattr(model.vis, "global_"):
            model.vis.global_.fovy = vertical_fov

        # Set near clipping plane to avoid robot clipping
        if hasattr(model, "vis") and hasattr(model.vis, "map"):
            extent = model.stat.extent
            model.vis.map.znear = 0.01 / extent

    def _set_fixed_camera(
        self,
        position: tuple[float, float, float],
        target: tuple[float, float, float],
    ) -> None:
        """Convert cartesian position/target to MuJoCo spherical camera params."""
        offset = np.array(position) - np.array(target)
        distance = float(np.linalg.norm(offset))
        azimuth = float(np.degrees(np.arctan2(offset[1], offset[0])))
        elevation = float(np.degrees(np.arcsin(offset[2] / distance))) if distance > 0 else 0.0

        self._camera.lookat[:] = target
        self._camera.distance = distance
        self._camera.azimuth = azimuth
        self._camera.elevation = -elevation

    def capture_frame(self, data: mujoco.MjData) -> None:
        """Render one frame and append to buffer."""
        self._renderer.update_scene(data, camera=self._camera)
        frame = self._renderer.render()
        if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 3:
            self.frames.append(frame.copy())

    def save(self, fps: float, tag: str = "") -> Path | None:
        """Encode buffered frames to MP4 and return the saved path."""
        import cv2

        if not self.frames:
            print("[video] No frames captured, skipping save.", flush=True)
            return None

        self.save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag_str = f"_{tag}" if tag else ""
        raw_path = self.save_dir / f"test_wbt{tag_str}_{timestamp}_raw.mp4"
        final_path = self.save_dir / f"test_wbt{tag_str}_{timestamp}.mp4"

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
        """Release renderer resources."""
        self._renderer = None
        self._camera = None
        self.frames.clear()


# ── Robot joint ordering (config order = MuJoCo XML order) ──────────────────
# 43 DOF total: legs(12) + waist(3) + left_arm(7) + left_hand(7) + right_arm(7) + right_hand(7)
DOF_NAMES = [
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

# Stabilization constants
STABILIZATION_OBS_ORDER = STABILIZATION_OBS_ORDER_29DOF
STABILIZATION_ACTION_ORDER = STABILIZATION_ACTION_ORDER_29DOF
ACTION_SCALE_43 = MASTER_POLICY_ACTION_SCALE_43
ACTION_OFFSET_43 = MASTER_POLICY_ACTION_OFFSET_43
ACTION_CLIP_MIN_43 = MASTER_POLICY_ACTION_CLIP_MIN_43
ACTION_CLIP_MAX_43 = MASTER_POLICY_ACTION_CLIP_MAX_43

# ── Default joint angles (43-DOF, config order) ────────────────────────────
DEFAULT_DOF_ANGLES = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # left leg (0-5)
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # right leg (6-11)
     0.0,   0.0, 0.0,                        # waist (12-14)
     0.2,   0.2, 0.0, 0.6, 0.0, 0.0, 0.0,   # left arm (15-21)
     0.0,   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # left hand (22-28)
     0.2,  -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,   # right arm (29-35)
     0.0,   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # right hand (36-42)
], dtype=np.float32)

# ── PD gains (43-DOF, config order) from g1-43dof-hotdex config ────────────
MOTOR_KP = np.array([
    40.179, 99.098, 40.179, 99.098, 28.501, 28.501,  # left leg
    40.179, 99.098, 40.179, 99.098, 28.501, 28.501,  # right leg
    40.179, 28.501, 28.501,                           # waist
    14.251, 14.251, 14.251, 14.251, 14.251, 16.778, 16.778,  # left arm
     2.000,  0.500, 0.500, 0.500, 0.500, 0.500, 0.500,  # left hand
    14.251, 14.251, 14.251, 14.251, 14.251, 16.778, 16.778,  # right arm
     2.000,  0.500, 0.500, 0.500, 0.500, 0.500, 0.500,  # right hand
], dtype=np.float32)

MOTOR_KD = np.array([
    1.814, 1.814, 1.814, 1.814, 1.814, 1.814,  # left leg
    1.814, 1.814, 1.814, 1.814, 1.814, 1.814,  # right leg
    2.558, 1.814, 1.814,                        # waist
    0.907, 0.907, 0.907, 0.907, 0.907, 1.068, 1.068,  # left arm
    0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100,  # left hand
    0.907, 0.907, 0.907, 0.907, 0.907, 1.068, 1.068,  # right arm
    0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100,  # right hand
], dtype=np.float32)

# ── Hotdex: hand and body joint names (matching wbt.py) ────────────────────
HAND_JOINT_NAMES = [
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
]

BODY_JOINT_NAMES = [
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

# ── Hotdex observation config ──────────────────────────────────────────────
HOTDEX_OBS_CONFIG = {
    "command": [("motion_obj_pos_rel_all", 42), ("motion_obj_ori_rel_all", 84)],
    "task": [("obj_pos_diff_b", 3), ("actions", 43)],
    "points": [("obj_pcd", 3072)],
    "command_points": [("motion_obj_pcd", 3072)],
    "proprio_body": [
        ("joint_pos_body", 29), ("joint_vel_body", 29),
        ("base_lin_vel", 3), ("base_ang_vel", 3), ("projected_gravity", 3),
    ],
    "proprio_hand": [("joint_pos_hand", 14), ("joint_vel_hand", 14)],
}
HOTDEX_HISTORY = {
    "proprio_hand": 5, "proprio_body": 5, "task": 5,
    "command": 1, "points": 1, "command_points": 1,
}

# ── Hotdex 43-DOF action scaling (in ACTION_ORDER_43DOF, from wbt.py:2437-2469) ──
HOTDEX_ACTION_SCALE_43 = np.array([
    0.5475, 0.5475, 0.5475, 0.3507, 0.3507, 0.4386, 0.5475, 0.5475, 0.4386,
    0.3507, 0.3507, 0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.4386,
    0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.0745, 0.0745,
    0.0745, 0.0745, 0.7000, 0.7000, 0.3063, 0.7000, 0.7000, 0.3063, 0.7000,
    0.7000, 0.7000, 0.7000, 0.7000, 0.7000, 0.7000, 0.7000,
], dtype=np.float32)

HOTDEX_ACTION_OFFSET_43 = np.array([
    -0.3120, -0.3120, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
     0.0000,  0.6690, 0.6690, 0.2000, 0.2000, -0.3630, -0.3630, 0.2000,
    -0.2000,  0.0000, 0.0000, 0.0000, 0.0000, 0.6000, 0.6000, 0.0000,
     0.0000,  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
     0.0000,  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
     0.0000,  0.0000, 0.0000,
], dtype=np.float32)

HOTDEX_ACTION_CLIP_MIN_43 = np.array([
    -10.9509, -10.9509, -10.9509, -7.0132, -7.0132, -8.7715, -10.9509, -10.9509, -8.7715,
     -7.0132,  -7.0132,  -8.7715, -8.7715, -8.7715, -8.7715, -8.7715, -8.7715, -8.7715,
     -8.7715,  -8.7715,  -8.7715, -8.7715, -8.7715, -8.7715, -8.7715, -1.4900, -1.4900,
     -1.4900,  -1.4900, -14.0000, -14.0000, -6.1250, -14.0000, -14.0000, -6.1250, -14.0000,
    -14.0000, -14.0000, -14.0000, -14.0000, -14.0000, -14.0000, -14.0000,
], dtype=np.float32)

HOTDEX_ACTION_CLIP_MAX_43 = np.array([
    10.9509, 10.9509, 10.9509, 7.0132, 7.0132, 8.7715, 10.9509, 10.9509, 8.7715,
    7.0132,  7.0132,  8.7715, 8.7715, 8.7715, 8.7715, 8.7715, 8.7715, 8.7715,
    8.7715,  8.7715,  8.7715, 8.7715, 8.7715, 8.7715, 8.7715, 1.4900, 1.4900,
    1.4900,  1.4900, 14.0000, 14.0000, 6.1250, 14.0000, 14.0000, 6.1250, 14.0000,
   14.0000, 14.0000, 14.0000, 14.0000, 14.0000, 14.0000, 14.0000,
], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Quaternion utilities (batch versions from holosoma_inference.utils.math.quat)
# ═══════════════════════════════════════════════════════════════════════════

def quat_rotate_inverse(q_wxyz, v):
    """Rotate vector v by inverse of unit quaternion q (wxyz format).
    Single (1D) input version.
    """
    w = q_wxyz[0]
    q_vec = q_wxyz[1:4]
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(q_vec, v) * w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return (a - b + c).astype(np.float32)


def quat_rotate_inverse_batch(q, v):
    """Batch version: q [N,4] wxyz, v [N,3] -> [N,3]."""
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0)[:, np.newaxis]
    b = np.cross(q_vec, v) * q_w[:, np.newaxis] * 2.0
    c = q_vec * np.sum(q_vec * v, axis=1, keepdims=True) * 2.0
    return (a - b + c).astype(np.float32)


def quat_apply(q, v):
    """Rotate vector v by quaternion q (wxyz). q [N,4], v [N,3] -> [N,3]."""
    q = q.reshape(-1, 4)
    v = v.reshape(-1, 3)
    xyz = q[:, 1:]
    w = q[:, :1]
    t = np.cross(xyz, v) * 2
    return (v + w * t + np.cross(xyz, t)).astype(np.float32)


def quat_inverse(q):
    """Conjugate of unit quaternion q [N,4] wxyz -> [N,4]."""
    return np.concatenate((q[:, 0:1], -q[:, 1:]), axis=1)


def quat_mul(a, b):
    """Hamilton product a*b. a,b [N,4] wxyz -> [N,4]."""
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return np.stack([w, x, y, z], axis=-1).astype(np.float32)


def quat_to_rpy(q):
    """Quaternion [w,x,y,z] -> (roll, pitch, yaw)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x ** 2 + y ** 2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = np.sign(sinp) * (np.pi / 2) if abs(sinp) >= 1 else np.arcsin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y ** 2 + z ** 2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def rpy_to_quat(rpy):
    """(roll, pitch, yaw) -> quaternion [w,x,y,z]."""
    roll, pitch, yaw = rpy
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=np.float32)


def matrix_from_quat(quaternions):
    """Quaternions [..., 4] wxyz -> rotation matrices [..., 3, 3]."""
    r, i, j, k = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = np.stack((
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ), -1)
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def xyzw_to_wxyz(xyzw):
    """[..., 4] xyzw -> wxyz."""
    return np.concatenate([xyzw[..., -1:], xyzw[..., :3]], axis=-1)


def normalize_quat(q):
    """Normalize quaternion(s) along last axis."""
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    return q / norm


# ═══════════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════════

def idx(names_from, names_in):
    """Get indices of names_from inside names_in."""
    return np.array([names_in.index(n) for n in names_from], dtype=np.int64)


def get_yaw_quat(base_quat_wxyz):
    """Extract yaw-only quaternion from full orientation. Returns [1,4] wxyz."""
    _, _, yaw = quat_to_rpy(base_quat_wxyz)
    return rpy_to_quat((0.0, 0.0, yaw)).reshape(1, 4)


# ═══════════════════════════════════════════════════════════════════════════
# MuJoCo helpers
# ═══════════════════════════════════════════════════════════════════════════

def _quat_normalize_wxyz(q_wxyz: np.ndarray | None) -> np.ndarray | None:
    if q_wxyz is None:
        return None
    q = np.asarray(q_wxyz, dtype=np.float32)
    n = float(np.linalg.norm(q))
    if n <= 0.0:
        return None
    return (q / n).astype(np.float32)


def _yaw_from_quat_wxyz(q_wxyz: np.ndarray) -> float:
    w, x, y, z = [float(v) for v in q_wxyz]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def _yaw_quat_wxyz(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.asarray([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = [float(v) for v in a]
    bw, bx, by, bz = [float(v) for v in b]
    return np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float32,
    )


def _quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    return np.asarray([w, -x, -y, -z], dtype=np.float32)


def _align_pos_to_root_xy_yaw(pos_w: np.ndarray, root_pos0: np.ndarray, root_yaw: float) -> np.ndarray:
    x, y, z = [float(v) for v in pos_w]
    rx, ry = float(root_pos0[0]), float(root_pos0[1])
    dx = x - rx
    dy = y - ry
    c = math.cos(root_yaw)
    s = math.sin(root_yaw)
    # Rotate by -yaw: [dx', dy'] = Rz(-yaw) * [dx, dy]
    xh = c * dx + s * dy
    yh = -s * dx + c * dy
    # Keep absolute height from the motion clip (do not modify z).
    return np.asarray([xh, yh, z], dtype=np.float32)


def _align_quat_to_root_yaw(quat_wxyz: np.ndarray, root_yaw: float) -> np.ndarray | None:
    q_yaw = _yaw_quat_wxyz(root_yaw)
    q_w2h = _quat_conjugate_wxyz(q_yaw)  # inverse for unit quats
    out = _quat_mul_wxyz(q_w2h, quat_wxyz)
    return _quat_normalize_wxyz(out)


def _select_clip_key(motion_data: dict, clip_key: str | None, obj_name_hint: str | None) -> str:
    if not motion_data:
        raise ValueError("Motion PKL contained no clips.")

    if clip_key:
        if clip_key in motion_data:
            return clip_key
        suffix = f"__{clip_key}"
        suffix_matches = [k for k in motion_data.keys() if isinstance(k, str) and k.endswith(suffix)]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        matches = [k for k in motion_data.keys() if isinstance(k, str) and clip_key in k]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(f"clip_key '{clip_key}' not found (exact or substring match).")
        preview = ", ".join(sorted(matches)[:10])
        raise ValueError(f"clip_key '{clip_key}' matched multiple clips ({len(matches)}): {preview}")

    if obj_name_hint:
        candidates: list[str] = []
        prefix = f"{obj_name_hint}__"
        for key, clip in motion_data.items():
            if isinstance(key, str) and key.startswith(prefix):
                candidates.append(key)
                continue
            if isinstance(clip, dict) and clip.get("obj_name") == obj_name_hint:
                candidates.append(key)
        if candidates:
            return sorted(candidates)[0]

    return next(iter(motion_data.keys()))


def _load_motion_init_first_frame(
    pkl_path: str,
    *,
    clip_key: str | None,
    obj_name_hint: str | None,
    align_to_root_xy_yaw: bool,
) -> tuple[str, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Return (chosen_key, obj_pos_xyz, obj_quat_wxyz, table_pos_xyz, table_quat_wxyz)."""
    motion_data = joblib.load(pkl_path)
    if not isinstance(motion_data, dict):
        raise TypeError(f"Motion PKL must be a dict, got {type(motion_data)}")

    chosen_key = _select_clip_key(motion_data, clip_key, obj_name_hint)
    clip = motion_data[chosen_key]
    if not isinstance(clip, dict):
        raise TypeError(f"Clip '{chosen_key}' must be a dict, got {type(clip)}")

    def _first_row(val, dims: int) -> np.ndarray | None:
        if val is None:
            return None
        arr = np.asarray(val, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != dims or arr.shape[0] < 1:
            return None
        return arr[0].astype(np.float32, copy=False)

    obj_pos = _first_row(clip.get("obj_pos"), 3)
    table_pos = _first_row(clip.get("table_pos"), 3)

    obj_rot_xyzw = _first_row(clip.get("obj_rot"), 4)
    table_rot_xyzw = _first_row(clip.get("table_rot"), 4)
    obj_quat = _quat_normalize_wxyz(xyzw_to_wxyz(obj_rot_xyzw)) if obj_rot_xyzw is not None else None
    table_quat = _quat_normalize_wxyz(xyzw_to_wxyz(table_rot_xyzw)) if table_rot_xyzw is not None else None

    if align_to_root_xy_yaw:
        root_pos0 = None
        root_quat0_wxyz = None
        try:
            root_pos_raw = clip.get("global_translation_extend")
            root_quat_raw = clip.get("global_rotation_extend")
            if root_pos_raw is not None:
                root_pos0 = np.asarray(root_pos_raw, dtype=np.float32)[0, 0]
            if root_quat_raw is not None:
                root_quat0_xyzw = np.asarray(root_quat_raw, dtype=np.float32)[0, 0]
                root_quat0_wxyz = _quat_normalize_wxyz(xyzw_to_wxyz(root_quat0_xyzw))
        except Exception:
            root_pos0 = None
            root_quat0_wxyz = None

        if root_pos0 is not None and root_quat0_wxyz is not None:
            yaw = _yaw_from_quat_wxyz(root_quat0_wxyz)
            if obj_pos is not None:
                obj_pos = _align_pos_to_root_xy_yaw(obj_pos, root_pos0, yaw)
            if table_pos is not None:
                table_pos = _align_pos_to_root_xy_yaw(table_pos, root_pos0, yaw)
            if obj_quat is not None:
                obj_quat = _align_quat_to_root_yaw(obj_quat, yaw)
            if table_quat is not None:
                table_quat = _align_quat_to_root_yaw(table_quat, yaw)

    return chosen_key, obj_pos, obj_quat, table_pos, table_quat


def _patch_mjcf_motion_init(
    xml_path: str,
    *,
    clip_key: str,
    table_pos: np.ndarray | None,
    table_quat_wxyz: np.ndarray | None,
    obj_pos: np.ndarray | None,
    obj_quat_wxyz: np.ndarray | None,
    table_body_name: str = "table",
    object_body_name: str = "object",
) -> tuple[str, object]:
    """Patch MJCF in memory to apply table/object pose from a motion clip first frame.
    
    Returns:
        (temp_file_path, temp_file_obj) - temp file object keeps file alive until explicitly closed.
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    def _fmt_vec(vals) -> str:
        return " ".join(f"{float(v):.8f}" for v in vals)

    touched = False
    for body in root.iter("body"):
        name = body.get("name")
        if name == table_body_name:
            if table_pos is not None:
                body.set("pos", _fmt_vec(table_pos))
                touched = True
            if table_quat_wxyz is not None:
                body.set("quat", _fmt_vec(table_quat_wxyz))
                touched = True
        elif name == object_body_name:
            if obj_pos is not None:
                body.set("pos", _fmt_vec(obj_pos))
                touched = True
            if obj_quat_wxyz is not None:
                body.set("quat", _fmt_vec(obj_quat_wxyz))
                touched = True

    if not touched:
        return xml_path, None

    # Create temporary file in same directory as original XML (so relative mesh paths work)
    xml_dir = str(Path(xml_path).parent)
    temp_file = tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.xml',
        delete=False,
        prefix=f"testwbt_motioninit_{clip_key.replace('/', '_')}_",
        dir=xml_dir  # Create temp file in same dir as source XML
    )
    tree.write(temp_file.name, encoding="utf-8")
    temp_file.flush()
    temp_file.close()  # Close but don't delete yet
    
    return temp_file.name, temp_file


def _mj_hinge_qpos_qvel_addrs(model, dof_names):
    """Return qpos/qvel addresses for hinge joints in the given name order."""
    qpos_addrs = np.zeros(len(dof_names), dtype=np.int64)
    qvel_addrs = np.zeros(len(dof_names), dtype=np.int64)
    for i, name in enumerate(dof_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise KeyError(f"Joint '{name}' not found in MJCF.")
        qpos_addrs[i] = int(model.jnt_qposadr[joint_id])
        qvel_addrs[i] = int(model.jnt_dofadr[joint_id])
    return qpos_addrs, qvel_addrs


def _mj_actuator_ids(model, actuator_names):
    ids = np.zeros(len(actuator_names), dtype=np.int64)
    for i, name in enumerate(actuator_names):
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise KeyError(f"Actuator '{name}' not found in MJCF.")
        ids[i] = int(actuator_id)
    return ids


def read_mujoco_state(data, dof_qpos_addrs, dof_qvel_addrs):
    """Read robot state from MuJoCo data.

    Returns (joint_pos[43], joint_vel[43], base_quat_wxyz[4],
             base_ang_vel[3], base_lin_vel[3], base_pos[3]).
    """
    joint_pos = data.qpos[dof_qpos_addrs].astype(np.float32)
    joint_vel = data.qvel[dof_qvel_addrs].astype(np.float32)
    base_quat_wxyz = data.qpos[3:7].astype(np.float32)
    base_ang_vel = data.qvel[3:6].astype(np.float32)
    base_lin_vel = data.qvel[0:3].astype(np.float32)
    base_pos = data.qpos[0:3].astype(np.float32)
    return joint_pos, joint_vel, base_quat_wxyz, base_ang_vel, base_lin_vel, base_pos


# ═══════════════════════════════════════════════════════════════════════════
# Observation builders
# ═══════════════════════════════════════════════════════════════════════════

def build_stabilization_obs(joint_pos, joint_vel, base_quat_wxyz, base_ang_vel,
                            last_action, motion_cmd_seq, obs_idx):
    """Build the 673-dim stabilization observation vector. Returns [1, 673]."""
    dof_pos_obs = joint_pos[obs_idx] - DEFAULT_DOF_ANGLES[obs_idx]
    dof_vel_obs = joint_vel[obs_idx]
    proj_grav = quat_rotate_inverse(base_quat_wxyz, np.array([0, 0, -1], dtype=np.float32))

    return np.concatenate([
        dof_pos_obs, dof_vel_obs, base_ang_vel, proj_grav,
        last_action, motion_cmd_seq,
    ]).reshape(1, -1)


def build_hotdex_obs(joint_pos, joint_vel, base_lin_vel, base_ang_vel, proj_grav,
                     body_idx, hand_idx, obj_pos_w, obj_quat_w, mesh_offset,
                     motion_obj_pos, motion_obj_rot_wxyz, motion_timestep, motion_length,
                     ref_pos, ref_yaw_quat, last_action_43, prev_obj_state):
    """Build current observation dict for hotdex (before history stacking)."""
    dof_pos_rel = joint_pos - DEFAULT_DOF_ANGLES

    obs = {}
    # Proprio body
    obs["joint_pos_body"] = dof_pos_rel[body_idx].reshape(1, -1).astype(np.float32)
    obs["joint_vel_body"] = joint_vel[body_idx].reshape(1, -1).astype(np.float32)
    obs["base_lin_vel"] = base_lin_vel.reshape(1, -1).astype(np.float32)
    obs["base_ang_vel"] = base_ang_vel.reshape(1, -1).astype(np.float32)
    obs["projected_gravity"] = proj_grav.reshape(1, -1).astype(np.float32)

    # Proprio hand
    obs["joint_pos_hand"] = dof_pos_rel[hand_idx].reshape(1, -1).astype(np.float32)
    obs["joint_vel_hand"] = joint_vel[hand_idx].reshape(1, -1).astype(np.float32)

    # Task
    obs["obj_pos_diff_b"] = compute_obj_pos_diff_b(obj_pos_w, prev_obj_state, ref_yaw_quat)
    obs["actions"] = last_action_43.reshape(1, -1).astype(np.float32)

    # Points
    obs["obj_pcd"] = compute_obj_pcd(mesh_offset, obj_pos_w, obj_quat_w, ref_pos[0], ref_yaw_quat)
    obs["motion_obj_pcd"] = compute_motion_obj_pcd(
        mesh_offset, motion_obj_pos, motion_obj_rot_wxyz,
        motion_timestep, motion_length, ref_pos[0], ref_yaw_quat,
    )

    # Command
    obs["motion_obj_pos_rel_all"] = compute_motion_obj_pos_rel_all(
        motion_obj_pos, motion_timestep, motion_length, ref_yaw_quat,
    )
    obs["motion_obj_ori_rel_all"] = compute_motion_obj_ori_rel_all(
        motion_obj_rot_wxyz, motion_timestep, motion_length, ref_yaw_quat,
    )

    return obs


# ═══════════════════════════════════════════════════════════════════════════
# Hotdex observation computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_obj_pcd(mesh_offset, obj_pos, obj_quat_wxyz, ref_pos, ref_yaw_quat):
    """Current object point cloud in robot heading frame. Returns [1, 3072]."""
    pts = mesh_offset.reshape(-1, 3)  # [1024, 3]
    n = pts.shape[0]
    q_rep = np.repeat(obj_quat_wxyz.reshape(1, 4), n, axis=0)
    pts_world = obj_pos.reshape(1, 3) + quat_apply(q_rep, pts)
    rel_w = pts_world - ref_pos.reshape(1, 3)
    q_yaw = np.repeat(ref_yaw_quat, n, axis=0)
    rel_h = quat_rotate_inverse_batch(q_yaw, rel_w)
    return rel_h.reshape(1, -1).astype(np.float32)


def compute_motion_obj_pcd(mesh_offset, motion_obj_pos, motion_obj_rot_wxyz,
                           timestep, motion_length, ref_pos, ref_yaw_quat):
    """Motion reference object point cloud in robot heading frame. Returns [1, 3072]."""
    base_idx = int(np.clip(timestep + 1, 0, motion_length - 1))
    obj_pos = motion_obj_pos[base_idx]
    obj_quat = motion_obj_rot_wxyz[base_idx]
    return compute_obj_pcd(mesh_offset, obj_pos, obj_quat, ref_pos, ref_yaw_quat)


def compute_obj_pos_diff_b(obj_pos_w, prev_state, current_yaw_quat_wxyz):
    """Object position delta in the *previous* robot heading frame. Returns [1, 3].

    prev_state is a dict with 'obj_pos' and 'yaw_quat', updated in-place.
    """
    if prev_state["obj_pos"] is None or prev_state["yaw_quat"] is None:
        prev_state["obj_pos"] = obj_pos_w.copy()
        prev_state["yaw_quat"] = current_yaw_quat_wxyz.copy()
        return np.zeros((1, 3), dtype=np.float32)

    delta_w = (obj_pos_w - prev_state["obj_pos"]).reshape(1, 3)
    diff_b = quat_rotate_inverse_batch(prev_state["yaw_quat"], delta_w).astype(np.float32)

    prev_state["obj_pos"] = obj_pos_w.copy()
    prev_state["yaw_quat"] = current_yaw_quat_wxyz.copy()
    return diff_b


def _motion_short_indices(base_index):
    return base_index + np.arange(0, 10, dtype=np.int64)


def _motion_long_indices(base_index):
    return base_index + np.array([19, 39, 59, 79, 99], dtype=np.int64)


def compute_motion_obj_pos_rel_all(motion_obj_pos, timestep, motion_length, ref_yaw_quat):
    """14 keypoint positions relative to base, in heading frame. Returns [1, 42]."""
    if motion_obj_pos is None or motion_length == 0:
        return np.zeros((1, 42), dtype=np.float32)
    motion_obj_pos = np.asarray(motion_obj_pos, dtype=np.float32)
    if motion_obj_pos.ndim != 2 or motion_obj_pos.shape[1] != 3:
        return np.zeros((1, 42), dtype=np.float32)
    ref_yaw_quat = np.asarray(ref_yaw_quat, dtype=np.float32).reshape(1, 4)

    base_index = int(np.clip(timestep + 1, 0, motion_length - 1))
    short_idx = _motion_short_indices(base_index)
    long_idx = _motion_long_indices(base_index)
    short_idx = np.clip(short_idx, 0, motion_length - 1)
    long_idx = np.clip(long_idx, 0, motion_length - 1)

    short_pos_w = motion_obj_pos[short_idx]  # [10, 3]
    long_pos_w = motion_obj_pos[long_idx]    # [5, 3]

    base_pos_w = short_pos_w[0].reshape(1, 3)
    rel_s_w = short_pos_w[1:] - base_pos_w
    rel_l_w = long_pos_w - base_pos_w

    q_s = np.repeat(ref_yaw_quat, rel_s_w.shape[0], axis=0)
    q_l = np.repeat(ref_yaw_quat, rel_l_w.shape[0], axis=0)
    rel_s_h = quat_rotate_inverse_batch(q_s, rel_s_w)
    rel_l_h = quat_rotate_inverse_batch(q_l, rel_l_w)

    out = np.concatenate([rel_s_h, rel_l_h], axis=0)
    return out.reshape(1, -1).astype(np.float32, copy=False)


def compute_motion_obj_ori_rel_all(motion_obj_rot_wxyz, timestep, motion_length, ref_yaw_quat):
    """14 keypoint orientations as 6D rotation, in heading frame. Returns [1, 84]."""
    if motion_obj_rot_wxyz is None or motion_length == 0:
        return np.zeros((1, 84), dtype=np.float32)
    base_index = int(np.clip(timestep + 1, 0, motion_length - 1))
    short_idx = np.clip(_motion_short_indices(base_index), 0, motion_length - 1)
    long_idx = np.clip(_motion_long_indices(base_index), 0, motion_length - 1)

    short_rot = normalize_quat(motion_obj_rot_wxyz[short_idx])
    long_rot = normalize_quat(motion_obj_rot_wxyz[long_idx])

    q_w2h = quat_inverse(ref_yaw_quat)
    q0_h = normalize_quat(quat_mul(q_w2h, short_rot[:1]))
    q0_h_inv = quat_inverse(q0_h)

    q_w2h_s = np.repeat(q_w2h, 9, axis=0)
    qt_h_s = normalize_quat(quat_mul(q_w2h_s, short_rot[1:]))
    dq_s = normalize_quat(quat_mul(np.repeat(q0_h_inv, 9, axis=0), qt_h_s))

    q_w2h_l = np.repeat(q_w2h, 5, axis=0)
    qt_h_l = normalize_quat(quat_mul(q_w2h_l, long_rot))
    dq_l = normalize_quat(quat_mul(np.repeat(q0_h_inv, 5, axis=0), qt_h_l))

    dq = np.concatenate([dq_s, dq_l], axis=0)
    mat = matrix_from_quat(dq)
    out = mat[..., :2].reshape(1, -1)
    return out.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# History buffer management
# ═══════════════════════════════════════════════════════════════════════════

def init_history_buffers():
    """Create empty deque-based history buffers for each obs term."""
    buffers = {}
    for group, terms in HOTDEX_OBS_CONFIG.items():
        history_len = HOTDEX_HISTORY[group]
        buffers[group] = {}
        for term_name, _ in terms:
            buffers[group][term_name] = deque(maxlen=history_len)
    return buffers


def update_history_and_flatten(current_obs, history_buffers):
    """Push current obs into buffers, return concatenated flattened obs [1, 6975]."""
    group_outputs = []
    for group, terms in HOTDEX_OBS_CONFIG.items():
        history_len = HOTDEX_HISTORY[group]
        flattened_terms = []
        for term_name, _ in terms:
            obs = current_obs[term_name]
            buf = history_buffers[group][term_name]
            buf.append(obs.copy())

            history = list(buf)
            if len(history) < history_len:
                missing = history_len - len(history)
                if group in ("proprio_body", "proprio_hand"):
                    pad = history[0] if history else obs
                    history = [pad.copy()] * missing + history
                else:
                    history = [np.zeros_like(obs)] * missing + history

            stacked = np.stack(history[-history_len:], axis=1)
            flattened_terms.append(stacked.reshape(1, -1))

        group_outputs.append(np.concatenate(flattened_terms, axis=1))
    return np.concatenate(group_outputs, axis=1).astype(np.float32)


def seed_history(current_obs, history_buffers):
    """Fill all history buffers by repeating current observation."""
    for group, terms in HOTDEX_OBS_CONFIG.items():
        history_len = HOTDEX_HISTORY[group]
        for term_name, _ in terms:
            obs = current_obs[term_name]
            buf = history_buffers[group][term_name]
            buf.clear()
            for _ in range(history_len):
                buf.append(obs.copy())


# ═══════════════════════════════════════════════════════════════════════════
# Action processing
# ═══════════════════════════════════════════════════════════════════════════

def scale_stabilization_action(raw_action, joint_pos, act_idx, hand_indices,
                               act_scale_29, act_offset_29,
                               act_clip_min_29, act_clip_max_29,
                               target_q_from_motion):
    """Scale 29-DOF raw action -> 43-DOF config-order position targets."""
    scaled_29 = raw_action * act_scale_29 + act_offset_29
    scaled_29 = np.clip(scaled_29, act_clip_min_29, act_clip_max_29)

    target_q = joint_pos.copy()
    target_q[act_idx] = scaled_29
    if target_q_from_motion is not None and hand_indices.size > 0:
        target_q[hand_indices] = target_q_from_motion[hand_indices]
    return target_q


def scale_hotdex_action(raw_action_43, config_to_action_order):
    """Scale 43-DOF raw action -> config-order position targets."""
    scaled_43 = raw_action_43 * HOTDEX_ACTION_SCALE_43 + HOTDEX_ACTION_OFFSET_43
    scaled_43 = np.clip(scaled_43, HOTDEX_ACTION_CLIP_MIN_43, HOTDEX_ACTION_CLIP_MAX_43)
    return scaled_43[0, config_to_action_order]


def apply_pd_control(data, target_q, dof_qpos_addrs, dof_qvel_addrs, actuator_ids):
    """Compute PD torques and write to data.ctrl."""
    curr_q = data.qpos[dof_qpos_addrs].astype(np.float32)
    curr_dq = data.qvel[dof_qvel_addrs].astype(np.float32)
    tau = MOTOR_KP * (target_q - curr_q) + MOTOR_KD * (0.0 - curr_dq)
    data.ctrl[:] = 0.0
    data.ctrl[actuator_ids] = tau


# ═══════════════════════════════════════════════════════════════════════════
# Resource loading
# ═══════════════════════════════════════════════════════════════════════════

def load_motion_clip(pkl_path, clip_key=None, mesh_root=None, *, debug_motion_obj: bool = False):
    """Load motion clip, object data, and mesh points.

    Returns dict with: motion_cmd_seq, init_q, target_q_from_motion,
    motion_obj_pos, motion_obj_rot_wxyz, motion_length, mesh_offset,
    interp_target_config.
    """
    motion_data = joblib.load(pkl_path)
    preferred_key = "GRAB_s1_cubemedium_pass_1"
    if clip_key is None:
        if preferred_key in motion_data:
            clip_key = preferred_key
        else:
            clip_key = next(iter(motion_data.keys()))
    else:
        clip_key = _select_clip_key(motion_data, clip_key, obj_name_hint=None)

    if clip_key not in motion_data:
        available = list(motion_data.keys())
        raise KeyError(
            f"clip_key '{clip_key}' not found in PKL. "
            f"Available clips: {available[:10]}{' ...' if len(available) > 10 else ''}"
        )

    clip_data = motion_data[clip_key]
    motion_fps = clip_data['fps']
    motion_dof_pos = clip_data['dof_pos']
    motion_length = motion_dof_pos.shape[0]
    print(f"Motion loaded: clip='{clip_key}', {motion_length} frames @ {motion_fps}Hz, "
          f"{motion_dof_pos.shape[1]} DOF")

    # Object data
    motion_obj_pos = (np.asarray(clip_data['obj_pos'], dtype=np.float32)
                      if clip_data.get('obj_pos') is not None else None)
    motion_obj_rot_raw = clip_data.get('obj_rot')
    if motion_obj_rot_raw is not None:
        motion_obj_rot_wxyz = xyzw_to_wxyz(np.asarray(motion_obj_rot_raw, dtype=np.float32))
    else:
        motion_obj_rot_wxyz = None
    obj_name = clip_data.get('obj_name', 'cubemedium')
    print(f"  Object: {obj_name}, obj_pos={motion_obj_pos.shape if motion_obj_pos is not None else None}, "
          f"obj_rot={motion_obj_rot_wxyz.shape if motion_obj_rot_wxyz is not None else None}")

    # Table data - load trajectory (all frames)
    motion_table_pos = (np.asarray(clip_data.get('table_pos'), dtype=np.float32)
                        if clip_data.get('table_pos') is not None else None)
    motion_table_rot_raw = clip_data.get('table_rot')
    if motion_table_rot_raw is not None:
        motion_table_rot_wxyz = xyzw_to_wxyz(np.asarray(motion_table_rot_raw, dtype=np.float32))
    else:
        motion_table_rot_wxyz = None
    if motion_table_pos is not None:
        print(f"  Table trajectory loaded: pos shape={motion_table_pos.shape}, "
              f"rot shape={motion_table_rot_wxyz.shape if motion_table_rot_wxyz is not None else None}")

    # Match `holosoma_inference.policies.wbt.WBTPolicy._align_motion_objects_to_root`:
    # align motion object poses by removing root xy translation and initial yaw.
    root_pos_raw = None
    root_quat_raw = None
    if "global_translation_extend" in clip_data and "global_rotation_extend" in clip_data:
        root_pos_raw = np.asarray(clip_data["global_translation_extend"], dtype=np.float32)[:, 0, :]
        root_quat_raw = np.asarray(clip_data["global_rotation_extend"], dtype=np.float32)[:, 0, :]

    if (
        motion_obj_pos is not None
        and motion_obj_rot_wxyz is not None
        and root_pos_raw is not None
        and root_quat_raw is not None
        and root_pos_raw.shape[0] > 0
        and root_quat_raw.shape[0] > 0
    ):
        obj_pos0_raw = motion_obj_pos[0].astype(np.float32, copy=False)
        obj_quat0_raw = motion_obj_rot_wxyz[0].astype(np.float32, copy=False)
        root_pos0 = np.asarray(root_pos_raw[0], dtype=np.float32).reshape(3,)
        root_quat0_wxyz = xyzw_to_wxyz(np.asarray(root_quat_raw[0], dtype=np.float32).reshape(1, 4))
        _, _, yaw0 = quat_to_rpy(root_quat0_wxyz.reshape(-1))
        yaw_quat0 = rpy_to_quat((0.0, 0.0, float(yaw0))).reshape(1, 4)

        # Position: subtract xy translation, keep z, then remove yaw
        trans_xy = np.asarray([root_pos0[0], root_pos0[1], 0.0], dtype=np.float32)
        rel_w = motion_obj_pos - trans_xy.reshape(1, 3)
        q_pos = np.repeat(yaw_quat0, rel_w.shape[0], axis=0)
        motion_obj_pos = quat_rotate_inverse_batch(q_pos, rel_w).astype(np.float32, copy=False)

        # Rotation: remove yaw from object orientation (wxyz)
        q_w2h = quat_inverse(yaw_quat0)
        q_w2h_rep = np.repeat(q_w2h, motion_obj_rot_wxyz.shape[0], axis=0)
        motion_obj_rot_wxyz = normalize_quat(quat_mul(q_w2h_rep, motion_obj_rot_wxyz))
        if debug_motion_obj:
            print(
                "  [motion_obj_align] "
                f"root_xy=({float(root_pos0[0]):.3f},{float(root_pos0[1]):.3f}) "
                f"yaw0_deg={float(np.degrees(yaw0)):.2f} "
                f"obj_pos0_raw={obj_pos0_raw.reshape(-1).tolist()} "
                f"obj_pos0_aligned={motion_obj_pos[0].reshape(-1).tolist()} "
                f"obj_quat0_raw_wxyz={obj_quat0_raw.reshape(-1).tolist()} "
                f"obj_quat0_aligned_wxyz={motion_obj_rot_wxyz[0].reshape(-1).tolist()}",
                flush=True,
            )
        else:
            print("  Aligned motion object poses to root xy/yaw of the first frame.")
        
        # Also align table trajectory if available
        if motion_table_pos is not None:
            rel_w_table = motion_table_pos - trans_xy.reshape(1, 3)
            q_pos_table = np.repeat(yaw_quat0, rel_w_table.shape[0], axis=0)
            motion_table_pos = quat_rotate_inverse_batch(q_pos_table, rel_w_table).astype(np.float32, copy=False)
            
            if motion_table_rot_wxyz is not None:
                q_w2h_rep_table = np.repeat(q_w2h, motion_table_rot_wxyz.shape[0], axis=0)
                motion_table_rot_wxyz = normalize_quat(quat_mul(q_w2h_rep_table, motion_table_rot_wxyz))

    # Mesh points
    if mesh_root is not None:
        mesh_path = mesh_root / f"src/holosoma/holosoma/data/objects_new/objects_new/{obj_name}/{obj_name}_sample_points_1024.pkl"
        if mesh_path.exists():
            mesh_data = joblib.load(mesh_path)
            mesh_offset = np.asarray(mesh_data["points"], dtype=np.float32)
            print(f"  Mesh points loaded: {mesh_offset.shape}")
        else:
            print(f"  WARNING: Mesh points not found at {mesh_path}, using zeros")
            mesh_offset = np.zeros((1024, 3), dtype=np.float32)
    else:
        mesh_offset = np.zeros((1024, 3), dtype=np.float32)

    # Compute velocities via finite differences
    dof_pos_next = np.roll(motion_dof_pos, -1, axis=0)
    dof_pos_next[-1] = motion_dof_pos[-1]
    motion_dof_vel = (dof_pos_next - motion_dof_pos) * motion_fps
    motion_dof_vel[-1] = 0.0

    motion_joint_names = G1_DEX3_JOINT_NAMES if motion_dof_pos.shape[1] == 43 else G1_MOTION_JOINT_NAMES_29
    motion_body_indices = idx(G1_MOTION_JOINT_NAMES_29, motion_joint_names)

    pos0 = motion_dof_pos[0][motion_body_indices].copy()
    vel0 = motion_dof_vel[0][motion_body_indices].copy()

    # motion_command_sequence: [pos0, vel0] repeated 10 times -> 580
    frame = np.concatenate([pos0, vel0])
    motion_cmd_seq = np.tile(frame, 10).astype(np.float32)
    print(f"  First frame motion command built: pos0[:3]={pos0[:3]}, vel0[:3]={vel0[:3]}")

    # Initial joint pose from motion first frame (mapped to config order)
    init_q = DEFAULT_DOF_ANGLES.copy()
    target_q_from_motion = None
    interp_target_config = DEFAULT_DOF_ANGLES.reshape(1, -1)
    if motion_dof_pos.shape[1] == 43:
        motion_to_config = idx(DOF_NAMES, motion_joint_names)
        init_q = motion_dof_pos[0][motion_to_config].astype(np.float32, copy=False)
        target_q_from_motion = init_q.copy()
        interp_target_config = init_q.reshape(1, -1).copy()
    else:
        config_body_indices = idx(G1_MOTION_JOINT_NAMES_29, DOF_NAMES)
        init_q[config_body_indices] = motion_dof_pos[0][: len(config_body_indices)].astype(np.float32, copy=False)

    return {
        "clip_key": clip_key,
        "motion_fps": motion_fps,
        "motion_cmd_seq": motion_cmd_seq,
        "init_q": init_q,
        "target_q_from_motion": target_q_from_motion,
        "motion_obj_pos": motion_obj_pos,
        "motion_obj_rot_wxyz": motion_obj_rot_wxyz,
        "motion_table_pos": motion_table_pos,
        "motion_table_rot_wxyz": motion_table_rot_wxyz,
        "motion_length": motion_length,
        "mesh_offset": mesh_offset,
        "interp_target_config": interp_target_config,
    }


def build_index_maps():
    """Precompute all observation/action index maps and scaling arrays.

    Returns dict with stabilization and hotdex index maps.
    """
    obs_idx = idx(STABILIZATION_OBS_ORDER, DOF_NAMES)
    act_idx = idx(STABILIZATION_ACTION_ORDER, DOF_NAMES)
    stab_in_43 = idx(STABILIZATION_ACTION_ORDER, ACTION_ORDER_43DOF)

    hand_mask = np.ones(len(DOF_NAMES), dtype=bool)
    hand_mask[act_idx] = False
    hand_indices = np.nonzero(hand_mask)[0].astype(np.int64)

    # Hotdex indices
    hand_idx = idx(HAND_JOINT_NAMES, DOF_NAMES)
    body_idx = idx(BODY_JOINT_NAMES, DOF_NAMES)
    action_order_to_config = idx(list(ACTION_ORDER_43DOF), DOF_NAMES)
    config_to_action_order = np.argsort(action_order_to_config)

    return {
        "obs_idx": obs_idx,
        "act_idx": act_idx,
        "hand_indices": hand_indices,
        "act_scale_29": ACTION_SCALE_43[stab_in_43],
        "act_offset_29": ACTION_OFFSET_43[stab_in_43],
        "act_clip_min_29": ACTION_CLIP_MIN_43[stab_in_43],
        "act_clip_max_29": ACTION_CLIP_MAX_43[stab_in_43],
        "hand_idx": hand_idx,
        "body_idx": body_idx,
        "config_to_action_order": config_to_action_order,
    }


def _mj_freejoint_addrs(model: mujoco.MjModel, body_name: str) -> tuple[int, int]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise KeyError(f"Body '{body_name}' not found in MJCF.")
    jnt_adr = int(model.body_jntadr[body_id])
    jnt_num = int(model.body_jntnum[body_id])
    if jnt_num != 1 or model.jnt_type[jnt_adr] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"Body '{body_name}' is not a single freejoint body (jnt_num={jnt_num}).")
    qpos_adr = int(model.jnt_qposadr[jnt_adr])
    qvel_adr = int(model.jnt_dofadr[jnt_adr])
    return qpos_adr, qvel_adr


def _set_freejoint_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    body_name: str,
    pos_xyz: list[float] | None,
    quat_wxyz: list[float] | None,
) -> None:
    qpos_adr, qvel_adr = _mj_freejoint_addrs(model, body_name)
    if pos_xyz is not None:
        data.qpos[qpos_adr : qpos_adr + 3] = np.asarray(pos_xyz, dtype=np.float32)
    if quat_wxyz is not None:
        # MuJoCo freejoint qpos is [x,y,z, qw,qx,qy,qz]
        data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.asarray(quat_wxyz, dtype=np.float32)
    data.qvel[qvel_adr : qvel_adr + 6] = 0.0


def _set_static_body_pose(
    model: mujoco.MjModel,
    body_name: str,
    pos_xyz: list[float] | None,
    quat_wxyz: list[float] | None,
) -> None:
    """Set position/quaternion of a static body by modifying model.body_pos and model.body_quat.
    
    Warning: This modifies the model in-place. Changes affect all future simulations with this model.
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise KeyError(f"Body '{body_name}' not found in MJCF.")
    
    if pos_xyz is not None:
        model.body_pos[body_id] = np.asarray(pos_xyz, dtype=np.float64)
    if quat_wxyz is not None:
        model.body_quat[body_id] = np.asarray(quat_wxyz, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# Simulation loop
# ═══════════════════════════════════════════════════════════════════════════

def run(model, data, args, *, stab_session, hotdex_session,
        idx_maps, motion, obj_body_id, ref_body_id,
        dof_qpos_addrs, dof_qvel_addrs, actuator_ids, inference_log_data, inference_log_path,
        temp_xml_file=None, video_recorder: SimpleVideoRecorder | None = None):
    """Main simulation loop with stabilization -> hotdex state machine."""
    debug_motion_obj = bool(getattr(args, "debug_motion_obj", False) or int(os.getenv("TEST_WBT_DEBUG_MOTION_OBJ", "0")))
    debug_motion_obj_logged: set[int] = set()
    # Unpack index maps
    obs_idx = idx_maps["obs_idx"]
    act_idx = idx_maps["act_idx"]
    hand_indices = idx_maps["hand_indices"]
    act_scale_29 = idx_maps["act_scale_29"]
    act_offset_29 = idx_maps["act_offset_29"]
    act_clip_min_29 = idx_maps["act_clip_min_29"]
    act_clip_max_29 = idx_maps["act_clip_max_29"]
    hand_idx = idx_maps["hand_idx"]
    body_idx = idx_maps["body_idx"]
    config_to_action_order = idx_maps["config_to_action_order"]

    # Unpack motion data
    motion_cmd_seq = motion["motion_cmd_seq"]
    motion_fps = float(motion["motion_fps"])
    target_q_from_motion = motion["target_q_from_motion"]
    motion_obj_pos = motion["motion_obj_pos"]
    motion_obj_rot_wxyz = motion["motion_obj_rot_wxyz"]
    motion_table_pos = motion.get("motion_table_pos")  # Table trajectory
    motion_table_rot_wxyz = motion.get("motion_table_rot_wxyz")
    motion_length = motion["motion_length"]
    mesh_offset = motion["mesh_offset"]
    interp_target_config = motion["interp_target_config"]

    # Action state
    last_action_29 = np.zeros(29, dtype=np.float32)
    last_action_43 = np.zeros((1, 43), dtype=np.float32)
    target_q = DEFAULT_DOF_ANGLES.copy()

    # Timing / decimation
    sim_dt = float(model.opt.timestep)
    if sim_dt <= 0.0:
        raise ValueError(f"Invalid MuJoCo dt={sim_dt}")

    steps_per_policy = int(getattr(args, "steps_per_policy", 4))
    if steps_per_policy <= 0:
        raise ValueError(f"--steps-per-policy must be > 0, got {steps_per_policy}")

    render_decimation = int(getattr(args, "render_decimation", 4))
    if render_decimation <= 0:
        raise ValueError(f"--render-decimation must be > 0, got {render_decimation}")

    target_rtf = float(getattr(args, "rtf", 1.0))
    if target_rtf < 0.0:
        raise ValueError(f"--rtf must be >= 0, got {target_rtf}")

    policy_hz = 1.0 / (sim_dt * steps_per_policy)
    print(
        f"  Motion timing: motion_fps={motion_fps}Hz, policy_hz={policy_hz:.1f}Hz, "
        f"policy_dt={sim_dt * steps_per_policy:.4f}s, "
        f"motion frames per policy step={sim_dt * steps_per_policy * motion_fps:.2f}"
    )

    step = 0  # physics step counter
    wall_last = time.perf_counter()
    sim_t_last = float(data.time)
    physics_steps_since = 0
    policy_steps_since = 0
    last_render_step = -render_decimation
    wall_next_step = time.perf_counter()

    # Hotdex state machine
    motion_active = False
    interpolating = False
    need_seed_hotdex_history = False
    interp_count = 0
    interp_steps = max(int(args.interp_steps), 0)
    motion_timestep = 0
    motion_time_acc = 0.0  # accumulated sim time since hotdex started (seconds)
    policy_dt = sim_dt * steps_per_policy  # sim time per policy step
    history_buffers = init_history_buffers()
    prev_obj_state = {"obj_pos": None, "yaw_quat": None}
    freeze_object_state = False
    cached_obj_pos_w = None
    cached_obj_quat_w = None
    hotdex_debug_every = 50
    inference_log_wait_notice_printed = False
    
    # Auto hotdex start
    auto_hotdex = getattr(args, "auto_hotdex", False)
    auto_hotdex_start_time = None
    if auto_hotdex:
        auto_hotdex_start_time = time.perf_counter()
        print(f"[auto-hotdex] Will start hotdex in 1 second...", flush=True)

    # Key press handling
    key_pressed = {"s": False}
    debug_key_events_remaining = 50 if args.debug_keys else 0

    def key_callback(keycode, *args):
        action = None
        if len(args) == 1:
            a0 = args[0]
            if isinstance(a0, int) and a0 in (1, 2):
                action = a0
        elif len(args) >= 2:
            a1 = args[1]
            if isinstance(a1, int) and a1 in (0, 1, 2):
                action = a1
            else:
                a0 = args[0]
                if isinstance(a0, int) and a0 in (0, 1, 2):
                    action = a0
        if action == 0:
            return

        nonlocal debug_key_events_remaining
        if debug_key_events_remaining > 0:
            debug_key_events_remaining -= 1
            as_char = None
            if isinstance(keycode, int) and 32 <= keycode <= 126:
                as_char = chr(keycode)
            print(
                f"[key_callback] keycode={keycode!r} as_char={as_char!r} args={args!r} action={action!r}",
                flush=True,
            )

        GLFW_KEY_S = 83
        if keycode in ("s", "S") or keycode in (ord("s"), ord("S"), GLFW_KEY_S):
            key_pressed["s"] = True

    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            while viewer.is_running():
                t = data.time

                # ── Handle auto hotdex start ──────────────────────────────
                if auto_hotdex and auto_hotdex_start_time is not None and not motion_active:
                    elapsed = time.perf_counter() - auto_hotdex_start_time
                    if elapsed >= 1.0:
                        print(f"\n[t={t:.2f}s] >>> auto-hotdex: switching to hotdex motion tracking")
                        motion_active = True
                        interpolating = interp_steps > 0
                        need_seed_hotdex_history = True
                        interp_count = 0
                        motion_timestep = 0
                        motion_time_acc = 0.0
                        last_action_43[:] = 0.0
                        prev_obj_state["obj_pos"] = None
                        prev_obj_state["yaw_quat"] = None
                        freeze_object_state = False
                        cached_obj_pos_w = None
                        cached_obj_quat_w = None
                        auto_hotdex_start_time = None  # Disable further checks

                # ── Handle 's' key press ──────────────────────────────────
                if key_pressed["s"]:
                    if not motion_active:
                        print(f"\n[t={t:.2f}s] >>> 's' pressed: switching to hotdex motion tracking")
                        motion_active = True
                        interpolating = interp_steps > 0
                        need_seed_hotdex_history = True
                        interp_count = 0
                        motion_timestep = 0
                        motion_time_acc = 0.0
                        last_action_43[:] = 0.0
                        prev_obj_state["obj_pos"] = None
                        prev_obj_state["yaw_quat"] = None
                        freeze_object_state = False
                        cached_obj_pos_w = None
                        cached_obj_quat_w = None
                        auto_hotdex_start_time = None  # Disable auto-start if 's' pressed
                    else:
                        freeze_object_state = not freeze_object_state
                        if freeze_object_state:
                            cached_obj_pos_w = data.xpos[obj_body_id].astype(np.float32).copy()
                            cached_obj_quat_w = data.xquat[obj_body_id].astype(np.float32).copy()
                            print(
                                f"\n[t={t:.2f}s] Object state frozen. "
                                f"cached_pos={cached_obj_pos_w.tolist()} cached_quat={cached_obj_quat_w.tolist()}"
                            )
                        else:
                            cached_obj_pos_w = None
                            cached_obj_quat_w = None
                            print(f"\n[t={t:.2f}s] Object state unfrozen (live updates)")
                    key_pressed["s"] = False

                # ── Policy step (fixed decimation) ───────────────────────
                if step % steps_per_policy == 0:
                    policy_steps_since += 1
                    obs_flat = None
                    raw_action_43 = None

                    joint_pos, joint_vel, base_quat_wxyz, base_ang_vel, base_lin_vel, base_pos = \
                        read_mujoco_state(data, dof_qpos_addrs, dof_qvel_addrs)
                    proj_grav = quat_rotate_inverse(base_quat_wxyz, np.array([0, 0, -1], dtype=np.float32))

                    # Object state (optionally frozen)
                    if freeze_object_state and cached_obj_pos_w is not None:
                        obj_pos_w, obj_quat_w = cached_obj_pos_w, cached_obj_quat_w
                    else:
                        obj_pos_w = data.xpos[obj_body_id].astype(np.float32).copy()
                        obj_quat_w = data.xquat[obj_body_id].astype(np.float32).copy()

                    # Match `holosoma_inference.policies.wbt.WBTPolicy._compute_motion_obj_pos_rel_all`:
                    # the "heading frame" is defined by the yaw of the ref body (config: torso_link),
                    # not necessarily the free-base quaternion.
                    ref_body_quat_wxyz = data.xquat[ref_body_id].astype(np.float32).copy()
                    ref_yaw_quat = get_yaw_quat(ref_body_quat_wxyz)
                    if debug_motion_obj and motion_active and motion_timestep not in debug_motion_obj_logged:
                        if len(debug_motion_obj_logged) < 20 and (motion_timestep < 3 or motion_timestep in (9, 19, 39, 59, 99)):
                            debug_motion_obj_logged.add(int(motion_timestep))
                            _, _, torso_yaw = quat_to_rpy(ref_body_quat_wxyz)
                            base_index = int(np.clip(motion_timestep + 1, 0, motion_length - 1))
                            base_obj_pos = None
                            if motion_obj_pos is not None and motion_length > 0:
                                base_obj_pos = motion_obj_pos[base_index].astype(np.float32, copy=False).reshape(-1).tolist()
                            print(
                                "  [motion_obj_rel] "
                                f"t={int(motion_timestep)} base_index={base_index} "
                                f"torso_yaw_deg={float(np.degrees(torso_yaw)):.2f} "
                                f"yaw_quat_wxyz={ref_yaw_quat.reshape(-1).tolist()} "
                                f"base_obj_pos={base_obj_pos}",
                                flush=True,
                            )
                    ref_pos = base_pos.reshape(1, 3)

                    if not motion_active:
                        # ── STABILIZATION ──────────────────────────────────
                        obs = build_stabilization_obs(
                            joint_pos, joint_vel, base_quat_wxyz, base_ang_vel,
                            last_action_29, motion_cmd_seq, obs_idx,
                        )
                        raw_action = stab_session.run(None, {"obs": obs})[0][0]
                        last_action_29 = raw_action.copy()

                        target_q = scale_stabilization_action(
                            raw_action, joint_pos, act_idx, hand_indices,
                            act_scale_29, act_offset_29, act_clip_min_29, act_clip_max_29,
                            target_q_from_motion,
                        )

                    elif interpolating:
                        # ── INTERPOLATION to first motion frame ───────────
                        dof_pos_1d = joint_pos.reshape(1, -1)
                        interp_ratio = min(interp_count / interp_steps, 1.0)
                        target_q_2d = dof_pos_1d + (interp_target_config - dof_pos_1d) * interp_ratio
                        target_q = target_q_2d[0]
                        interp_count += 1

                        if interp_count >= interp_steps:
                            interpolating = False
                            print(f"[t={t:.2f}s] Interpolation complete, starting hotdex inference")
                            need_seed_hotdex_history = True

                    else:
                        # ── HOTDEX MOTION TRACKING ────────────────────────
                        hotdex_obs_args = (
                            joint_pos, joint_vel, base_lin_vel, base_ang_vel, proj_grav,
                            body_idx, hand_idx, obj_pos_w, obj_quat_w, mesh_offset,
                            motion_obj_pos, motion_obj_rot_wxyz, motion_timestep, motion_length,
                            ref_pos, ref_yaw_quat, last_action_43, prev_obj_state,
                        )

                        if need_seed_hotdex_history:
                            seed_history(build_hotdex_obs(*hotdex_obs_args), history_buffers)
                            need_seed_hotdex_history = False

                        obs_flat = update_history_and_flatten(
                            build_hotdex_obs(*hotdex_obs_args), history_buffers,
                        )

                        try:
                            raw_action_43 = hotdex_session.run(None, {"obs": obs_flat})[0]
                        except Exception as exc:  # noqa: BLE001
                            print(
                                f"[t={t:.2f}s] ERROR: hotdex onnxruntime failed: {exc}\n"
                                f"  obs_flat dtype={obs_flat.dtype} shape={obs_flat.shape}",
                                file=sys.stderr, flush=True,
                            )
                            raise
                        last_action_43 = raw_action_43.copy()

                        target_q = scale_hotdex_action(raw_action_43, config_to_action_order)

                        motion_time_acc += policy_dt
                        motion_timestep = int(motion_time_acc * motion_fps)
                        
                        # ── Update table position from motion trajectory ─────────────────
                        if motion_table_pos is not None and motion_timestep < motion_length:
                            table_pos_w = motion_table_pos[motion_timestep].astype(np.float32)
                            table_quat_w = (motion_table_rot_wxyz[motion_timestep].astype(np.float32)
                                          if motion_table_rot_wxyz is not None else None)
                            try:
                                _set_static_body_pose(
                                    model,
                                    body_name="table",
                                    pos_xyz=table_pos_w.tolist(),
                                    quat_wxyz=table_quat_w.tolist() if table_quat_w is not None else None
                                )
                                mujoco.mj_kinematics(model, data)  # Update body positions in data
                            except Exception as e:
                                if motion_timestep == 1:  # Only warn once
                                    print(f"Warning: Could not update table pose: {e}", flush=True)
                        
                        if args.debug_hotdex and (motion_timestep % hotdex_debug_every == 0):
                            mean_abs = float(np.mean(np.abs(raw_action_43)))
                            max_abs = float(np.max(np.abs(raw_action_43)))
                            tq_delta = float(np.max(np.abs(target_q - joint_pos)))
                            print(
                                f"[t={t:6.2f}s] hotdex_dbg: step={step} motion_t={motion_timestep} "
                                f"obs={tuple(obs_flat.shape)} action_mean_abs={mean_abs:.4g} "
                                f"action_max_abs={max_abs:.4g} max|target_q-q|={tq_delta:.4g}",
                                flush=True,
                            )
                    # ── Logging ─────────────────────────────────────────────
                    if inference_log_data is not None:
                        if obs_flat is None or raw_action_43 is None:
                            if not inference_log_wait_notice_printed:
                                inference_log_wait_notice_printed = True
                                print(
                                    f"[t={t:.2f}s] Inference logging is enabled, but HOTDEX inference "
                                    "has not run yet (stabilization/interpolation). "
                                    "Logging will start once HOTDEX starts.",
                                    flush=True,
                                )
                        else:
                            obs_full = obs_flat  # [1, 6975]

                            inference_log_data["command"].append(obs_full[0, :126].copy())
                            inference_log_data["task"].append(obs_full[0, 126:356].copy())
                            inference_log_data["points"].append(obs_full[0, 356:3428].copy())
                            inference_log_data["command_points"].append(obs_full[0, 3428:6500].copy())
                            inference_log_data["proprio_body"].append(obs_full[0, 6500:6835].copy())
                            inference_log_data["proprio_hand"].append(obs_full[0, 6835:].copy())

                            inference_log_data["policy_action"].append(raw_action_43.copy().flatten())
                            inference_log_data["motion_timestep"].append(motion_timestep)
                            inference_log_data["global_timestep"].append(step)

                            # world frame poses
                            inference_log_data["robot_pos_w"].append(base_pos.copy())
                            inference_log_data["robot_quat_w"].append(base_quat_wxyz.copy())
                            inference_log_data["object_pos_w"].append(obj_pos_w.copy())
                            inference_log_data["object_quat_w"].append(obj_quat_w.copy())

                            robot_origin = np.array([base_pos[0], base_pos[1], 0.0], dtype=np.float32)
                            inference_log_data["robot_world_origin_w"].append(robot_origin)
                    # ── Debug (every 2 sec) ───────────────────────────────
                    if step % 100 == 0:
                        h = data.qpos[2]
                        mode = "STAB" if not motion_active else ("INTERP" if interpolating else f"HOTDEX t={motion_timestep}")
                        print(f"[t={t:6.2f}s] base_z={h:.3f}m  mode={mode}")

                # ── PD control + physics step ─────────────────────────────
                apply_pd_control(data, target_q, dof_qpos_addrs, dof_qvel_addrs, actuator_ids)
                mujoco.mj_step(model, data)
                physics_steps_since += 1

                # ── Video capture (at policy frequency) ────────────────
                if video_recorder is not None and step % steps_per_policy == 0:
                    video_recorder.capture_frame(data)

                # ── Render / input pump (step decimation) ────────────────
                if step - last_render_step >= render_decimation:
                    viewer.sync()
                    last_render_step = step

                if args.debug_fps:
                    wall_now = time.perf_counter()
                    if wall_now - wall_last >= 1.0:
                        sim_now = float(data.time)
                        wall_dt = wall_now - wall_last
                        sim_dt = sim_now - sim_t_last
                        sim_hz = physics_steps_since / wall_dt
                        policy_hz = policy_steps_since / wall_dt
                        rtf = (sim_dt / wall_dt) if wall_dt > 0 else 0.0
                        print(
                            f"[fps] wall_dt={wall_dt:.2f}s sim_dt={sim_dt:.2f}s "
                            f"sim_hz={sim_hz:.1f} policy_hz={policy_hz:.1f} rtf={rtf:.2f}",
                            flush=True,
                        )
                        wall_last = wall_now
                        sim_t_last = sim_now
                        physics_steps_since = 0
                        policy_steps_since = 0

                # ── Optional real-time throttling ────────────────────────
                if target_rtf > 0.0:
                    wall_next_step += sim_dt / target_rtf
                    wall_now = time.perf_counter()
                    sleep_s = wall_next_step - wall_now
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    else:
                        # If we fall behind, reset the schedule to avoid runaway lag.
                        wall_next_step = wall_now

                step += 1
    except KeyboardInterrupt:
        print("\n[run] KeyboardInterrupt received, stopping simulation.", flush=True)
    finally:
        # ── Clean up temp XML file ──────────────────────────────
        if temp_xml_file is not None:
            temp_file_path = temp_xml_file if isinstance(temp_xml_file, str) else temp_xml_file.name
            try:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    print(f"✓ Cleaned up temp XML file: {temp_file_path}", flush=True)
            except Exception as e:
                print(f"Warning: Could not delete temp XML file {temp_file_path}: {e}", flush=True)
        
        # ── Save inference log ──────────────────────────────────
        if inference_log_data is not None:
            for k in inference_log_data:
                inference_log_data[k] = np.asarray(inference_log_data[k], dtype=np.float32)
            np.savez_compressed(inference_log_path, **inference_log_data)
            print(f"✓ Inference log saved: {inference_log_path.resolve()}", flush=True)

        # ── Save video ────────────────────────────────────────────
        if video_recorder is not None:
            sim_dt = float(model.opt.timestep)
            video_fps = 1.0 / (sim_dt * steps_per_policy) if sim_dt > 0 else 50.0
            video_recorder.save(fps=video_fps)
            video_recorder.cleanup()

# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default=None)
    parser.add_argument("--policy", type=str, default=None, help="Stabilization master policy")
    parser.add_argument("--hotdex-policy", type=str, default=None, help="Hotdex tracking policy")
    parser.add_argument("--pkl-path", type=str, default=None, help="Motion clip PKL file")
    parser.add_argument(
        "--clip-key", type=str, default="GRAB_s1_cubemedium_pass_1",
        help="Clip key in PKL (default: GRAB_s1_cubemedium_pass_1)",
    )
    parser.add_argument("--auto-hotdex", action="store_true",
                        help="Automatically start hotdex inference after 1 second (no need to press 's').")
    parser.add_argument("--debug-keys", action="store_true",
                        help="Print MuJoCo viewer key callback events.")
    parser.add_argument("--debug-hotdex", action="store_true",
                        help="Print hotdex inference diagnostics.")
    parser.add_argument(
        "--debug-motion-obj",
        action="store_true",
        help="Print motion object alignment + heading-yaw debug values for comparing with wbt.py.",
    )
    parser.add_argument("--interp-steps", type=int, default=0,
                        help="Number of 50Hz steps to interpolate before hotdex (default: 0).")
    parser.add_argument("--debug-fps", action="store_true",
                        help="Print achieved sim Hz / policy Hz and real-time factor.")
    parser.add_argument("--dt", type=float, default=None,
                        help="Override MuJoCo physics dt (seconds). Default: use MJCF <option timestep=...>.")
    parser.add_argument("--steps-per-policy", type=int, default=4,
                        help="Run policy every N physics steps (default: 4 → 50Hz when dt=0.005).")
    parser.add_argument("--render-decimation", type=int, default=4,
                        help="Call viewer.sync() every N physics steps (default: 4 → ~50Hz when dt=0.005).")
    parser.add_argument("--rtf", type=float, default=1.0,
                        help="Target real-time factor. 1.0=real-time, 0=unthrottled (default: 1.0).")
    parser.add_argument(
        "--no-motion-init",
        dest="motion_init",
        action="store_false",
        help="Disable motion-init: do not patch MJCF table/object pose from the motion clip first frame.",
    )
    parser.set_defaults(motion_init=True)
    parser.add_argument(
        "--no-motion-init-table",
        dest="motion_init_table",
        action="store_false",
        help="Do not apply motion-init to the MJCF 'table' body.",
    )
    parser.set_defaults(motion_init_table=True)
    parser.add_argument(
        "--no-motion-init-object",
        dest="motion_init_object",
        action="store_false",
        help="Do not apply motion-init to the MJCF 'object' body/freejoint.",
    )
    parser.set_defaults(motion_init_object=True)
    parser.add_argument(
        "--no-motion-init-align-to-root-xy-yaw",
        dest="motion_init_align_to_root_xy_yaw",
        action="store_false",
        help="Do not align motion table/object pose by subtracting root (x,y) and removing root yaw.",
    )
    parser.set_defaults(motion_init_align_to_root_xy_yaw=True)
    parser.add_argument(
        "--no-motion-init-apply-object-quat",
        dest="motion_init_apply_object_quat",
        action="store_false",
        help="Do not apply object quaternion from the motion clip first frame (keep MJCF quat).",
    )
    parser.set_defaults(motion_init_apply_object_quat=True)
    parser.add_argument(
        "--no-motion-init-apply-table-quat",
        dest="motion_init_apply_table_quat",
        action="store_false",
        help="Do not apply table quaternion from the motion clip first frame (keep MJCF quat).",
    )
    parser.set_defaults(motion_init_apply_table_quat=True)
    # ── Video recording ──
    parser.add_argument("--record", action="store_true",
                        help="Enable offscreen video recording (saved on exit).")
    parser.add_argument("--video-dir", type=str, default="logs/videos",
                        help="Directory for saved videos (default: logs/videos).")
    parser.add_argument("--video-width", type=int, default=640,
                        help="Video frame width in pixels (default: 640).")
    parser.add_argument("--video-height", type=int, default=360,
                        help="Video frame height in pixels (default: 360).")
    parser.add_argument("--video-format", type=str, default="h264", choices=["h264", "mp4"],
                        help="Video output format (default: h264).")
    parser.add_argument("--camera-pos", type=float, nargs=3, default=[-2.0, 0.0, 1.5],
                        help="Camera position [x y z] (default: 3.0 0.0 2.0).")
    parser.add_argument("--camera-target", type=float, nargs=3, default=[1.0, 0.0, 1.0],
                        help="Camera target [x y z] (default: 1.0 0.0 1.0).")
    args = parser.parse_args()

    root = Path(__file__).parent
    xml_path = args.xml or str(root / "src/holosoma/holosoma/data/robots/g1/g1_object/g1_43dof_cubemedium.xml")
    policy_path = args.policy or str(root / "src/holosoma_inference/holosoma_inference/models/wbt/base/master_policy.onnx")
    hotdex_policy_path = args.hotdex_policy or str(root / "src/holosoma_inference/holosoma_inference/models/wbt/object/cube_tracking_policy.onnx")
    pkl_path = args.pkl_path or str(root / "src/holosoma/holosoma/data/motions/g1_43dof/cubemediums_12_0116_with_text_traj.pkl")
    clip_key = "GRAB_s1_cubemedium_pass_1"

    xml_used = xml_path
    motion_init = None
    clip_key_used = clip_key
    temp_xml_file = None  # Keep temp file object alive
    table_pos_init = None
    table_quat_init = None
    
    if args.motion_init and (args.motion_init_table or args.motion_init_object):
        obj_name_hint = None
        stem = Path(xml_path).stem
        if stem.startswith("g1_43dof_"):
            obj_name_hint = stem[len("g1_43dof_") :]

        chosen_key, obj_pos, obj_quat, table_pos, table_quat = _load_motion_init_first_frame(
            pkl_path,
            clip_key=args.clip_key,
            obj_name_hint=obj_name_hint,
            align_to_root_xy_yaw=True,
        )
        clip_key_used = chosen_key
        table_pos_init = table_pos
        table_quat_init = table_quat if args.motion_init_apply_table_quat else None
        motion_init = {
            "clip_key": chosen_key,
            "obj_pos": obj_pos,
            "obj_quat": obj_quat if args.motion_init_apply_object_quat else None,
            "table_pos": table_pos,
            "table_quat": table_quat if args.motion_init_apply_table_quat else None,
        }
        xml_used, temp_xml_file = _patch_mjcf_motion_init(
            xml_path,
            clip_key=chosen_key,
            table_pos=table_pos if args.motion_init_table else None,
            table_quat_wxyz=motion_init["table_quat"] if args.motion_init_table else None,
            obj_pos=obj_pos if args.motion_init_object else None,
            obj_quat_wxyz=motion_init["obj_quat"] if args.motion_init_object else None,
        )
        print(
            f"Motion-init MJCF: clip='{chosen_key}' (temp file) "
            f"table_pos={None if table_pos is None else table_pos.tolist()} "
            f"obj_pos={None if obj_pos is None else obj_pos.tolist()}",
            flush=True,
        )

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(xml_used)
    if args.dt is not None:
        model.opt.timestep = float(args.dt)
    data = mujoco.MjData(model)
    print(f"Model loaded: {model.nq} qpos, {model.nv} qvel, {model.nu} actuators")
    print(f"  dt = {model.opt.timestep}s, gravity = {model.opt.gravity}")
    sim_dt = float(model.opt.timestep)
    physics_hz = 1.0 / sim_dt if sim_dt > 0 else float("nan")
    policy_hz = physics_hz / args.steps_per_policy if args.steps_per_policy > 0 else float("nan")
    render_hz = physics_hz / args.render_decimation if args.render_decimation > 0 else float("nan")
    print(
        f"  targets: physics={physics_hz:.1f}Hz | policy={policy_hz:.1f}Hz "
        f"(every {args.steps_per_policy} steps) | render~={render_hz:.1f}Hz "
        f"(every {args.render_decimation} steps) | rtf={args.rtf}"
    )

    dof_qpos_addrs, dof_qvel_addrs = _mj_hinge_qpos_qvel_addrs(model, DOF_NAMES)
    actuator_ids = _mj_actuator_ids(model, DOF_NAMES)

    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if obj_body_id < 0:
        raise KeyError("Body 'object' not found in MJCF.")
    print(f"  Object body id: {obj_body_id}")

    ref_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    if ref_body_id < 0:
        raise KeyError("Body 'torso_link' not found in MJCF (expected WBT ref body).")
    print(f"  Ref body id (torso_link): {ref_body_id}")

    # Load ONNX policies
    stab_session = onnxruntime.InferenceSession(policy_path)
    print(f"Stabilization policy: inputs={[i.name for i in stab_session.get_inputs()]}, "
          f"output={stab_session.get_outputs()[0].shape}")

    hotdex_session = onnxruntime.InferenceSession(hotdex_policy_path)
    print(f"Hotdex policy: inputs={[i.name for i in hotdex_session.get_inputs()]}, "
          f"output={hotdex_session.get_outputs()[0].shape}")

    # Load motion clip, object data, mesh points
    debug_motion_obj = bool(args.debug_motion_obj or int(os.getenv("TEST_WBT_DEBUG_MOTION_OBJ", "0")))
    motion = load_motion_clip(pkl_path, clip_key_used, mesh_root=root, debug_motion_obj=debug_motion_obj)

    # Initialize robot pose
    data.qpos[dof_qpos_addrs] = motion["init_q"]
    data.qvel[:] = 0.0

    # Redundant safety: if object is a freejoint, also set qpos directly (MJCF patch sets defaults only).
    if motion_init is not None and args.motion_init_object:
        try:
            _set_freejoint_pose(
                model,
                data,
                body_name="object",
                pos_xyz=None if motion_init["obj_pos"] is None else motion_init["obj_pos"].tolist(),
                quat_wxyz=None if motion_init["obj_quat"] is None else motion_init["obj_quat"].tolist(),
            )
        except Exception:
            pass

    # Initialize table pose from motion clip first frame (will be updated during hotdex)
    if table_pos_init is not None or (motion_init is not None and motion_init.get("table_pos") is not None):
        try:
            table_pos = table_pos_init if table_pos_init is not None else motion_init.get("table_pos")
            table_quat = table_quat_init if table_pos_init is not None else motion_init.get("table_quat")
            _set_static_body_pose(
                model,
                body_name="table",
                pos_xyz=None if table_pos is None else table_pos.tolist(),
                quat_wxyz=None if table_quat is None else table_quat.tolist(),
            )
        except Exception as e:
            print(f"Warning: Could not initialize table pose: {e}", flush=True)

    mujoco.mj_forward(model, data)

    # Run
    idx_maps = build_index_maps()

    # ── Inference logging init ─────────────────────────────────────
    log_dir = Path("logs/action_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_stem = Path(__file__).stem
    inference_log_path = log_dir / f"{script_stem}_inference_log_{timestamp}.npz"

    inference_log_data = {
        "command": [],
        "task": [],
        "points": [],
        "command_points": [],
        "proprio_body": [],
        "proprio_hand": [],
        "policy_action": [],
        "motion_timestep": [],
        "global_timestep": [],
        "robot_pos_w": [],
        "robot_quat_w": [],
        "object_pos_w": [],
        "object_quat_w": [],
        "robot_world_origin_w": [],
    }
    print(f"✓ Inference logging initialized: {inference_log_path.resolve()}")
    
    # ── Video recorder init ──────────────────────────────────────
    video_recorder = None
    if args.record:
        video_recorder = SimpleVideoRecorder(
            model,
            camera_pos=tuple(args.camera_pos),
            camera_target=tuple(args.camera_target),
            width=args.video_width,
            height=args.video_height,
            save_dir=args.video_dir,
            output_format=args.video_format,
        )
        print(f"✓ Video recording enabled: {args.video_width}x{args.video_height}, {args.video_format} format")
        print(f"  Camera pos: {tuple(args.camera_pos)}, target: {tuple(args.camera_target)}")
    
    run(model, data, args,
        stab_session=stab_session, hotdex_session=hotdex_session,
        idx_maps=idx_maps, motion=motion, obj_body_id=obj_body_id,
        ref_body_id=ref_body_id,
        dof_qpos_addrs=dof_qpos_addrs, dof_qvel_addrs=dof_qvel_addrs,
        actuator_ids=actuator_ids, inference_log_data=inference_log_data, inference_log_path=inference_log_path,
        temp_xml_file=temp_xml_file, video_recorder=video_recorder)
    # Note: video_recorder.save() is called in run() finally block, no need to call again


if __name__ == "__main__":
    main()

"""43-DOF locomotion policy using walk_prior_dr ONNX model.

Observation layout (138 dims) — identical to test_loco_mw.py build_obs():
  vel_command   (3)  : [lin_x, lin_y, ang_z]
  body_pos_rel  (29) : body joint pos relative to default (BODY_JOINT_NAMES order)
  body_vel      (29) : body joint vel
  base_ang_vel  (3)  : body-frame angular velocity
  proj_grav     (3)  : projected gravity vector
  hand_pos_rel  (14) : hand joint pos relative to default (HAND_JOINT_NAMES order)
  hand_vel      (14) : hand joint vel
  actions       (43) : last raw action (ACTION_ORDER_43DOF, before scale/offset)

Action processing — identical to test_loco_mw.py process_action():
  raw (ACTION_ORDER_43DOF) → raw * ACTION_SCALE + ACTION_OFFSET → clip
  → reorder to DOF_NAMES config order → absolute target_q sent to hardware
"""

from __future__ import annotations

import re

import numpy as np

from holosoma_inference.utils.math.quat import quat_rotate_inverse

from .locomotion import LocomotionPolicy

# ── Joint ordering constants ─────────────────────────────────────────────────

# 43-DOF config order (MuJoCo XML hinge-joint order)
_DOF_NAMES: list[str] = [
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

# Body joints (29): waist + legs + arms, in BODY_JOINT_NAMES order from g1_robot_common
_BODY_JOINT_NAMES: list[str] = [
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

# Hand joints (14): HAND_JOINT_NAMES order from g1_robot_common
_HAND_JOINT_NAMES: list[str] = [
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
]

# Policy action order (ACTION_ORDER_43DOF from g1_robot_common)
_ACTION_ORDER_43DOF: list[str] = [
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


def _name_indices(names: list[str], reference: list[str]) -> np.ndarray:
    lookup = {n: i for i, n in enumerate(reference)}
    return np.array([lookup[n] for n in names], dtype=np.int64)


# Pre-computed index arrays
_BODY_IDX = _name_indices(_BODY_JOINT_NAMES, _DOF_NAMES)     # (29,) body joints → DOF_NAMES idx
_HAND_IDX = _name_indices(_HAND_JOINT_NAMES, _DOF_NAMES)     # (14,) hand joints → DOF_NAMES idx
_ACTION_IN_DOF = _name_indices(_ACTION_ORDER_43DOF, _DOF_NAMES)  # (43,) action order → DOF_NAMES idx

# ── Motor & action scaling constants ─────────────────────────────────────────
# Mirrors g1_robot_common.py exactly

_ARMATURE_5020    = 0.003609725
_ARMATURE_7520_14 = 0.010177520
_ARMATURE_7520_22 = 0.025101925
_ARMATURE_4010    = 0.00425

_NATURAL_FREQ  = 10 * 2.0 * np.pi   # 10 Hz
_DAMPING_RATIO = 2.0

_KP_5020    = _ARMATURE_5020    * _NATURAL_FREQ ** 2
_KP_7520_14 = _ARMATURE_7520_14 * _NATURAL_FREQ ** 2
_KP_7520_22 = _ARMATURE_7520_22 * _NATURAL_FREQ ** 2
_KP_4010    = _ARMATURE_4010    * _NATURAL_FREQ ** 2


def _resolve(patterns: dict, names: list[str], default: float = 0.0) -> np.ndarray:
    result = np.full(len(names), default, dtype=np.float32)
    for i, name in enumerate(names):
        for pattern, value in patterns.items():
            if re.fullmatch(pattern, name):
                result[i] = value
                break
    return result


_DEFAULT_DOF_ANGLES = _resolve({
    ".*_hip_pitch_joint":        -0.312,
    ".*_knee_joint":              0.669,
    ".*_ankle_pitch_joint":      -0.363,
    ".*_elbow_joint":             0.6,
    "left_shoulder_pitch_joint":  0.2,
    "left_shoulder_roll_joint":   0.2,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
}, _DOF_NAMES)

_TORQUE_LIMITS = _resolve({
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
}, _DOF_NAMES)

_MOTOR_KP = _resolve({
    ".*_hip_pitch_joint":      _KP_7520_14,
    ".*_hip_yaw_joint":        _KP_7520_14,
    ".*_hip_roll_joint":       _KP_7520_22,
    ".*_knee_joint":           _KP_7520_22,
    ".*_ankle_.*_joint":       2 * _KP_5020,
    "waist_yaw_joint":         _KP_7520_14,
    "waist_roll_joint":        2 * _KP_5020,
    "waist_pitch_joint":       2 * _KP_5020,
    ".*_shoulder_pitch_joint": _KP_5020,
    ".*_shoulder_roll_joint":  _KP_5020,
    ".*_shoulder_yaw_joint":   _KP_5020,
    ".*_elbow_joint":          _KP_5020,
    ".*_wrist_roll_joint":     _KP_5020,
    ".*_wrist_pitch_joint":    _KP_4010,
    ".*_wrist_yaw_joint":      _KP_4010,
    ".*_hand_thumb_0_joint":   2.0,
    ".*_hand_.*_0_joint":      0.5,
    ".*_hand_.*":              0.5,
}, _DOF_NAMES)

# In ACTION_ORDER_43DOF (per-joint arrays, length 43)
_ao_idx = _name_indices(_ACTION_ORDER_43DOF, _DOF_NAMES)

_ACTION_SCALE    = (0.25 * _TORQUE_LIMITS / _MOTOR_KP)[_ao_idx]
_ACTION_OFFSET   = _DEFAULT_DOF_ANGLES[_ao_idx]
_ACTION_CLIP_MAX = (5.0  * _TORQUE_LIMITS / _MOTOR_KP)[_ao_idx]
_ACTION_CLIP_MIN = -_ACTION_CLIP_MAX

# Body/hand defaults in their respective joint orders
_BODY_DEFAULT = _DEFAULT_DOF_ANGLES[_BODY_IDX]   # (29,)
_HAND_DEFAULT = _DEFAULT_DOF_ANGLES[_HAND_IDX]   # (14,)


# ── Policy class ─────────────────────────────────────────────────────────────

class LocomotionPrior43DOF(LocomotionPolicy):
    """43-DOF locomotion policy driven by the walk_prior_dr ONNX model.

    Obs (138 dims) and action pipeline match test_loco_mw.py exactly.
    Keyboard/joystick controls (WASD / QE / =) are inherited from LocomotionPolicy.
    """

    def prepare_obs_for_rl(self, robot_state_data: np.ndarray) -> dict[str, np.ndarray]:
        """Build 138-dim obs in the exact layout expected by walk_prior_dr.

        robot_state_data layout for 43-DOF (shape = (1, 99)):
          [3:7]   base_quat wxyz
          [7:50]  dof_pos (43, DOF_NAMES config order)
          [50:53] base_lin_vel (world frame) — unused in obs
          [53:56] base_ang_vel (body frame)
          [56:99] dof_vel (43, DOF_NAMES config order)
        """
        base_quat = robot_state_data[:, 3:7]    # (1, 4)
        dof_pos   = robot_state_data[:, 7:50]   # (1, 43) DOF_NAMES order
        ang_vel   = robot_state_data[:, 53:56]  # (1, 3)  body frame
        dof_vel   = robot_state_data[:, 56:99]  # (1, 43) DOF_NAMES order

        proj_grav = quat_rotate_inverse(base_quat, np.array([[0.0, 0.0, -1.0]]))  # (1, 3)

        # vel_command: [lin_x, lin_y, ang_z]
        vel_cmd = np.concatenate(
            [self.lin_vel_command, self.ang_vel_command], axis=1
        )  # (1, 3)

        # Body joints (BODY_JOINT_NAMES order), relative to default
        body_pos = dof_pos[:, _BODY_IDX] - _BODY_DEFAULT  # (1, 29)
        body_vel = dof_vel[:, _BODY_IDX]                  # (1, 29)

        # Hand joints (HAND_JOINT_NAMES order), relative to default
        hand_pos = dof_pos[:, _HAND_IDX] - _HAND_DEFAULT  # (1, 14)
        hand_vel = dof_vel[:, _HAND_IDX]                  # (1, 14)

        # last_policy_action: (1, 43) raw action in ACTION_ORDER_43DOF (set in rl_inference)
        obs = np.concatenate([
            vel_cmd,                    # (1, 3)
            body_pos,                   # (1, 29)
            body_vel,                   # (1, 29)
            ang_vel,                    # (1, 3)
            proj_grav,                  # (1, 3)
            hand_pos,                   # (1, 14)
            hand_vel,                   # (1, 14)
            self.last_policy_action,    # (1, 43) ACTION_ORDER_43DOF
        ], axis=1)  # (1, 138)

        return {"obs": obs.astype(np.float32)}

    def rl_inference(self, robot_state_data: np.ndarray) -> np.ndarray:
        """ONNX inference with per-joint action scaling and order reordering.

        Returns scaled_policy_action (1, 43) in DOF_NAMES config order,
        representing absolute target joint positions.
        (Requires use_absolute_action=True in task config.)
        """
        obs = self.prepare_obs_for_rl(robot_state_data)

        # ONNX expects input name "obs" (not "actor_obs")
        raw_action = self.policy(obs)                      # (1, 43) ACTION_ORDER_43DOF
        raw_action = np.clip(raw_action, -100.0, 100.0)

        # Store raw action for next observation (ACTION_ORDER_43DOF)
        self.last_policy_action = raw_action.copy()

        # Per-joint: scale + offset + clip  (all in ACTION_ORDER_43DOF)
        processed = raw_action[0] * _ACTION_SCALE + _ACTION_OFFSET
        processed = np.clip(processed, _ACTION_CLIP_MIN, _ACTION_CLIP_MAX)

        # Reorder ACTION_ORDER_43DOF → DOF_NAMES (43-DOF config order)
        target_q = np.zeros(43, dtype=np.float32)
        target_q[_ACTION_IN_DOF] = processed

        self.scaled_policy_action = target_q.reshape(1, -1)
        return self.scaled_policy_action
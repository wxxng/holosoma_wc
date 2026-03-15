"""Utility for reading initial simulation state from motion-clip PKL files.

Used by ``run_sim.py`` to set the starting robot/object/table state to a selected
frame of a given motion clip.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from loguru import logger


@dataclasses.dataclass
class MotionInitPoses:
    """Initial state extracted from a selected frame of a motion clip."""

    clip_key: str
    """The resolved clip key that was used."""

    motion_start_timestep: int
    """The resolved motion frame index that was used."""

    object_pos: list[float] | None
    """Object position [x, y, z] in simulation world frame, or None if not available."""

    object_quat_wxyz: list[float] | None
    """Object orientation [w, x, y, z], or None if not available."""

    table_pos: list[float] | None = None
    """Table position [x, y, z] in simulation world frame, or None if not available."""

    table_quat_wxyz: list[float] | None = None
    """Table orientation [w, x, y, z], or None if not available."""

    robot_root_pos: list[float] | None = None
    """Robot root position [x, y, z] in simulation world frame, or None if not available."""

    robot_root_quat_wxyz: list[float] | None = None
    """Robot root orientation [w, x, y, z], or None if not available."""

    robot_joint_angles: dict[str, float] | None = None
    """Robot joint angles keyed by motion-joint name, or None if not available."""


G1_MOTION_JOINT_NAMES_29 = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

G1_DEX3_JOINT_NAMES_43 = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)


def _yaw_from_quat_xyzw(q: np.ndarray) -> float:
    """Extract yaw angle from an xyzw quaternion."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _rotate_xy(xy: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate a 2-D vector by yaw (radians)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * xy[0] - s * xy[1], s * xy[0] + c * xy[1]], dtype=np.float32)


def _normalize_timestep(frame_count: int, motion_start_timestep: int) -> int:
    """Validate and normalize the requested motion frame index."""
    if frame_count <= 0:
        raise ValueError("Motion clip does not contain any frames.")
    if motion_start_timestep < 0:
        raise ValueError(f"motion_start_timestep must be >= 0, got {motion_start_timestep}")
    if motion_start_timestep >= frame_count:
        raise ValueError(
            f"motion_start_timestep {motion_start_timestep} is out of range for clip with {frame_count} frames"
        )
    return int(motion_start_timestep)


def _quat_mul_wxyz(q1: list[float] | np.ndarray, q2: list[float] | np.ndarray) -> list[float]:
    """Multiply quaternions in wxyz convention."""
    w1, x1, y1, z1 = [float(v) for v in q1]
    w2, x2, y2, z2 = [float(v) for v in q2]
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def _normalize_quat_wxyz(quat_wxyz: list[float] | np.ndarray) -> list[float]:
    """Normalize a quaternion in wxyz convention."""
    quat = np.asarray(quat_wxyz, dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        return [1.0, 0.0, 0.0, 0.0]
    return (quat / norm).astype(np.float32).tolist()


def load_motion_clip_init_poses(
    pkl_path: str,
    clip_key: str | None = None,
    motion_start_timestep: int = 0,
    obj_name_hint: str | None = None,
    align_to_root_xy_yaw: bool = True,
    require_clip_key: bool = True,
) -> MotionInitPoses:
    """Load a selected-frame initial state from a motion-clip PKL file.

    Parameters
    ----------
    pkl_path : str
        Path to the PKL file produced by the motion-retargeting pipeline.
    clip_key : str | None
        Specific clip to use.  If None, the clip is auto-selected based on
        ``obj_name_hint`` (partial match) or the first available clip.
    obj_name_hint : str | None
        Substring hint used to auto-select a clip when ``clip_key`` is None.
    align_to_root_xy_yaw : bool
        If True, transform the object position so the robot starts at world
        origin (x=0, y=0) facing forward (yaw=0).
    require_clip_key : bool
        If True and ``clip_key`` is not found in the PKL, raise ``KeyError``.
        If False, fall back to the first available clip.

    Returns
    -------
    MotionInitPoses
        Resolved clip key plus the object's world-frame position / orientation.
    """
    try:
        import joblib  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("joblib is required to load motion-clip PKL files: pip install joblib") from e

    logger.info(f"Loading motion-clip init poses from: {pkl_path}")
    data = joblib.load(pkl_path)
    available_keys = list(data.keys())

    # ── Resolve clip key ──────────────────────────────────────────────────────
    if clip_key is None:
        if obj_name_hint:
            matching = [k for k in available_keys if obj_name_hint in k]
            if matching:
                clip_key = matching[0]
                logger.info(f"Auto-selected clip '{clip_key}' from obj_name_hint '{obj_name_hint}'")
            else:
                clip_key = available_keys[0]
                logger.warning(
                    f"No clip matched obj_name_hint '{obj_name_hint}'; using first clip '{clip_key}'"
                )
        else:
            clip_key = available_keys[0]
            logger.info(f"No clip_key specified; using first clip '{clip_key}'")
    elif clip_key not in data:
        if require_clip_key:
            raise KeyError(
                f"Clip key '{clip_key}' not found in PKL. Available: {available_keys}"
            )
        clip_key = available_keys[0]
        logger.warning(f"Clip key not found; falling back to '{clip_key}'")

    clip = data[clip_key]
    logger.info(f"Using clip '{clip_key}' (fps={clip.get('fps', '?')})")

    frame_count = 0
    if "dof_pos" in clip:
        frame_count = int(np.asarray(clip["dof_pos"]).shape[0])
    elif "obj_pos" in clip:
        frame_count = int(np.asarray(clip["obj_pos"]).shape[0])
    elif "table_pos" in clip:
        frame_count = int(np.asarray(clip["table_pos"]).shape[0])
    elif "global_translation_extend" in clip:
        frame_count = int(np.asarray(clip["global_translation_extend"]).shape[0])
    resolved_timestep = _normalize_timestep(frame_count, motion_start_timestep)
    logger.info(f"Using motion_start_timestep={resolved_timestep}")

    # ── Extract selected-frame object/table/robot state ───────────────────────
    obj_pos: list[float] | None = None
    obj_quat_wxyz: list[float] | None = None
    table_pos: list[float] | None = None
    table_quat_wxyz: list[float] | None = None
    robot_root_pos: list[float] | None = None
    robot_root_quat_wxyz: list[float] | None = None
    robot_joint_angles: dict[str, float] | None = None

    obj_pos_raw = clip.get("obj_pos")
    if obj_pos_raw is not None:
        arr = np.asarray(obj_pos_raw, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 3:
            obj_pos = arr[resolved_timestep].tolist()
        else:
            logger.warning(f"Unexpected obj_pos shape {arr.shape}; skipping position override.")

    obj_rot_raw = clip.get("obj_rot")
    if obj_rot_raw is not None:
        arr = np.asarray(obj_rot_raw, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 4:
            xyzw = arr[resolved_timestep]  # motion clips store xyzw
            obj_quat_wxyz = [float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])]
        else:
            logger.warning(f"Unexpected obj_rot shape {arr.shape}; skipping orientation override.")

    table_pos_raw = clip.get("table_pos")
    if table_pos_raw is not None:
        arr = np.asarray(table_pos_raw, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 3:
            table_pos = arr[resolved_timestep].tolist()
        else:
            logger.warning(f"Unexpected table_pos shape {arr.shape}; skipping table position override.")

    table_rot_raw = clip.get("table_rot")
    if table_rot_raw is not None:
        arr = np.asarray(table_rot_raw, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 4:
            xyzw = arr[resolved_timestep]  # motion clips store xyzw
            table_quat_wxyz = [float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])]
        else:
            logger.warning(f"Unexpected table_rot shape {arr.shape}; skipping table orientation override.")

    dof_pos_raw = clip.get("dof_pos")
    if dof_pos_raw is not None:
        arr = np.asarray(dof_pos_raw, dtype=np.float32)
        if arr.ndim == 2:
            joint_names: tuple[str, ...] | None = None
            if arr.shape[1] == len(G1_MOTION_JOINT_NAMES_29):
                joint_names = G1_MOTION_JOINT_NAMES_29
            elif arr.shape[1] == len(G1_DEX3_JOINT_NAMES_43):
                joint_names = G1_DEX3_JOINT_NAMES_43
            else:
                logger.warning(f"Unexpected dof_pos width {arr.shape[1]}; skipping robot joint init.")

            if joint_names is not None:
                robot_joint_angles = {
                    joint_name: float(angle)
                    for joint_name, angle in zip(joint_names, arr[resolved_timestep], strict=True)
                }
        else:
            logger.warning(f"Unexpected dof_pos shape {arr.shape}; skipping robot joint init.")

    # ── Align to robot starting at world origin ───────────────────────────────
    root_pos: np.ndarray | None = None
    root_quat_wxyz: list[float] | None = None
    root_yaw: float = 0.0

    if "global_translation_extend" in clip:
        root_traj = np.asarray(clip["global_translation_extend"], dtype=np.float32)
        # shape: [frames, bodies, 3] — body 0 is the root
        if root_traj.ndim == 3 and root_traj.shape[2] == 3:
            root_pos = root_traj[resolved_timestep, 0]
            robot_root_pos = root_pos.tolist()

    if "global_rotation_extend" in clip:
        root_rot_traj = np.asarray(clip["global_rotation_extend"], dtype=np.float32)
        # shape: [frames, bodies, 4] — xyzw quaternion
        if root_rot_traj.ndim == 3 and root_rot_traj.shape[2] == 4:
            root_quat_xyzw = root_rot_traj[resolved_timestep, 0]
            root_yaw = _yaw_from_quat_xyzw(root_quat_xyzw)
            root_quat_wxyz = [
                float(root_quat_xyzw[3]),
                float(root_quat_xyzw[0]),
                float(root_quat_xyzw[1]),
                float(root_quat_xyzw[2]),
            ]
            robot_root_quat_wxyz = root_quat_wxyz

    if align_to_root_xy_yaw and (root_pos is not None or root_quat_wxyz is not None):
        half = -root_yaw / 2.0
        yaw_remove_quat_wxyz = [float(np.cos(half)), 0.0, 0.0, float(np.sin(half))]

        if root_pos is not None:
            if obj_pos is not None:
                # Translate object so root XY → (0, 0), then rotate by -root_yaw.
                obj_arr = np.array(obj_pos, dtype=np.float32)
                dxy = obj_arr[:2] - root_pos[:2]
                aligned_xy = _rotate_xy(dxy, -root_yaw)
                obj_pos = [float(aligned_xy[0]), float(aligned_xy[1]), float(obj_arr[2])]
                logger.info(
                    f"Aligned object pos: root_xy={root_pos[:2]}, root_yaw={root_yaw:.3f} → obj_pos={obj_pos}"
                )

            if table_pos is not None:
                tbl_arr = np.array(table_pos, dtype=np.float32)
                dxy = tbl_arr[:2] - root_pos[:2]
                aligned_xy = _rotate_xy(dxy, -root_yaw)
                table_pos = [float(aligned_xy[0]), float(aligned_xy[1]), float(tbl_arr[2])]
                logger.info(
                    f"Aligned table pos: root_xy={root_pos[:2]}, root_yaw={root_yaw:.3f} → table_pos={table_pos}"
                )

            robot_root_pos = [0.0, 0.0, float(root_pos[2])]

        if obj_quat_wxyz is not None:
            obj_quat_wxyz = _normalize_quat_wxyz(_quat_mul_wxyz(yaw_remove_quat_wxyz, obj_quat_wxyz))

        if table_quat_wxyz is not None:
            table_quat_wxyz = _normalize_quat_wxyz(_quat_mul_wxyz(yaw_remove_quat_wxyz, table_quat_wxyz))

        if root_quat_wxyz is not None:
            robot_root_quat_wxyz = _normalize_quat_wxyz(_quat_mul_wxyz(yaw_remove_quat_wxyz, root_quat_wxyz))

    return MotionInitPoses(
        clip_key=clip_key,
        motion_start_timestep=resolved_timestep,
        object_pos=obj_pos,
        object_quat_wxyz=obj_quat_wxyz,
        table_pos=table_pos,
        table_quat_wxyz=table_quat_wxyz,
        robot_root_pos=robot_root_pos,
        robot_root_quat_wxyz=robot_root_quat_wxyz,
        robot_joint_angles=robot_joint_angles,
    )

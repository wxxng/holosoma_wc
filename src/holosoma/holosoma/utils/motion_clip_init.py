"""Utility for reading initial object/robot pose from motion-clip PKL files.

Used by run_sim.py to set the starting position of scene objects (e.g. cubemedium)
to the first frame of a given motion clip.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from loguru import logger


@dataclasses.dataclass
class MotionInitPoses:
    """Poses extracted from the first frame of a motion clip."""

    clip_key: str
    """The resolved clip key that was used."""

    object_pos: list[float] | None
    """Object position [x, y, z] in simulation world frame, or None if not available."""

    object_quat_wxyz: list[float] | None
    """Object orientation [w, x, y, z], or None if not available."""

    table_pos: list[float] | None = None
    """Table position [x, y, z] in simulation world frame, or None if not available."""

    table_quat_wxyz: list[float] | None = None
    """Table orientation [w, x, y, z], or None if not available."""


def _yaw_from_quat_xyzw(q: np.ndarray) -> float:
    """Extract yaw angle from an xyzw quaternion."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _rotate_xy(xy: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate a 2-D vector by yaw (radians)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * xy[0] - s * xy[1], s * xy[0] + c * xy[1]], dtype=np.float32)


def load_motion_clip_init_poses(
    pkl_path: str,
    clip_key: str | None = None,
    obj_name_hint: str | None = None,
    align_to_root_xy_yaw: bool = True,
    require_clip_key: bool = True,
) -> MotionInitPoses:
    """Load the first-frame object pose from a motion-clip PKL file.

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

    # ── Extract first-frame object and table positions ────────────────────────
    obj_pos: list[float] | None = None
    obj_quat_wxyz: list[float] | None = None
    table_pos: list[float] | None = None
    table_quat_wxyz: list[float] | None = None

    obj_pos_raw = clip.get("obj_pos")
    if obj_pos_raw is not None:
        arr = np.asarray(obj_pos_raw, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 3:
            obj_pos = arr[0].tolist()
        else:
            logger.warning(f"Unexpected obj_pos shape {arr.shape}; skipping position override.")

    obj_rot_raw = clip.get("obj_rot")
    if obj_rot_raw is not None:
        arr = np.asarray(obj_rot_raw, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 4:
            xyzw = arr[0]  # motion clips store xyzw
            obj_quat_wxyz = [float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])]
        else:
            logger.warning(f"Unexpected obj_rot shape {arr.shape}; skipping orientation override.")

    table_pos_raw = clip.get("table_pos")
    if table_pos_raw is not None:
        arr = np.asarray(table_pos_raw, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 3:
            table_pos = arr[0].tolist()
        else:
            logger.warning(f"Unexpected table_pos shape {arr.shape}; skipping table position override.")

    table_rot_raw = clip.get("table_rot")
    if table_rot_raw is not None:
        arr = np.asarray(table_rot_raw, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 4:
            xyzw = arr[0]  # motion clips store xyzw
            table_quat_wxyz = [float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])]
        else:
            logger.warning(f"Unexpected table_rot shape {arr.shape}; skipping table orientation override.")

    # ── Align to robot starting at world origin ───────────────────────────────
    if align_to_root_xy_yaw and (obj_pos is not None or table_pos is not None):
        root_pos: np.ndarray | None = None
        root_yaw: float = 0.0

        if "global_translation_extend" in clip:
            root_traj = np.asarray(clip["global_translation_extend"], dtype=np.float32)
            # shape: [frames, bodies, 3] — body 0 is the root
            if root_traj.ndim == 3 and root_traj.shape[2] == 3:
                root_pos = root_traj[0, 0]  # first frame, root body

        if "global_rotation_extend" in clip:
            root_rot_traj = np.asarray(clip["global_rotation_extend"], dtype=np.float32)
            # shape: [frames, bodies, 4] — xyzw quaternion
            if root_rot_traj.ndim == 3 and root_rot_traj.shape[2] == 4:
                root_quat_xyzw = root_rot_traj[0, 0]
                root_yaw = _yaw_from_quat_xyzw(root_quat_xyzw)

        if root_pos is not None:
            if obj_pos is not None:
                # Translate object so root XY → (0, 0), then rotate by -root_yaw
                obj_arr = np.array(obj_pos, dtype=np.float32)
                dxy = obj_arr[:2] - root_pos[:2]
                aligned_xy = _rotate_xy(dxy, -root_yaw)
                obj_pos = [float(aligned_xy[0]), float(aligned_xy[1]), float(obj_arr[2])]
                logger.info(
                    f"Aligned object pos: root_xy={root_pos[:2]}, root_yaw={root_yaw:.3f} → obj_pos={obj_pos}"
                )

            if table_pos is not None:
                # Same alignment for table
                tbl_arr = np.array(table_pos, dtype=np.float32)
                dxy = tbl_arr[:2] - root_pos[:2]
                aligned_xy = _rotate_xy(dxy, -root_yaw)
                table_pos = [float(aligned_xy[0]), float(aligned_xy[1]), float(tbl_arr[2])]
                logger.info(
                    f"Aligned table pos: root_xy={root_pos[:2]}, root_yaw={root_yaw:.3f} → table_pos={table_pos}"
                )

            # Rotate object quaternion by -root_yaw around Z if available
            if obj_quat_wxyz is not None:
                # Build rotation quaternion for -root_yaw around Z: [cos, 0, 0, -sin] in wxyz
                half = -root_yaw / 2.0
                rot_w = float(np.cos(half))
                rot_z = float(np.sin(half))
                # Multiply: q_aligned = q_rot * q_obj  (both wxyz)
                ow, ox, oy, oz = obj_quat_wxyz
                nw = rot_w * ow - rot_z * oz
                nx = rot_w * ox - rot_z * oy
                ny = rot_w * oy + rot_z * ox
                nz = rot_w * oz + rot_z * ow
                obj_quat_wxyz = [float(nw), float(nx), float(ny), float(nz)]

            # Rotate table quaternion by -root_yaw around Z if available
            if table_quat_wxyz is not None:
                half = -root_yaw / 2.0
                rot_w = float(np.cos(half))
                rot_z = float(np.sin(half))
                tw, tx, ty, tz = table_quat_wxyz
                nw = rot_w * tw - rot_z * tz
                nx = rot_w * tx - rot_z * ty
                ny = rot_w * ty + rot_z * tx
                nz = rot_w * tz + rot_z * tw
                table_quat_wxyz = [float(nw), float(nx), float(ny), float(nz)]

    return MotionInitPoses(
        clip_key=clip_key,
        object_pos=obj_pos,
        object_quat_wxyz=obj_quat_wxyz,
        table_pos=table_pos,
        table_quat_wxyz=table_quat_wxyz,
    )

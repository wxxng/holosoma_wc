"""Unit tests for motion-clip initialization helpers."""

from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.modules.setdefault(
    "loguru",
    types.SimpleNamespace(
        logger=types.SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None)
    ),
)

from holosoma.utils.motion_clip_init import load_motion_clip_init_poses


def _yaw_quat_xyzw(yaw_rad: float) -> np.ndarray:
    half = yaw_rad / 2.0
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=np.float32)


def _write_motion_pkl(path: Path) -> None:
    clip = {
        "fps": 50,
        "dof_pos": np.stack(
            [
                np.arange(43, dtype=np.float32),
                np.arange(100, 143, dtype=np.float32),
            ],
            axis=0,
        ),
        "obj_pos": np.array([[2.0, 2.0, 0.4], [4.0, 5.0, 0.6]], dtype=np.float32),
        "obj_rot": np.stack([_yaw_quat_xyzw(0.0), _yaw_quat_xyzw(np.pi / 2.0)], axis=0),
        "table_pos": np.array([[1.5, 3.0, 0.75], [3.0, 7.0, 0.8]], dtype=np.float32),
        "table_rot": np.stack([_yaw_quat_xyzw(0.0), _yaw_quat_xyzw(np.pi / 2.0)], axis=0),
        "global_translation_extend": np.array(
            [
                [[1.0, 2.0, 0.8]],
                [[3.0, 5.0, 0.9]],
            ],
            dtype=np.float32,
        ),
        "global_rotation_extend": np.array(
            [
                [_yaw_quat_xyzw(0.0)],
                [_yaw_quat_xyzw(np.pi / 2.0)],
            ],
            dtype=np.float32,
        ),
    }
    with open(path, "wb") as handle:
        pickle.dump({"clip_a": clip}, handle)


@pytest.fixture(autouse=True)
def fake_joblib_module(monkeypatch: pytest.MonkeyPatch):
    def _load(path: str):
        with open(path, "rb") as handle:
            return pickle.load(handle)

    monkeypatch.setitem(sys.modules, "joblib", types.SimpleNamespace(load=_load))


def test_load_motion_clip_init_poses_uses_selected_start_timestep(tmp_path: Path):
    pkl_path = tmp_path / "motion.pkl"
    _write_motion_pkl(pkl_path)

    poses = load_motion_clip_init_poses(
        str(pkl_path),
        clip_key="clip_a",
        motion_start_timestep=1,
        align_to_root_xy_yaw=True,
    )

    assert poses.clip_key == "clip_a"
    assert poses.motion_start_timestep == 1
    assert poses.object_pos == pytest.approx([0.0, -1.0, 0.6], abs=1e-6)
    assert poses.table_pos == pytest.approx([2.0, 0.0, 0.8], abs=1e-6)
    assert poses.object_quat_wxyz == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-6)
    assert poses.table_quat_wxyz == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-6)
    assert poses.robot_root_pos == pytest.approx([0.0, 0.0, 0.9])
    assert poses.robot_root_quat_wxyz == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-6)
    assert poses.robot_joint_angles is not None
    assert poses.robot_joint_angles["left_hip_pitch_joint"] == pytest.approx(100.0)
    assert poses.robot_joint_angles["right_hand_middle_1_joint"] == pytest.approx(142.0)


def test_load_motion_clip_init_poses_rejects_out_of_range_timestep(tmp_path: Path):
    pkl_path = tmp_path / "motion.pkl"
    _write_motion_pkl(pkl_path)

    with pytest.raises(ValueError, match="out of range"):
        load_motion_clip_init_poses(
            str(pkl_path),
            clip_key="clip_a",
            motion_start_timestep=2,
        )

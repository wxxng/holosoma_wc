from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
for package_root in (
    REPO_ROOT / "src" / "holosoma",
    REPO_ROOT / "src" / "holosoma_inference",
):
    package_root_str = str(package_root)
    if package_root.exists() and package_root_str not in sys.path:
        sys.path.insert(0, package_root_str)

from holosoma.config_types.simulator import VirtualGantryCfg
from holosoma.simulator.shared.virtual_gantry import VirtualGantry, create_virtual_gantry
from holosoma.utils.simulator_config import SimulatorType, set_simulator_type_enum


NOTEBOOK_ROOT_HEIGHT = 0.75
DEFAULT_NOTEBOOK_GANTRY_CFG = dataclasses.replace(
    VirtualGantryCfg(enabled=True),
    height=NOTEBOOK_ROOT_HEIGHT,
    length=0.0,
)


class NotebookMujocoSimulatorAdapter:
    """Expose the subset of BaseSimulator used by VirtualGantry."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.num_envs = 1
        self.device = torch.device("cpu")
        self.applied_forces = data.xfrc_applied
        self.body_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
            for body_id in range(model.nbody)
        ]

    @property
    def robot_root_states(self) -> torch.Tensor:
        root_state = np.zeros((1, 13), dtype=np.float32)
        root_state[0, 0:3] = self.data.qpos[0:3]
        root_state[0, 3:7] = self.data.qpos[3:7]
        root_state[0, 7:10] = self.data.qvel[0:3]
        root_state[0, 10:13] = self.data.qvel[3:6]
        return torch.from_numpy(root_state)

    def find_rigid_body_indice(self, body_name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id >= 0:
            return body_id
        raise RuntimeError(f"Body '{body_name}' not found in MuJoCo model.")


def _make_point_force_apply(
    sim: NotebookMujocoSimulatorAdapter,
    body_id: int,
    body_local_offset: np.ndarray,
) -> callable:
    """Return a drop-in replacement for VirtualGantry._apply_force_mujoco that applies
    the gantry force at a specific point on the body (instead of CoM).

    body_local_offset: 3D offset in the body's LOCAL frame (e.g. [0, 0, 0.28]).

    MuJoCo xfrc_applied[body_id] = [fx, fy, fz, tx, ty, tz] in world frame, applied at CoM.
    When force F acts at point p (world) and CoM is at c (world):
        torque_world = (p - c) × F
    """
    def _apply(link_id: int, force: np.ndarray) -> None:
        # body origin in world frame (data.xpos = CoM position in MuJoCo)
        # body_ipos = offset from body origin to CoM, in world frame = R @ ipos_local
        body_xpos  = sim.data.xpos[body_id]        # CoM world position
        body_xmat  = sim.data.xmat[body_id].reshape(3, 3)  # body rotation matrix (world frame)
        # world position of apply point = body_origin_world + R @ local_offset
        # body_origin_world = body_xpos - R @ ipos_local  (ipos_local = CoM offset in body frame)
        ipos_local = sim.model.body_ipos[body_id]   # CoM in body local frame
        body_origin_world = body_xpos - body_xmat @ ipos_local
        apply_point_world = body_origin_world + body_xmat @ body_local_offset

        r = apply_point_world - body_xpos  # moment arm from CoM
        torque = np.cross(r, force)

        sim.data.xfrc_applied[body_id, :3] = force
        sim.data.xfrc_applied[body_id, 3:6] = torque

    return _apply


def create_mujoco_virtual_gantry(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    cfg: VirtualGantryCfg | None = None,
    enabled: bool = True,
    attachment_body_name: str | None = None,
    force_local_offset: np.ndarray | None = None,
) -> tuple[NotebookMujocoSimulatorAdapter, VirtualGantry]:
    """Create the project virtual gantry on top of raw MuJoCo model/data.

    attachment_body_name: if given, force attachment to this specific body name
                          (overrides cfg.attachment_body_names search order).
    force_local_offset:   3D offset in the attachment body's LOCAL frame where the
                          gantry force is applied (e.g. np.array([0, 0, 0.28])).
                          If None, force is applied at the body CoM (default behaviour).
    """
    set_simulator_type_enum(SimulatorType.MUJOCO)
    sim = NotebookMujocoSimulatorAdapter(model, data)
    gantry_cfg = cfg if cfg is not None else DEFAULT_NOTEBOOK_GANTRY_CFG

    search_names = (
        [attachment_body_name] if attachment_body_name is not None
        else gantry_cfg.attachment_body_names
    )
    gantry = create_virtual_gantry(
        sim=sim,
        enable=enabled,
        attachment_body_names=search_names,
        cfg=gantry_cfg,
    )

    if force_local_offset is not None:
        # Monkey-patch the force application to use the specified body-local offset
        body_id = gantry.body_link_id
        gantry._apply_force_impl = _make_point_force_apply(
            sim, body_id, np.asarray(force_local_offset, dtype=np.float64)
        )

    return sim, gantry


def reset_robot_state_on_gantry(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    dof_qpos_addrs: np.ndarray,
    dof_qvel_addrs: np.ndarray,
    *,
    root_qpos_addr: int = 0,
    root_qvel_addr: int = 0,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray | None = None,
    root_quat_wxyz: np.ndarray | None = None,
    root_height: float = NOTEBOOK_ROOT_HEIGHT,
    root_xy: tuple[float, float] = (0.0, 0.0),
    root_lin_vel: np.ndarray | None = None,
    root_ang_vel: np.ndarray | None = None,
    gantry: VirtualGantry | None = None,
    sync_anchor: bool = True,
) -> None:
    """Reset the floating-base pose and joint state, then align gantry anchor."""

    data.qpos[root_qpos_addr:root_qpos_addr + 3] = np.array(
        [root_xy[0], root_xy[1], root_height], dtype=np.float64
    )
    if root_quat_wxyz is not None:
        data.qpos[root_qpos_addr + 3:root_qpos_addr + 7] = np.asarray(
            root_quat_wxyz, dtype=np.float64
        )

    data.qvel[root_qvel_addr:root_qvel_addr + 6] = 0.0
    if root_lin_vel is not None:
        data.qvel[root_qvel_addr:root_qvel_addr + 3] = np.asarray(
            root_lin_vel, dtype=np.float64
        )
    if root_ang_vel is not None:
        data.qvel[root_qvel_addr + 3:root_qvel_addr + 6] = np.asarray(
            root_ang_vel, dtype=np.float64
        )

    data.qpos[dof_qpos_addrs] = np.asarray(joint_pos, dtype=np.float64)
    if joint_vel is None:
        data.qvel[dof_qvel_addrs] = 0.0
    else:
        data.qvel[dof_qvel_addrs] = np.asarray(joint_vel, dtype=np.float64)

    mujoco.mj_forward(model, data)
    if gantry is not None and sync_anchor:
        gantry.set_position_to_robot()

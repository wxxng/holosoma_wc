#!/usr/bin/env python3
"""
Simulation Runner Script

This script provides a direct simulation runner for holosoma with bridge support without training
or evaluation environments.
"""

import dataclasses
import os
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import tyro
from loguru import logger

from holosoma.config_types.robot import RobotInitState
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.config_types.simulator import RigidObjectConfig
from holosoma.utils.eval_utils import init_eval_logging
from holosoma.utils.module_utils import get_holosoma_root
from holosoma.utils.sim_utils import DirectSimulation, setup_simulation_environment
from holosoma.utils.tyro_utils import TYRO_CONIFG


def _robot_xml_has_body(xml_path: str, body_name: str) -> bool:
    if not xml_path or not os.path.exists(xml_path):
        return False
    try:
        with open(xml_path, "r", encoding="utf-8") as handle:
            return f'name="{body_name}"' in handle.read()
    except OSError:
        return False


def _read_body_pose_from_robot_xml(xml_path: str, body_name: str) -> tuple[list[float], list[float]] | None:
    if not xml_path or not os.path.exists(xml_path):
        return None
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except (ET.ParseError, OSError):
        return None

    for body in root.iter("body"):
        if body.get("name") != body_name:
            continue
        pos_raw = body.get("pos")
        quat_raw = body.get("quat")
        pos = [float(v) for v in pos_raw.strip().split()] if pos_raw else [0.0, 0.0, 0.0]
        quat = [float(v) for v in quat_raw.strip().split()] if quat_raw else [1.0, 0.0, 0.0, 0.0]
        return pos, quat

    return None


def _maybe_apply_mujoco_motion_init(config: RunSimConfig) -> RunSimConfig:
    """Apply motion-clip initial state to MuJoCo robot/object setup when configured."""
    sim_cfg = config.simulator.config
    if sim_cfg.name != "mujoco":
        return config

    scene_cfg = sim_cfg.scene
    robot_asset = config.robot.asset
    asset_root = robot_asset.asset_root
    if asset_root.startswith("@holosoma/"):
        asset_root = asset_root.replace("@holosoma", get_holosoma_root())
    xml_file = robot_asset.xml_file
    xml_path = os.path.join(asset_root, xml_file) if xml_file else None
    has_object_body = bool(xml_path and _robot_xml_has_body(xml_path, "object"))
    object_pose = _read_body_pose_from_robot_xml(xml_path, "object") if has_object_body else None
    obj_pos_xml, obj_quat_xml = object_pose if object_pose else ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])

    poses = None
    motion_init_cfg = getattr(sim_cfg, "mujoco_motion_init", None)
    chosen_clip_key = None
    if motion_init_cfg and getattr(motion_init_cfg, "pkl_path", None):
        obj_name_hint = None
        stem = Path(xml_file).stem if xml_file else ""
        if stem.startswith("g1_43dof_"):
            obj_name_hint = stem[len("g1_43dof_"):]
        try:
            from holosoma.utils.motion_clip_init import load_motion_clip_init_poses  # noqa: PLC0415

            poses = load_motion_clip_init_poses(
                str(motion_init_cfg.pkl_path),
                clip_key=getattr(motion_init_cfg, "clip_key", None),
                motion_start_timestep=getattr(motion_init_cfg, "motion_start_timestep", 0),
                obj_name_hint=obj_name_hint,
                align_to_root_xy_yaw=getattr(motion_init_cfg, "align_to_root_xy_yaw", True),
                require_clip_key=getattr(motion_init_cfg, "require_clip_key", True),
            )
            chosen_clip_key = poses.clip_key
        except Exception as exc:  # noqa: BLE001 -- best-effort override
            logger.warning(f"Failed to load motion init pose from PKL; falling back to MJCF values. Error: {exc}")

    if chosen_clip_key and motion_init_cfg and getattr(motion_init_cfg, "clip_key", None) != chosen_clip_key:
        # Propagate resolved clip key so the MuJoCo scene builder can patch the MJCF consistently.
        new_motion_cfg = dataclasses.replace(motion_init_cfg, clip_key=chosen_clip_key)
        new_sim_init = dataclasses.replace(sim_cfg, mujoco_motion_init=new_motion_cfg)
        new_sim = dataclasses.replace(config.simulator, config=new_sim_init)
        config = dataclasses.replace(config, simulator=new_sim)
        sim_cfg = config.simulator.config
        motion_init_cfg = sim_cfg.mujoco_motion_init

    if poses is not None:
        init_state = config.robot.init_state
        joint_angles = dict(init_state.default_joint_angles)
        if poses.robot_joint_angles:
            allowed_joint_names = set(joint_angles.keys())
            joint_angles.update(
                {
                    joint_name: angle
                    for joint_name, angle in poses.robot_joint_angles.items()
                    if joint_name in allowed_joint_names
                }
            )

        root_pos = init_state.pos
        if poses.robot_root_pos is not None:
            root_pos = poses.robot_root_pos

        root_rot = init_state.rot
        if poses.robot_root_quat_wxyz is not None:
            w, x, y, z = poses.robot_root_quat_wxyz
            root_rot = [x, y, z, w]

        new_init_state = RobotInitState(
            pos=root_pos,
            rot=root_rot,
            lin_vel=init_state.lin_vel,
            ang_vel=init_state.ang_vel,
            default_joint_angles=joint_angles,
        )
        new_robot = dataclasses.replace(config.robot, init_state=new_init_state)
        config = dataclasses.replace(config, robot=new_robot)
        logger.info(
            "Applied robot init from motion clip "
            f"(clip='{poses.clip_key}', timestep={poses.motion_start_timestep})"
        )

    if scene_cfg.rigid_objects or not has_object_body:
        return config

    motion_obj_pos = poses.object_pos if poses is not None else None
    motion_obj_quat = poses.object_quat_wxyz if poses is not None else None
    obj_pos = motion_obj_pos if motion_obj_pos is not None else obj_pos_xml
    obj_quat = obj_quat_xml
    if motion_init_cfg and getattr(motion_init_cfg, "apply_object_quat", False) and motion_obj_quat is not None:
        obj_quat = motion_obj_quat

    logger.info("Detected 'object' body in robot MJCF; registering as scene rigid_object.")
    rigid_object = RigidObjectConfig(name="object", position=obj_pos, orientation=obj_quat)
    new_scene = dataclasses.replace(scene_cfg, rigid_objects=[rigid_object])
    new_sim_init = dataclasses.replace(sim_cfg, scene=new_scene)
    new_sim = dataclasses.replace(config.simulator, config=new_sim_init)
    return dataclasses.replace(config, simulator=new_sim)


def run_simulation(config: RunSimConfig):
    """Run simulation with direct simulator control.

    This function provides direct access to the simulator for continuous simulation
    with bridge support using the DirectSimulation class.

    Parameters
    ----------
    config : RunSimConfig
        Configuration containing all simulation settings.
    """
    # Auto-set device for GPU-accelerated backends if still on default CPU
    if config.device == "cpu":
        # Check if using Warp backend (requires CUDA)
        if hasattr(config.simulator.config, "mujoco_backend"):
            from holosoma.config_types.simulator import MujocoBackend  # noqa: PLC0415 -- deferred

            if config.simulator.config.mujoco_backend == MujocoBackend.WARP:
                logger.info("Auto-detected MuJoCo Warp backend - setting device to cuda:0")
                config = dataclasses.replace(config, device="cuda:0")

    config = dataclasses.replace(config, device=config.device)
    config = _maybe_apply_mujoco_motion_init(config)

    logger.info("Starting Holosoma Direct Simulation...")
    logger.info(f"Robot: {config.robot.asset.robot_type}")
    logger.info(f"Simulator: {config.simulator._target_}")
    logger.info(f"Terrain: {config.terrain.terrain_term.mesh_type} ({config.terrain.terrain_term.func})")

    try:
        # Use shared utils for setup
        env, device, simulation_app = setup_simulation_environment(config, device=config.device)

        # Create and run direct simulation using context manager for automatic clean-up
        with DirectSimulation(config, env, device, simulation_app) as sim:
            sim.run()

    except Exception as e:
        logger.error(f"Error during simulation: {e}")
        traceback.print_exc()
        sys.exit(1)


def main() -> None:
    """Main function using tyro configuration with compositional subcommands."""
    # Initialize logging
    init_eval_logging()

    logger.info("Holosoma Direct Simulation Runner")
    logger.info("Compositional configuration via subcommands (like eval_agent.py)")

    # Parse configuration with tyro - same pattern as ExperimentConfig
    config = tyro.cli(
        RunSimConfig,
        description="Run simulation with direct simulator control and bridge support.\n\n"
        "Usage: python -m holosoma.run_sim simulator:<sim> robot:<robot> terrain:<terrain>\n"
        "Examples:\n"
        "  python -m holosoma.run_sim # defaults \n"
        "  python -m holosoma.run_sim simulator:mujoco robot:t1_29dof_waist_wrist terrain:terrain_locomotion_plane\n"
        "  python -m holosoma.run_sim simulator:isaacgym robot:g1_29dof terrain:terrain_locomotion_mix",
        config=TYRO_CONIFG,
    )

    # Run simulation directly with parsed config
    run_simulation(config)


if __name__ == "__main__":
    main()

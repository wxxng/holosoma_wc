"""Default inference configurations for holosoma_inference."""

from dataclasses import replace
from importlib.metadata import entry_points

import tyro
from typing_extensions import Annotated

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_values import observation, robot, task

g1_29dof_loco = InferenceConfig(
    robot=robot.g1_29dof,
    observation=observation.loco_g1_29dof,
    task=task.locomotion,
)

t1_29dof_loco = InferenceConfig(
    robot=robot.t1_29dof,
    observation=observation.loco_t1_29dof,
    task=task.locomotion,
)

# fmt: off
g1_29dof_dancing = InferenceConfig(
    robot=replace(
        robot.g1_29dof,
        stiff_startup_pos=(
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # left leg
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # right leg
            0.0, 0.0, 0.0,                          # waist
            0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,      # left arm
            0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,     # right arm
        ),
        stiff_startup_kp=(
            350.0, 200.0, 200.0, 300.0, 300.0, 150.0,
            350.0, 200.0, 200.0, 300.0, 300.0, 150.0,
            200.0, 200.0, 200.0,
            40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
            40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
        ),
        stiff_startup_kd=(
            5.0, 5.0, 5.0, 10.0, 5.0, 5.0,
            5.0, 5.0, 5.0, 10.0, 5.0, 5.0,
            5.0, 5.0, 5.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
        ),
    ),
# fmt: on
    observation=observation.wbt_29dof_dancing,
    task=task.wbt,
)

g1_29dof_tracking = InferenceConfig(
    robot=replace(
        robot.g1_29dof,
        stiff_startup_pos=(
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # left leg
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # right leg
            0.0, 0.0, 0.0,  # waist
            0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,  # left arm
            0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,  # right arm
        ),
        stiff_startup_kp=(
            28.5012, 28.5012, 28.5012, 28.5012, 28.5012, 28.5012,  # left leg
            28.5012, 28.5012, 28.5012, 28.5012, 28.5012, 28.5012,  # right leg
            40.1792, 28.5012, 28.5012,  # waist
            14.2506, 14.2506, 14.2506, 14.2506, 14.2506, 16.7783, 16.7783,  # left arm
            14.2506, 14.2506, 14.2506, 14.2506, 14.2506, 16.7783, 16.7783,  # right arm
        ),
        stiff_startup_kd=(
            1.8144, 1.8144, 1.8144, 1.8144, 1.8144, 1.8144,  # left leg
            1.8144, 1.8144, 1.8144, 1.8144, 1.8144, 1.8144,  # right leg
            2.5579, 1.8144, 1.8144,  # waist
            0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 1.0681, 1.0681,  # left arm
            0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 1.0681, 1.0681,  # right arm
        ),
        motor_kp=(
            40.17923737, 99.09842682, 40.17923737, 99.09842682, 28.5012, 28.5012,  # left leg
            40.17923737, 99.09842682, 40.17923737, 99.09842682, 28.5012, 28.5012,  # right leg
            40.17923737, 28.50124550, 28.50124550,  # waist
            14.25062275, 14.25062275, 14.25062275, 14.25062275, 14.25062275, 16.77832794, 16.77832794,  # left arm
            14.25062275, 14.25062275, 14.25062275, 14.25062275, 14.25062275, 16.77832794, 16.77832794,  # right arm
        ),
        motor_kd=(
            1.81444573, 1.81444573, 1.81444573, 1.81444573, 1.81444573, 1.81444573,  # left leg
            1.81444573, 1.81444573, 1.81444573, 1.81444573, 1.81444573, 1.81444573,  # right leg
            2.55788970, 1.81444573, 1.81444573,  # waist
            0.90722287, 0.90722287, 0.90722287, 0.90722287, 0.90722287, 1.06814146, 1.06814146,  # left arm
            0.90722287, 0.90722287, 0.90722287, 0.90722287, 0.90722287, 1.06814146, 1.06814146,  # right arm
        ),
    ),
    observation=observation.wbt_29dof_tracking,
    task=task.wbt_tracking,
)

# G1 Whole-Body Tracking with Hands (43 DOF)
g1_43dof_object = InferenceConfig(
    robot=replace(
        robot.g1_43dof,
        stiff_startup_pos=(
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # left leg
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # right leg
            0.0, 0.0, 0.0,  # waist
            0.2, 0.6, 0.0, 0.1, 0.0, 0.0, 0.0,  # left arm
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # left hand
            0.2, -0.6, 0.0, 0.1, 0.0, 0.0, 0.0,  # right arm
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # right hand
        ),
        stiff_startup_kp=(
            28.5012, 28.5012, 28.5012, 28.5012, 28.5012, 28.5012,  # left leg
            28.5012, 28.5012, 28.5012, 28.5012, 28.5012, 28.5012,  # right leg
            40.1792, 28.5012, 28.5012,  # waist
            14.2506, 14.2506, 14.2506, 14.2506, 14.2506, 16.7783, 16.7783,  # left arm
            5.0000, 5.000, 5.000, 5.0000, 5.000, 5.000, 5.000,  # left hand
            14.2506, 14.2506, 14.2506, 14.2506, 14.2506, 16.7783, 16.7783,  # right arm
            5.0000, 5.0000, 5.0000, 5.0000, 5.0000, 5.0000, 5.0000,  # right hand
        ),
        stiff_startup_kd=(
            1.8144, 1.8144, 1.8144, 1.8144, 1.8144, 1.8144,  # left leg
            1.8144, 1.8144, 1.8144, 1.8144, 1.8144, 1.8144,  # right leg
            2.5579, 1.8144, 1.8144,  # waist
            0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 1.0681, 1.0681,  # left arm
            0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000,  # left hand
            0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 1.0681, 1.0681,  # right arm
            0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000,  # right hand
        ),

        
        # Add motor KP/KD for policy control
        motor_kp=(
            40.17923737, 99.09842682, 40.17923737, 99.09842682, 28.5012, 28.5012,  # left leg
            40.17923737, 99.09842682, 40.17923737, 99.09842682, 28.5012, 28.5012,  # right leg
            40.17923737, 28.50124550, 28.50124550,  # waist
            14.25062275, 14.25062275, 14.25062275, 14.25062275, 14.25062275, 16.77832794, 16.77832794,  # left arm
            2.0000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,  # left hand
            14.25062275, 14.25062275, 14.25062275, 14.25062275, 14.25062275, 16.77832794, 16.77832794,  # right arm
            2.0000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,  # right hand
        ),
        motor_kd=(
            1.81444573, 1.81444573, 1.81444573, 1.81444573, 1.81444573, 1.81444573,  # left leg
            1.81444573, 1.81444573, 1.81444573, 1.81444573, 1.81444573, 1.81444573,  # right leg
            2.55788970, 1.81444573, 1.81444573,  # waist
            0.90722287, 0.90722287, 0.90722287, 0.90722287, 0.90722287, 1.06814146, 1.06814146,  # left arm
            0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000,  # left hand
            0.90722287, 0.90722287, 0.90722287, 0.90722287, 0.90722287, 1.06814146, 1.06814146,  # right arm
            0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000,  # right hand
        ),
    ),
    observation=observation.wbt_43dof_object,
    task=task.wbt_tracking,
)

DEFAULTS = {
    "g1-29dof-loco": g1_29dof_loco,
    "t1-29dof-loco": t1_29dof_loco,
    "g1-29dof-wbt": g1_29dof_dancing,
    "g1-29dof-tracking" : g1_29dof_tracking,
    "g1-43dof-object" : g1_43dof_object
}

# Auto-discover inference configs from installed extensions
for ep in entry_points(group="holosoma.config.inference"):
    DEFAULTS[ep.name] = ep.load()

AnnotatedInferenceConfig = Annotated[
    InferenceConfig,
    tyro.conf.arg(
        constructor=tyro.extras.subcommand_type_from_defaults({f"inference:{k}": v for k, v in DEFAULTS.items()})
        ),
]

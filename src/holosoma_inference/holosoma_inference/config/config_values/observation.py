"""Default observation configurations for holosoma_inference.

This module provides pre-configured observation spaces for different
robot types and tasks, converted from the original YAML configurations.
"""

from __future__ import annotations

from importlib.metadata import entry_points

from holosoma_inference.config.config_types.observation import ObservationConfig

# =============================================================================
# Locomotion Observation Configurations
# =============================================================================

loco_g1_29dof = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "base_ang_vel",
            "projected_gravity",
            "command_lin_vel",
            "command_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
            "sin_phase",
            "cos_phase",
        ]
    },
    obs_dims={
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "command_lin_vel": 2,
        "command_ang_vel": 1,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "sin_phase": 2,
        "cos_phase": 2,
    },
    obs_scales={
        "base_lin_vel": 2.0,
        "base_ang_vel": 0.25,
        "projected_gravity": 1.0,
        "command_lin_vel": 1.0,
        "command_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
        "actions": 1.0,
        "sin_phase": 1.0,
        "cos_phase": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)

loco_t1_29dof = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "base_ang_vel",
            "projected_gravity",
            "command_lin_vel",
            "command_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
            "sin_phase",
            "cos_phase",
        ]
    },
    obs_dims={
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "command_lin_vel": 2,
        "command_ang_vel": 1,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "sin_phase": 2,
        "cos_phase": 2,
    },
    obs_scales={
        "base_lin_vel": 1.0,  # T1 uses 1.0 (vs G1's 2.0)
        "base_ang_vel": 1.0,  # T1 uses 1.0 (vs G1's 0.25)
        "projected_gravity": 1.0,
        "command_lin_vel": 1.0,
        "command_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.1,  # T1 uses 0.1 (vs G1's 0.05)
        "actions": 1.0,
        "sin_phase": 1.0,
        "cos_phase": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)


# =============================================================================
# 43-DOF Prior Locomotion Observation Configuration
# =============================================================================

loco_g1_43dof_prior = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "vel_command",        # (3)  [lin_x, lin_y, ang_z]  — identifies this policy type
            "joint_pos_body",     # (29) body joints relative to default
            "joint_vel_body",     # (29) body joint velocities
            "base_ang_vel",       # (3)
            "projected_gravity",  # (3)
            "joint_pos_hand",     # (14) hand joints relative to default
            "joint_vel_hand",     # (14) hand joint velocities
            "actions",            # (43) last raw action (ACTION_ORDER_43DOF)
        ]
    },
    obs_dims={
        "vel_command":       3,
        "joint_pos_body":   29,
        "joint_vel_body":   29,
        "base_ang_vel":      3,
        "projected_gravity": 3,
        "joint_pos_hand":   14,
        "joint_vel_hand":   14,
        "actions":          43,
    },
    obs_scales={
        "vel_command":       1.0,
        "joint_pos_body":    1.0,
        "joint_vel_body":    1.0,
        "base_ang_vel":      1.0,
        "projected_gravity": 1.0,
        "joint_pos_hand":    1.0,
        "joint_vel_hand":    1.0,
        "actions":           1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)

# =============================================================================
# WBT (Whole Body Tracking) Observation Configurations
# =============================================================================

wbt_29dof_dancing = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "motion_command",
            "motion_ref_ori_b",
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
        ]
    },
    obs_dims={
        "motion_command": 58,
        "motion_ref_pos_b": 3,
        "motion_ref_ori_b": 6,
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "actions": 1.0,
        "motion_command": 1.0,
        "motion_ref_pos_b": 1.0,
        "motion_ref_ori_b": 1.0,
        "base_lin_vel": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "robot_body_pos_b": 1.0,
        "robot_body_ori_b": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)

wbt_29dof_tracking = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "dof_pos",
            "dof_vel",
            "base_ang_vel",
            "projected_gravity",
            "actions",
            "motion_command_sequence",
        ]
    },
    obs_dims={
        "dof_pos": 29,
        "dof_vel": 29,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "actions": 29,
        "motion_command_sequence": 580,  # 10 frames × (29*2)
    },
    obs_scales={
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "base_ang_vel": 1.0,
        "projected_gravity": 1.0,
        "actions": 1.0,
        "motion_command_sequence": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)

wbt_43dof_object_bps = ObservationConfig(
    obs_dict={
        "task": [
            "obj_pos_rel",           # current obj pos in heading frame (3)
            "obj_rot_6d",            # current obj 6D rotation in heading frame (6)
        ],
        "proprio_hand": [
            "joint_pos_hand",        # (14)
            "joint_vel_hand",        # (14)
        ],
        "proprio_body": [
            "joint_pos_body",        # (29)
            "joint_vel_body",        # (29)
            "base_ang_vel",          # (3)
            "projected_gravity",     # (3)
        ],
        "command": [
            "obj_bps",               # BPS shape code (512)
            "motion_obj_pos_short",  # 10-frame short-horizon obj pos in heading frame (10×3=30)
            "motion_obj_ori_short",  # 10-frame short-horizon obj ori 6D (10×6=60)
            "motion_obj_pos_long",   # 5-frame long-horizon obj pos (5×3=15)
            "motion_obj_ori_long",   # 5-frame long-horizon obj ori 6D (5×6=30)
        ],
    },
    obs_dims={
        "obj_pos_rel":          3,
        "obj_rot_6d":           6,
        "joint_pos_hand":      14,
        "joint_vel_hand":      14,
        "joint_pos_body":      29,
        "joint_vel_body":      29,
        "base_ang_vel":         3,
        "projected_gravity":    3,
        "obj_bps":            512,
        "motion_obj_pos_short": 30,
        "motion_obj_ori_short": 60,
        "motion_obj_pos_long":  15,
        "motion_obj_ori_long":  30,
    },
    obs_scales={
        "obj_pos_rel":          1.0,
        "obj_rot_6d":           1.0,
        "joint_pos_hand":       1.0,
        "joint_vel_hand":       1.0,
        "joint_pos_body":       1.0,
        "joint_vel_body":       1.0,
        "base_ang_vel":         1.0,
        "projected_gravity":    1.0,
        "obj_bps":              1.0,
        "motion_obj_pos_short": 1.0,
        "motion_obj_ori_short": 1.0,
        "motion_obj_pos_long":  1.0,
        "motion_obj_ori_long":  1.0,
    },
    history_length_dict={
        "task":         5,
        "proprio_hand": 5,
        "proprio_body": 5,
        "command":      1,
    },
)
# Obs layout: task(45) + proprio_hand(140) + proprio_body(320) + command(647) = 1152
# command: bps(512) + short_pos(30) + short_ori(60) + long_pos(15) + long_ori(30)

wbt_43dof_object = ObservationConfig(
    obs_dict={
        "command": [
            "motion_obj_pos_rel_all",
            "motion_obj_ori_rel_all",
        ],
        "task": [
            "obj_pos_diff_b",
            "actions",
        ],
        "points": [
            "obj_pcd",
        ],
        "command_points": [
            "motion_obj_pcd",
        ],
        "proprio_body": [
            "joint_pos_body",
            "joint_vel_body",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
        ],
        "proprio_hand": [
            "joint_pos_hand",
            "joint_vel_hand",
        ],
    },
    obs_dims={
        "joint_pos_hand": 14,
        "joint_vel_hand": 14,
        "joint_pos_body": 29,
        "joint_vel_body": 29,
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "obj_pos_diff_b": 3,
        "actions": 43,
        "motion_obj_pos_rel_all": 42,
        "motion_obj_ori_rel_all": 84,
        "obj_pcd": 1024 * 3,
        "motion_obj_pcd": 1024 * 3,
    },
    obs_scales={
        "joint_pos_hand": 1.0,
        "joint_vel_hand": 1.0,
        "joint_pos_body": 1.0,
        "joint_vel_body": 1.0,
        "base_lin_vel": 1.0,
        "base_ang_vel": 1.0,
        "projected_gravity": 1.0,
        "obj_pos_diff_b": 1.0,
        "actions": 1.0,
        "motion_obj_pos_rel_all": 1.0,
        "motion_obj_ori_rel_all": 1.0,
        "obj_pcd": 1.0,
        "motion_obj_pcd": 1.0,
    },
    history_length_dict={
        "proprio_hand": 5,
        "proprio_body": 5,
        "task": 5,
        "command": 1,
        "points": 1,
        "command_points": 1,
    },
)

# Alias for wbt_object.py stabilization policy import
wbt = wbt_29dof_dancing

# =============================================================================
# Default Configurations Dictionary
# =============================================================================

DEFAULTS = {
    "loco-g1-29dof": loco_g1_29dof,
    "loco-t1-29dof": loco_t1_29dof,
    "loco-g1-43dof-prior": loco_g1_43dof_prior,
    "wbt-29dof-dancing": wbt_29dof_dancing,
    "wbt-29dof-tracking": wbt_29dof_tracking,
    "wbt-43dof-object": wbt_43dof_object,
    "wbt-43dof-object-bps": wbt_43dof_object_bps,
}
"""Dictionary of all available observation configurations.

Keys use hyphen-case naming convention for CLI compatibility.
"""

# Auto-discover observation configs from installed extensions
for ep in entry_points(group="holosoma.config.observation"):
    DEFAULTS[ep.name] = ep.load()

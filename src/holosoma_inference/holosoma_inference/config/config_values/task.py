"""Default task configurations for holosoma_inference."""

from __future__ import annotations

from holosoma_inference.config.config_types.task import TaskConfig

# Locomotion task
locomotion = TaskConfig(
    model_path="",  # Must be provided by user
    rl_rate=50,
    policy_action_scale=0.25,
    use_phase=True,
    gait_period=1.0,
    desired_base_height=0.75,
    residual_upper_body_action=False,
    domain_id=0,
    interface="lo",
    use_joystick=False,
    joystick_type="xbox",
    joystick_device=0,
    use_ros=False,
    wandb_download_dir="/tmp",
)

# Whole-body tracking task
wbt = TaskConfig(
    model_path="",  # Must be provided by user
    rl_rate=50,
    policy_action_scale=1.0,
    use_phase=False,
    gait_period=1.0,
    desired_base_height=0.75,
    residual_upper_body_action=False,
    domain_id=0,
    interface="lo",
    use_joystick=False,
    joystick_type="xbox",
    joystick_device=0,
    use_ros=False,
    wandb_download_dir="/tmp",
)

# 43-DOF prior locomotion: per-joint action scaling, rl_inference returns absolute target_q
locomotion_43dof_prior = TaskConfig(
    model_path="",  # Must be provided by user
    rl_rate=50,
    policy_action_scale=1.0,   # unused: rl_inference applies per-joint scale directly
    use_phase=False,
    gait_period=1.0,
    desired_base_height=0.75,
    residual_upper_body_action=False,
    domain_id=0,
    interface="lo",
    use_joystick=False,
    joystick_type="xbox",
    joystick_device=0,
    use_ros=False,
    wandb_download_dir="/tmp",
    use_absolute_action=True,  # rl_inference returns absolute target_q (not residual)
)

# Tracking task (g1-29dof-tracking, g1-43dof-object): policy returns absolute joint positions
wbt_tracking = TaskConfig(
    model_path="",  # Must be provided by user
    rl_rate=50,
    policy_action_scale=1.0,
    use_phase=False,
    gait_period=1.0,
    desired_base_height=0.75,
    residual_upper_body_action=False,
    domain_id=0,
    interface="lo",
    use_joystick=False,
    joystick_type="xbox",
    joystick_device=0,
    use_ros=False,
    wandb_download_dir="/tmp",
    use_absolute_action=True,
)

DEFAULTS = {
    "locomotion": locomotion,
    "locomotion-43dof-prior": locomotion_43dof_prior,
    "wbt": wbt,
    "wbt-tracking": wbt_tracking,
}

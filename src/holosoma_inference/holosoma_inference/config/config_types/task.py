"""Task configuration types for holosoma_inference."""

from __future__ import annotations

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class TaskConfig:
    """Task execution configuration for policy inference."""

    model_path: str | list[str]
    """Path to ONNX model(s). Supports local paths and wandb:// URIs. Required field."""

    rl_rate: float = 50
    """Policy inference rate in Hz."""

    policy_action_scale: float = 0.25
    """Scaling factor applied to policy actions."""

    use_phase: bool = True
    """Whether to use gait phase observations."""

    gait_period: float = 1.0
    """Gait cycle period in seconds."""

    domain_id: int = 0
    """DDS domain ID for communication."""

    interface: str = "lo"
    """Network interface name."""

    use_joystick: bool = False
    """Enable joystick control input."""

    joystick_type: str = "xbox"
    """Joystick type."""

    joystick_device: int = 0
    """Joystick device index."""

    use_sim_time: bool = False
    """Use synchronized simulation time for WBT policies."""

    wandb_download_dir: str = "/tmp"
    """Directory for downloading W&B checkpoints."""

    # Deprecation candidates:
    desired_base_height: float = 0.75
    """Target base height in meters."""

    residual_upper_body_action: bool = False
    """Whether to use residual control for upper body."""

    use_ros: bool = False
    """Use ROS2 for rate limiting."""

    print_observations: bool = False
    """Print observation vectors for debugging."""

    motion_pkl_path: str | None = None
    """Path to motion PKL file. If None, uses the default path in the policy."""

    motion_clip_key: str | None = None
    """Clip key to use from the motion PKL file. If None, uses the first clip."""

    motion_start_timestep: int = 0
    """Starting timestep for motion clip playback."""

    motion_end_timestep: int | None = None
    """Ending timestep for motion clip playback. If None, plays until the end."""

    use_absolute_action: bool = False
    """If True, policy outputs absolute joint positions (no default_dof_angles added)."""

    use_gen_traj: bool = False
    """Whether to use trajectory generator for object motion."""

    debug_traj_viz: bool = True
    """Enable short/long-horizon trajectory visualization in the MuJoCo viewer."""

    debug_traj_viz_port: int = 10006
    """UDP port used to stream trajectory visualization points to the simulator bridge."""

    change_loco_order: bool = False
    """If True, use alternative obs order for loco prior: [last_action, proprio_hand, proprio_body, vel_command]."""

    ignore_hand_action: bool = False
    """If True, override hand joint targets with default positions instead of policy output."""

    log: bool = False
    """Enable task-specific inference logging when supported by the active policy."""

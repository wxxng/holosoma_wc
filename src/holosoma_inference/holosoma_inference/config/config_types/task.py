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

    switch_hands: bool = False
    """Canonicalize swapped-hand DDS topics, including side swap, middle/index remap, and sign fixes."""

    hand_gain_scale: float = 1.0
    """Multiplicative scale applied to both kp and kd for both hands. E.g. 1.2 for 20% higher gains."""

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

    gen_traj_mode: str = "stay"
    """Trajectory generator mode. One of: lift, lift_down, right, left, left_down, left_back_down, back, lift_back_left_down, stay."""

    gen_traj_down_height_m: float = 0.3
    """For lift_down mode: how far to lower the object after lifting (meters). Default 0.3m."""

    gen_traj_trapezoid: bool = False
    """Use trapezoidal velocity profile for trajectory ramps. If False, use linear interpolation."""

    inference_time: bool = False
    """If True, print inference time for the first 10 steps after motion starts."""

    record_traj: bool = False
    """If True, allow recording an object trajectory in torso frame after stabilization."""

    use_recorded_traj: bool = False
    """If True, load and use a previously recorded world-frame object trajectory for commands."""

    recorded_traj_path: str | None = None
    """Optional path to a saved recorded object trajectory PKL.
    If None and use_recorded_traj=True, the newest file under logs/recorded_object_traj is used."""

    debug_traj_viz: bool = True
    """Enable short/long-horizon trajectory visualization in the MuJoCo viewer."""

    debug_traj_viz_port: int = 10006
    """UDP port used to stream trajectory visualization points to the simulator bridge."""

    rviz_traj: bool = False
    """Enable UDP streaming of sampled object trajectory poses for RViz visualization."""

    rviz_traj_host: str = "127.0.0.1"
    """Destination host for sampled RViz object trajectory UDP packets."""

    rviz_traj_port: int = 10007
    """UDP port used to stream sampled object trajectory poses for RViz visualization."""

    rviz_traj_dt: float = 0.2
    """Sampling interval in seconds for RViz object trajectory UDP packets."""

    change_loco_order: bool = False
    """If True, use alternative obs order for loco prior: [last_action, proprio_hand, proprio_body, vel_command]."""

    ignore_hand_action: bool = False
    """If True, override hand joint targets with default positions instead of policy output."""

    debug_hand: bool = False
    """If True on 43-DOF robots, hold body joints at current positions and repeat the Dex3 open/grip cycle on the hands."""

    debug_hand_demo: bool = False
    """If True on 43-DOF robots, hold body joints at zero and drive canonical finger groups 1-6 for switched/non-switched hand debugging."""

    debug_hand_action: str | None = None
    """Optional path to saved hand joint targets. When provided on 43-DOF robots, replays that sequence instead of the default debug_hand cycle."""

    reference_speed: float = 1.0
    """Playback speed multiplier for the reference motion trajectory
    (e.g. 2.0 = 2× faster, 0.5 = half speed).
    All motion arrays are resampled via linear interpolation."""

    log: bool = False
    """Enable task-specific inference logging when supported by the active policy."""

    object_dropout: bool = False
    """Simulate apriltag undetection by randomly dropping object observations."""

    object_dropout_prob: float = 0.3
    """Per-step probability of starting a dropout block each step (when not already in one)."""

    object_dropout_sec_min: float = 0.1
    """Minimum duration (seconds) of a simulated dropout block."""

    object_dropout_sec_max: float = 1.0
    """Maximum duration (seconds) of a simulated dropout block."""

    cache_world: bool = False
    """When True, cache detected world pose and re-project to current torso frame when apriltag is undetected.
    When False, just reuse the last torso-frame observation as-is."""

    world_pose_noise: float = 0.0
    """Simulate noisy world-pose estimation by adding uniform noise to torso_pos_w
    in motion command computation. Value is half-range: noise ~ U[-val, val] per xyz axis.
    0 means disabled."""

    mujoco_twin: bool = False
    """Enable real-time MuJoCo twin visualization by streaming robot state via UDP."""

    mujoco_twin_host: str = "127.0.0.2"
    """Destination host for MuJoCo twin UDP packets."""

    mujoco_twin_port: int = 10008
    """UDP port for MuJoCo twin state streaming."""

    fd_hand_vel: bool = False
    """Use finite-difference hand joint velocity (dof_pos difference / dt) instead of raw velocity sensor readings."""

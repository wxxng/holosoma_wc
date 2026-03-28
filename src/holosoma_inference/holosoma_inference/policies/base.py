from __future__ import annotations

import itertools
import json
import math
import sys
import threading
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import netifaces as ni
import numpy as np
import onnx
import onnxruntime
from loguru import logger
from sshkeyboard import listen_keyboard
from termcolor import colored

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_types.robot import RobotConfig
from holosoma_inference.sdk import create_interface
from holosoma_inference.sdk.interface_wrapper import InterfaceWrapper
from holosoma_inference.utils.latency import LatencyTracker
from holosoma_inference.utils.math.quat import quat_rotate_inverse
from holosoma_inference.utils.rate import RateLimiter
from holosoma_inference.utils.wandb import load_checkpoint

_DEBUG_HAND_DEMO_FINGER_GROUPS = {
    1: ("left_thumb", np.array([22, 23, 24], dtype=np.int64)),
    2: ("left_middle", np.array([25, 26], dtype=np.int64)),
    3: ("left_index", np.array([27, 28], dtype=np.int64)),
    4: ("right_thumb", np.array([36, 37, 38], dtype=np.int64)),
    5: ("right_middle", np.array([39, 40], dtype=np.int64)),
    6: ("right_index", np.array([41, 42], dtype=np.int64)),
}


class BasePolicy:
    """
    Base policy class for Holosoma deployment on humanoid robots.

    Supports both simulation and real robot deployment with keyboard/joystick controls.
    """

    def __init__(self, config: InferenceConfig):
        """Initialize the base policy with configuration and model."""
        self.config = config
        # Initialize robot config
        self._init_robot_config(self.config.robot)
        # Initialize SDK components
        self._init_sdk_components()
        # Initialize observation config
        self._init_obs_config()
        # Initialize communication components
        self._init_communication_components()
        # Initialize policy components
        self._init_policy_components(
            self.config.task.model_path, self.config.task.policy_action_scale, self.config.task.rl_rate
        )
        # Initialize command components
        self._init_command_components()
        # Initialize input handlers
        self._init_input_handlers()
        # Initialize phase components
        self._init_phase_components()
        # Initialize latency tracking
        self._init_latency_tracking()
        # Initialize safety monitoring
        self._init_safety_components()

    # ============================================================================
    # Initialization Methods
    # ============================================================================

    def _init_robot_config(self, robot_config: RobotConfig):
        """Initialize robot configuration and parameters."""
        self.robot_config = robot_config
        self.num_dofs = self.robot_config.num_joints
        self.default_dof_angles = np.array(self.robot_config.default_dof_angles)
        self.num_upper_dofs = robot_config.num_upper_body_joints

        # Initialize motor limits (only position limits are used)
        q_max = getattr(self.robot_config, "joint_pos_max", None)
        q_min = getattr(self.robot_config, "joint_pos_min", None)
        self.q_max_arr: np.array | None = np.array(q_max) if q_max is not None else None
        self.q_min_arr: np.array | None = np.array(q_min) if q_min is not None else None

        # Setup dof names and indices
        self._setup_dof_mappings()

    def _setup_dof_mappings(self):
        """Setup DOF names and their corresponding indices."""
        self.dof_names = self.robot_config.dof_names
        # TODO: Remove upper body mentions as it's not used anymore.
        self.upper_dof_names = self.robot_config.dof_names_upper_body
        self.lower_dof_names = self.robot_config.dof_names_lower_body

        # These are used by derived classes, so keep them
        if self.upper_dof_names:
            self.upper_dof_indices = [self.dof_names.index(dof) for dof in self.upper_dof_names]
        else:
            self.upper_dof_indices = []

        if self.lower_dof_names:
            self.lower_dof_indices = [self.dof_names.index(dof) for dof in self.lower_dof_names]
        else:
            self.lower_dof_indices = []

    def _init_sdk_components(self):
        """Additional SDK components initialization based on robot type."""
        self.sdk_type = self.robot_config.sdk_type
        if self.sdk_type == "booster":
            from booster_robotics_sdk import ChannelFactory

            ip = ni.ifaddresses(self.config.task.interface)[ni.AF_INET][0]["addr"]
            ChannelFactory.Instance().Init(self.config.task.domain_id, ip)
        else:
            pass  # No channel initialization needed for Unitree binding / other robots

    def _init_obs_config(self):
        """Initialize observation metadata and history buffers."""
        self.obs_config = self.config.observation
        self.obs_scales = self.obs_config.obs_scales
        self.obs_dims = self.obs_config.obs_dims
        self.obs_dict = self.obs_config.obs_dict
        self.obs_dim_dict = self._calculate_obs_dim_dict()
        self.history_length_dict = self.obs_config.history_length_dict

        # Initialize per-term history buffers using deques
        self._initialize_history_state()

    def _initialize_history_state(self):
        """Create per-term history deques and zero-initialized flattened buffers."""
        self.obs_history_buffers: dict[str, dict[str, deque[np.ndarray]]] = {}
        self.obs_terms_sorted: dict[str, list[str]] = {}
        self.obs_buf_dict: dict[str, np.ndarray] = {}

        for group, term_names in self.obs_dict.items():
            self.obs_terms_sorted[group] = sorted(term_names)
            history_len = self.history_length_dict.get(group, 1)
            self.obs_history_buffers[group] = {}
            flattened_terms: list[np.ndarray] = []

            for term in self.obs_terms_sorted[group]:
                term_dim = self.obs_dims[term]
                self.obs_history_buffers[group][term] = deque(maxlen=history_len)
                flattened_terms.append(np.zeros((1, term_dim * history_len), dtype=np.float32))

            self.obs_buf_dict[group] = np.concatenate(flattened_terms, axis=1) if flattened_terms else np.zeros((1, 0))

    def _policy_step_hook(self, robot_state_data):
        """Optional per-cycle hook for derived policies."""
        return None

    def _init_communication_components(self):
        """Initialize appropriate robot interface."""

        # Use InterfaceWrapper for all Unitree robots (including 29-DOF),
        # and keep the existing 43-DOF path for hand/full-state support.
        if self.robot_config.sdk_type == "unitree" or self.robot_config.num_joints == 43:
            self.interface = InterfaceWrapper(
                self.robot_config,
                domain_id=self.config.task.domain_id,
                interface_str=self.config.task.interface,
                use_joystick=self.config.task.use_joystick,
                use_hands=(self.robot_config.num_joints == 43),
                switch_hands=getattr(self.config.task, "switch_hands", False),
            )
        else:
            self.interface = create_interface(
                self.robot_config,
                self.config.task.domain_id,
                self.config.task.interface,
                self.config.task.use_joystick,
            )

    def _init_policy_components(self, model_path, policy_action_scale, rl_rate):
        """Initialize policy-related components."""
        self.policy_action_scale = policy_action_scale
        self.rl_rate = rl_rate
        self.model_paths = self._collect_model_paths(model_path)
        self._policy_states: list[dict] = []
        self.last_policy_action = np.zeros((1, self.num_dofs))
        self.scaled_policy_action = np.zeros((1, self.num_dofs))
        resolved_paths: list[str] = []

        for path in self.model_paths:
            local_path = self._resolve_model_path(str(path))
            resolved_paths.append(local_path)
            self.setup_policy(local_path)
            self._policy_states.append(self._capture_policy_state())

        self.model_paths = resolved_paths
        self.active_policy_index = 0
        self.active_model_path = None
        self._activate_policy(0, announce=False)

        # Determine KP/KD values: config override > ONNX metadata > error
        self._resolve_control_gains()
        self._default_kp = np.array(self.robot_config.motor_kp, dtype=np.float32)
        self._default_kd = np.array(self.robot_config.motor_kd, dtype=np.float32)

    def _collect_model_paths(self, model_path):
        """Normalize model_path into a list of up to nine entries."""
        if isinstance(model_path, (list, tuple)):
            paths = list(model_path)
        elif model_path is not None:
            paths = [model_path]
        else:
            paths = []

        paths = [str(path) for path in paths if path]
        if not paths:
            raise ValueError("At least one model_path must be provided for policy initialization.")
        if len(paths) > 9:
            # Error out instead of warning
            raise ValueError("Received more than nine model paths. Only up to nine model paths are supported.")
        return paths

    def _resolve_model_path(self, model_path: str) -> str:
        """Resolve model path, downloading from W&B if required."""
        if model_path.startswith(("wandb://", "https://")):
            download_dir = self.config.task.wandb_download_dir
            logger.info(f"Downloading checkpoint from W&B: {model_path}")
            checkpoint_path = load_checkpoint(None, model_path, download_dir)
            resolved_path = str(checkpoint_path)
            logger.info("Checkpoint downloaded to: %s", resolved_path)
            return resolved_path
        return model_path

    def _capture_policy_state(self) -> dict:
        """Capture the current policy state for later reuse."""
        return {
            "onnx_policy_session": self.onnx_policy_session,
            "onnx_input_names": self.onnx_input_names,
            "onnx_output_names": self.onnx_output_names,
            "policy_callable": self.policy,
            "onnx_kp": self.onnx_kp,
            "onnx_kd": self.onnx_kd,
        }

    def _restore_policy_state(self, state: dict):
        """Restore a previously captured policy state."""
        self.onnx_policy_session = state["onnx_policy_session"]
        self.onnx_input_names = state["onnx_input_names"]
        self.onnx_output_names = state["onnx_output_names"]
        self.policy = state["policy_callable"]
        self.onnx_kp = state["onnx_kp"]
        self.onnx_kd = state["onnx_kd"]

    def _activate_policy(self, index: int, announce: bool = True):
        """Activate a preloaded policy."""
        if not (0 <= index < len(self.model_paths)):
            return

        self._restore_policy_state(self._policy_states[index])
        self.last_policy_action.fill(0.0)
        self.scaled_policy_action.fill(0.0)
        self.active_policy_index = index
        self.active_model_path = self.model_paths[index]
        self._on_policy_switched(self.active_model_path)

        if announce and len(self.model_paths) > 1 and hasattr(self, "logger"):
            name = Path(self.active_model_path).name
            self.logger.info(colored(f"Switched to policy [{index + 1}]: {name}", "blue"))

    def _try_switch_policy_key(self, keycode: str) -> bool:
        """Switch policy slot if a numeric key is pressed."""
        if len(self.model_paths) <= 1:
            return False
        if not keycode.isdigit():
            return False
        slot = int(keycode)
        if slot == 0:
            return False
        index = slot - 1
        if index == self.active_policy_index:
            return True
        if 0 <= index < len(self.model_paths):
            self._activate_policy(index)
            return True
        return False

    def _on_policy_switched(self, model_path: str):
        """Hook for derived classes to reset state after loading a new policy."""
        _ = model_path

    def _init_command_components(self):
        """Initialize control-related components and commands."""
        self.use_policy_action = False
        self.init_count = 0
        self.get_ready_state = False
        self.desired_base_height = self.config.task.desired_base_height
        self.gait_period = self.config.task.gait_period

        # Initialize command arrays
        self.lin_vel_command = np.array([[0.0, 0.0]])
        self.ang_vel_command = np.array([[0.0]])
        self.stand_command = np.array([[0]])
        self.base_height_command = np.array([[self.desired_base_height]])

        # These are used by derived classes, so keep them
        self.waist_dofs_command = np.zeros((1, 3))
        self.phase_time = np.zeros((1, 1))

        # Upper body controller
        self.upper_body_controller = None

        # Pre-allocate command arrays for postprocessing
        self.cmd_q = np.zeros(self.num_dofs)
        self.cmd_dq = np.zeros(self.num_dofs)
        self.cmd_tau = np.zeros(self.num_dofs)
        self._prev_cmd_q = None
        self._cmd_q_max_delta = 0.1  # rad/step

        # Used by WBT to request stiff gains during startup interpolation.
        self._stiff_startup_active = False

        # Hand default positions for 43-DOF robots (when using 29-DOF policy)
        # Config order: left_hand(7), right_hand(7)
        self.hand_default_pos = np.array([
            # Left hand (7 joints)
            0.0, 1.0, 1.7, -1.57, -1.7, -1.57, -1.7,
            # Right hand (7 joints)
            0.0, -1.0, -1.7, 1.57, 1.7, 1.57, 1.7
        ], dtype=np.float32)

        # Finite-difference hand velocity: use (dof_pos - prev_dof_pos) * rl_rate for hand joints
        self._fd_hand_vel = getattr(self.config.task, "fd_hand_vel", False)
        self._fd_hand_vel_prev_dof_pos = None  # populated on first observation step
        if self._fd_hand_vel:
            logger.info(colored("Finite-difference hand vel enabled (pos diff / dt)", "cyan"))

        self._debug_hand_half_cycle_steps = max(1, int(round(4.0 * self.rl_rate)))
        self._debug_hand_demo_cycle_steps = max(1, int(round(2.0 * self.rl_rate)))
        self._debug_hand_hand_idx = np.array(list(range(22, 29)) + list(range(36, 43)), dtype=np.int64)
        self._debug_hand_body_hold_q = np.zeros(self.num_dofs, dtype=np.float32)
        stiff_kp_source = self.robot_config.stiff_startup_kp or self.robot_config.motor_kp
        stiff_kd_source = self.robot_config.stiff_startup_kd or self.robot_config.motor_kd
        self._debug_hand_kp = np.array(stiff_kp_source, dtype=np.float32) if stiff_kp_source is not None else None
        self._debug_hand_kd = np.array(stiff_kd_source, dtype=np.float32) if stiff_kd_source is not None else None
        self._debug_hand_open_pos = np.array([
            0.0, -0.65, 0.1, -0.1, -0.1, -0.1, -0.1,
            0.0, 0.65, -0.1, 0.1, 0.1, 0.1, 0.1,
        ], dtype=np.float32)
        self._debug_hand_grip_pos = np.array([
            0.0, 1.0, 1.6, -1.5, -1.7, -1.5, -1.7,
            0.0, -1.0, -1.6, 1.5, 1.7, 1.5, 1.7,
        ], dtype=np.float32)
        self._debug_hand_start_pos = self._debug_hand_open_pos.copy()
        self._debug_hand_current_pos = self._debug_hand_open_pos.copy()
        self._debug_hand_step = 0
        self._debug_hand_demo_step = 0
        self._debug_hand_is_opening = True
        self._debug_hand_initialized = False
        self.debug_hand = bool(getattr(self.config.task, "debug_hand", False))
        self.debug_hand_demo = bool(getattr(self.config.task, "debug_hand_demo", False))
        self.debug_hand_action = getattr(self.config.task, "debug_hand_action", None)
        self._debug_hand_action_path: Path | None = None
        self._debug_hand_action_sequence: np.ndarray | None = None
        self._debug_hand_action_source_hz = float(self.rl_rate)
        self._debug_hand_action_phase = 0.0
        self._debug_hand_demo_selected_finger = 1

        if self.debug_hand and self.num_dofs != 43:
            logger.warning("debug_hand is only supported for 43-DOF robots. Ignoring flag for this configuration.")
        if self.debug_hand_demo and self.num_dofs != 43:
            logger.warning("debug_hand_demo is only supported for 43-DOF robots. Ignoring flag for this configuration.")
        if self.debug_hand_action and self.num_dofs != 43:
            logger.warning("debug_hand_action is only supported for 43-DOF robots. Ignoring flag for this configuration.")

        if self.num_dofs == 43 and self.debug_hand_action:
            loaded_debug_hand_action = self._load_debug_hand_action(self.debug_hand_action)
            if loaded_debug_hand_action is not None:
                (
                    self._debug_hand_action_sequence,
                    self._debug_hand_action_source_hz,
                    self._debug_hand_action_path,
                ) = loaded_debug_hand_action

        self._debug_hand_action_enabled = self._debug_hand_action_sequence is not None
        self._debug_hand_demo_enabled = self.num_dofs == 43 and self.debug_hand_demo
        self._debug_hand_enabled = self.num_dofs == 43 and (
            self.debug_hand or self._debug_hand_demo_enabled or self._debug_hand_action_enabled
        )

    def _init_phase_components(self):
        """Initialize phase components."""
        self.use_phase = self.config.task.use_phase
        if self.use_phase:
            self.phase = np.zeros((1, 2))
            self.phase[:, 0] = 0.0  # left foot starts at 0
            self.phase[:, 1] = np.pi  # right foot starts at pi
            self.phase_dt = 2 * np.pi / (self.rl_rate * self.gait_period)

    def _init_latency_tracking(self):
        """Initialize latency tracking components."""
        self.latency_tracker = LatencyTracker(window_size=int(self.rl_rate))

    def _init_input_handlers(self):
        """Initialize input handlers (ROS, joystick, keyboard)."""
        self._init_rate_handler()
        self._init_input_device()

    def _init_rate_handler(self):
        """Initialize ROS handler if enabled."""
        self.rl_rate = self.config.task.rl_rate
        if self.config.task.use_ros:
            import rclpy

            rclpy.init(args=None)
            self.node = rclpy.create_node("policy_node")
            self.logger = self.node.get_logger()
            self.rate = self.node.create_rate(self.rl_rate)
            thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
            thread.start()
        else:
            self.logger = logger
            self.rate = RateLimiter(self.rl_rate)

    def _init_input_device(self):
        """Initialize input device (joystick or keyboard)."""
        if self.config.task.use_joystick:
            self._init_joystick_handler()
        else:
            self._init_keyboard_handler()

    def _init_joystick_handler(self):
        """Initialize joystick handler."""
        if sys.platform == "darwin":
            self.logger.warning("Joystick is not supported on Windows or Mac.")
            self.logger.warning("Using keyboard instead")
            self.use_joystick = False
            self._init_keyboard_handler()
        else:
            self.logger.info("Using joystick")
            self.use_joystick = True

    def _init_keyboard_handler(self):
        """Initialize keyboard handler."""
        self.logger.info("Using keyboard")
        self.use_joystick = False
        # Check if running in a TTY environment
        if not sys.stdin.isatty():
            self.logger.warning("Not running in a TTY environment - keyboard input disabled")
            self.logger.warning("This is normal for automated tests or non-interactive environments")
            self.logger.info("Auto-starting policy in non-interactive mode")
            self.use_policy_action = True
            return
        # Start keyboard listener in a daemon thread
        threading.Thread(target=self.start_key_listener, daemon=True).start()
        self.logger.info("Keyboard Listener Initialized")

    def _init_safety_components(self):
        """Initialize safety monitoring components."""
        if hasattr(self.robot_config, 'dof_effort_limit_list') and self.robot_config.dof_effort_limit_list:
            self.safety_torque_limits = np.array(self.robot_config.dof_effort_limit_list) * 0.8
        else:
            default_limit = self.config.task.safety_torque_limit if hasattr(self.config.task, 'safety_torque_limit') else 30.0
            self.safety_torque_limits = np.full(self.num_dofs, default_limit)

        self.orientation_safety_threshold = self.config.task.orientation_safety_threshold if hasattr(self.config.task, 'orientation_safety_threshold') else -0.5

        self.safety_enabled = False
        self.damping_mode_active = False

        self.logger.info(f"Safety system initialized - Orientation threshold: {self.orientation_safety_threshold:.2f}")

    def activate_damping_mode(self):
        """Activate damping mode to safely stop the robot."""
        if self.damping_mode_active:
            return

        self.damping_mode_active = True
        self.use_policy_action = False
        self.get_ready_state = False

        self.logger.error(colored("!!! DAMPING MODE ACTIVATED !!!", "red", attrs=["bold"]))

        damping_kp = np.zeros(self.num_dofs)
        damping_kd = np.full(self.num_dofs, 3.0)
        damping_q = np.zeros(self.num_dofs)
        damping_dq = np.zeros(self.num_dofs)
        damping_tau = np.zeros(self.num_dofs)

        if self.num_dofs == 43:
            robot_state_data = self.interface.get_full_state_43dof()
        else:
            robot_state_data = self.interface.get_low_state()

        current_q = robot_state_data[0, 7 : 7 + self.num_dofs]

        if self.num_dofs == 43:
            self.interface.send_full_command_43dof(
                damping_q, damping_dq, damping_tau, current_q,
                kp_override_43=damping_kp, kd_override_43=damping_kd,
            )
        else:
            self.interface.send_low_command(
                damping_q, damping_dq, damping_tau, current_q,
                kp_override=damping_kp, kd_override=damping_kd,
            )

    def is_unsafe(self, robot_state_data, projected_gravity):
        """Check if robot is in an unsafe state."""
        if not self.safety_enabled or self.damping_mode_active:
            return False

        if self.use_joystick and hasattr(self, 'key_states') and self.key_states:
            if self.key_states.get('L2+B', False):
                self.logger.error(colored("[SAFETY TRIGGER] Manual emergency stop (L2+B) pressed!", "red", attrs=["bold"]))
                return True

        if projected_gravity[0, 2] > self.orientation_safety_threshold:
            self.logger.error(
                colored(f"[SAFETY TRIGGER] Robot is upside down! projected_gravity_z: {projected_gravity[0, 2]:.3f}", "red", attrs=["bold"])
            )
            return True

        return False

    # ============================================================================
    # Policy Methods
    # ============================================================================

    def setup_policy(self, model_path):
        """Setup ONNX policy model and extract metadata."""
        self.onnx_policy_session = onnxruntime.InferenceSession(model_path)
        input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        output_names = [out.name for out in self.onnx_policy_session.get_outputs()]

        self.onnx_input_names = input_names
        self.onnx_output_names = output_names

        # Extract metadata from ONNX model (hard fault if fails)
        onnx_model = onnx.load(model_path)
        metadata = {}
        for prop in onnx_model.metadata_props:
            metadata[prop.key] = json.loads(prop.value)

        # Extract KP/KD from metadata (will be None if not present)
        self.onnx_kp = np.array(metadata["kp"]) if "kp" in metadata else None
        self.onnx_kd = np.array(metadata["kd"]) if "kd" in metadata else None

        if self.onnx_kp is not None:
            logger.info(f"Loaded KP/KD from ONNX metadata: {Path(model_path).name}")

        def policy_act(obs_dict):
            # For example,obs_dict contains:
            # {
            #     'actor_obs_lower_body': np.array([...]),
            #     'actor_obs_upper_body': np.array([...]),
            #     'estimator_obs': np.array([...])
            # }
            input_feed = {name: obs_dict[name] for name in self.onnx_input_names}
            outputs = self.onnx_policy_session.run(self.onnx_output_names, input_feed)
            return outputs[0]  # just return outputs[0] as only "action" is needed

        self.policy = policy_act

    def _resolve_control_gains(self):
        """Resolve KP/KD values with priority: config override > ONNX metadata > error.

        Creates a new config instance with resolved values if needed.
        """
        # Check if config has explicit KP/KD values
        config_has_kp = hasattr(self.robot_config, "motor_kp") and self.robot_config.motor_kp is not None
        config_has_kd = hasattr(self.robot_config, "motor_kd") and self.robot_config.motor_kd is not None

        if config_has_kp and config_has_kd:
            # Config already has values (override) - nothing to do
            logger.info(colored("Using KP/KD from config (override)", "yellow"))
            kp_values = np.array(self.robot_config.motor_kp)
            kd_values = np.array(self.robot_config.motor_kd)
        elif self.onnx_kp is not None and self.onnx_kd is not None:
            # Use ONNX metadata (default) - create new config with values
            logger.info(colored("Using KP/KD from ONNX metadata", "green"))
            kp_values = self.onnx_kp
            kd_values = self.onnx_kd
            # Create new config instance with ONNX values
            self.robot_config = replace(
                self.robot_config, motor_kp=tuple(kp_values.tolist()), motor_kd=tuple(kd_values.tolist())
            )
            # Propagate updated config to active interface since replace() creates a new object.
            if hasattr(self.interface, "update_config"):
                self.interface.update_config(self.robot_config)
            else:
                self.interface.robot_config = self.robot_config
            # Update sdk2py backend components (booster SDK only)
            if getattr(self.interface, "backend", None) == "sdk2py":
                if hasattr(self.interface, "command_sender"):
                    self.interface.command_sender.config = self.robot_config
                if hasattr(self.interface, "state_processor"):
                    self.interface.state_processor.config = self.robot_config
        else:
            # No values available - error
            raise ValueError(
                "No KP/KD values found. Either provide them in robot config "
                "or ensure ONNX model has metadata attached during training."
            )

        # Validate dimensions
        if len(kp_values) != self.robot_config.num_motors:
            raise ValueError(
                f"KP array length ({len(kp_values)}) does not match num_motors ({self.robot_config.num_motors})"
            )
        if len(kd_values) != self.robot_config.num_motors:
            raise ValueError(
                f"KD array length ({len(kd_values)}) does not match num_motors ({self.robot_config.num_motors})"
            )

    def _calculate_obs_dim_dict(self):
        """Calculate observation dimensions for each observation type."""
        obs_dim_dict = {}
        for key in self.obs_dict:
            obs_dim_dict[key] = 0
            for obs_name in self.obs_dict[key]:
                obs_dim_dict[key] += self.obs_dims[obs_name]
        return obs_dim_dict

    def _print_observations(self, obs: dict[str, np.ndarray]) -> None:
        """Print observation vector with term naming for debugging.

        Args:
            obs: Dictionary mapping observation group names to their flattened arrays.
        """
        np.set_printoptions(suppress=True, precision=3)
        print("\n========== Observation Vector ==========")
        for group_name, group_obs in obs.items():
            print(f"\n{group_name}:")
            if group_name in self.obs_dict:
                start_idx = 0
                for term_name in self.obs_terms_sorted.get(group_name, []):
                    term_dim = self.obs_dims[term_name]
                    history_len = self.history_length_dict.get(group_name, 1)
                    total_dim = term_dim * history_len
                    term_values = group_obs[0, start_idx : start_idx + total_dim]
                    print(f"  {term_name:20s} (dim={term_dim:2d}, hist={history_len}): {term_values}")
                    start_idx += total_dim
        print("========================================\n")

    def rl_inference(self, robot_state_data):
        """Perform RL inference to get policy action."""
        obs = self.prepare_obs_for_rl(robot_state_data)
        if self.config.task.print_observations:
            self._print_observations(obs)

        policy_action = self.policy(obs)
        policy_action = np.clip(policy_action, -100, 100)

        self.last_policy_action = policy_action.copy()
        self.scaled_policy_action = policy_action * self.policy_action_scale

        return self.scaled_policy_action

    # ============================================================================
    # Observation Processing Methods
    # ============================================================================

    def get_current_obs_buffer_dict(self, robot_state_data):
        """Extract current observation data from robot state."""
        current_obs_buffer_dict = {}

        # Extract base and joint data
        current_obs_buffer_dict["base_quat"] = robot_state_data[:, 3:7]
        current_obs_buffer_dict["base_ang_vel"] = robot_state_data[:, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6]
        current_obs_buffer_dict["dof_pos"] = robot_state_data[:, 7 : 7 + self.num_dofs] - self.default_dof_angles
        current_obs_buffer_dict["dof_vel"] = robot_state_data[
            :, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs
        ]

        # Use pre-computed corrected gravity if available from interface, else compute
        # This logic seems very brittle. TODO: Return a dataclass instead of just a numpy array.
        expected_len = (
            7 + self.num_dofs + 6 + self.num_dofs
        )  # base_pos(3) + quat(4) + dof_pos + lin_vel(3) + ang_vel(3) + dof_vel
        if robot_state_data.shape[1] == expected_len + 3:
            current_obs_buffer_dict["projected_gravity"] = robot_state_data[:, expected_len : expected_len + 3]
        else:
            v = np.array([[0, 0, -1]])
            current_obs_buffer_dict["projected_gravity"] = quat_rotate_inverse(current_obs_buffer_dict["base_quat"], v)

        return current_obs_buffer_dict

    def parse_current_obs_dict(self, current_obs_buffer_dict):
        """Parse observation buffer into observation dictionary with per-term scaling."""
        current_obs_dict: dict[str, dict[str, np.ndarray]] = {}
        for group, term_names in self.obs_terms_sorted.items():
            grouped_terms: dict[str, np.ndarray] = {}
            for term in term_names:
                if term not in current_obs_buffer_dict:
                    raise KeyError(f"Observation term '{term}' missing from current observation buffer.")
                term_obs = current_obs_buffer_dict[term]
                if term_obs.ndim == 1:
                    term_obs = term_obs.reshape(1, -1)
                scale = self.obs_scales[term]
                grouped_terms[term] = (term_obs * scale).astype(np.float32, copy=False)
            current_obs_dict[group] = grouped_terms
        return current_obs_dict

    def _prepare_group_observations(self, robot_state_data):
        """Return flattened observations per group with history applied per term."""
        current_obs_buffer_dict = self.get_current_obs_buffer_dict(robot_state_data)
        current_obs_dict = self.parse_current_obs_dict(current_obs_buffer_dict)
        return self._update_obs_history(current_obs_dict)

    def _update_obs_history(self, current_obs_dict: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        """Update observation history buffers and return flattened observations per group."""
        group_outputs: dict[str, np.ndarray] = {}

        for group, term_dict in current_obs_dict.items():
            history_len = self.history_length_dict.get(group, 1)
            flattened_terms: list[np.ndarray] = []

            for term in self.obs_terms_sorted[group]:
                obs = np.asarray(term_dict[term], dtype=np.float32, order="C")
                if obs.ndim == 1:
                    obs = obs.reshape(1, -1)

                buffer = self.obs_history_buffers[group][term]
                buffer.append(obs.copy())

                history = list(buffer)
                if len(history) < history_len:
                    missing = history_len - len(history)
                    if group in {"proprio_body", "proprio_hand"}:
                        pad_value = history[0] if history else obs
                        history = [pad_value.copy()] * missing + history
                    else:
                        history = [np.zeros_like(obs)] * missing + history

                # Match training order: time dimension first, then flatten into [history_len * term_dim].
                stacked = np.stack(history[-history_len:], axis=1)
                flattened_terms.append(stacked.reshape(obs.shape[0], -1))

            group_outputs[group] = (
                np.concatenate(flattened_terms, axis=1).astype(np.float32, copy=False)
                if flattened_terms
                else np.zeros((1, 0), dtype=np.float32)
            )

        self.obs_buf_dict = {group: value.copy() for group, value in group_outputs.items()}
        return group_outputs

    def prepare_obs_for_rl(self, robot_state_data):
        """Prepare observations for RL inference."""
        group_outputs = self._prepare_group_observations(robot_state_data)
        if "actor_obs" not in group_outputs:
            raise KeyError("Observation group 'actor_obs' is not configured for this policy.")
        return {"actor_obs": group_outputs["actor_obs"].astype(np.float32, copy=False)}

    # ============================================================================
    # Control/Command Methods
    # ============================================================================

    def get_init_target(self, robot_state_data):
        """Get initialization target joint positions."""
        dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
        if self.get_ready_state:
            # Interpolate from current dof_pos to default angles
            q_target = dof_pos + (self.default_dof_angles - dof_pos) * (self.init_count / 500)
            self.init_count += 1
            return q_target
        return dof_pos

    def _resolve_command_gains(self, kp_override, kd_override):
        """Resolve KP/KD overrides for command publishing."""
        if kp_override is not None or kd_override is not None:
            kp = kp_override if kp_override is not None else self._default_kp
            kd = kd_override if kd_override is not None else self._default_kd
            return kp, kd

        if getattr(self, "_stiff_startup_active", False):
            kp = getattr(self, "_stiff_hold_kp", None)
            kd = getattr(self, "_stiff_hold_kd", None)
            if kp is not None and kd is not None:
                kp_arr = np.asarray(kp, dtype=np.float32).reshape(-1)
                kd_arr = np.asarray(kd, dtype=np.float32).reshape(-1)
                if kp_arr.shape[0] == self.num_dofs and kd_arr.shape[0] == self.num_dofs:
                    return kp_arr, kd_arr
                if not getattr(self, "_stiff_gain_warned", False):
                    self.logger.warning(
                        "Stiff startup gains shape mismatch: kp=%s kd=%s expected=%s. Falling back to default gains.",
                        kp_arr.shape[0],
                        kd_arr.shape[0],
                        self.num_dofs,
                    )
                    self._stiff_gain_warned = True

        return self._default_kp, self._default_kd

    def _apply_fd_hand_vel(self, dof_pos: np.ndarray, dof_vel: np.ndarray, hand_indices: np.ndarray) -> np.ndarray:
        """Override hand joint velocities with finite-difference: (pos - prev_pos) * rl_rate."""
        if self._fd_hand_vel_prev_dof_pos is None:
            self._fd_hand_vel_prev_dof_pos = dof_pos.copy()
            dof_vel = dof_vel.copy()
            dof_vel[:, hand_indices] = 0.0
            return dof_vel
        fd_vel = (dof_pos[:, hand_indices] - self._fd_hand_vel_prev_dof_pos[:, hand_indices]) * self.rl_rate
        self._fd_hand_vel_prev_dof_pos = dof_pos.copy()
        dof_vel = dof_vel.copy()
        dof_vel[:, hand_indices] = fd_vel
        return dof_vel

    def _reset_debug_hand_cycle(self):
        """Reset the debug hand repeat cycle to re-seed from live hand positions."""
        self._debug_hand_step = 0
        self._debug_hand_demo_step = 0
        self._debug_hand_is_opening = True
        self._debug_hand_initialized = False
        self._debug_hand_action_phase = 0.0

    def _get_debug_hand_demo_groups(self):
        return _DEBUG_HAND_DEMO_FINGER_GROUPS

    def _select_debug_hand_demo_finger(self, finger_id: int):
        if not self._debug_hand_demo_enabled:
            return
        finger_groups = self._get_debug_hand_demo_groups()
        if finger_id not in finger_groups:
            return
        self._debug_hand_demo_selected_finger = finger_id
        finger_label, _ = finger_groups[finger_id]
        self.logger.info(f"debug_hand_demo finger: {finger_id} ({finger_label})")

    def _build_debug_hand_demo_target(self):
        """Hold the body at zero and animate one canonical finger group with the viewer demo cycle."""
        target_q = self._debug_hand_body_hold_q.copy()
        finger_groups = self._get_debug_hand_demo_groups()
        finger_label, finger_indices = finger_groups[self._debug_hand_demo_selected_finger]

        if self.q_min_arr is None or self.q_max_arr is None:
            self.logger.warning("debug_hand_demo requires joint position limits; falling back to zero target.")
            return target_q.reshape(1, -1)

        phase = 0.5 * (
            1.0 + math.sin((2.0 * math.pi * self._debug_hand_demo_step) / float(self._debug_hand_demo_cycle_steps))
        )
        target_q[finger_indices] = self.q_min_arr[finger_indices] + phase * (
            self.q_max_arr[finger_indices] - self.q_min_arr[finger_indices]
        )
        self._debug_hand_demo_step = (self._debug_hand_demo_step + 1) % self._debug_hand_demo_cycle_steps
        self._debug_hand_demo_active_label = finger_label
        return target_q.reshape(1, -1)

    def _load_debug_hand_action(self, action_path: str):
        """Load saved hand joint targets from an NPZ file produced by test_loco_mw.py."""
        resolved_path = Path(action_path).expanduser()
        if not resolved_path.is_absolute():
            resolved_path = (Path.cwd() / resolved_path).resolve()

        try:
            with np.load(resolved_path, allow_pickle=False) as payload:
                target = np.asarray(payload["target"], dtype=np.float32)
                source_hz = float(np.asarray(payload["policy_hz"], dtype=np.float32).reshape(-1)[0])
        except FileNotFoundError:
            logger.warning(f"debug_hand_action file not found: {resolved_path}")
            return None
        except KeyError as exc:
            logger.warning(f"debug_hand_action file is missing required key {exc}: {resolved_path}")
            return None
        except Exception as exc:
            logger.warning(f"Failed to load debug_hand_action from {resolved_path}: {exc}")
            return None

        num_hand_dofs = len(self._debug_hand_hand_idx)
        if target.ndim != 2 or target.shape[1] != num_hand_dofs:
            logger.warning(
                f"debug_hand_action target shape {target.shape} is invalid; "
                f"expected (N, {num_hand_dofs}): {resolved_path}"
            )
            return None
        if target.shape[0] == 0:
            logger.warning(f"debug_hand_action file has no samples: {resolved_path}")
            return None

        source_hz = max(source_hz, 1e-6)
        logger.info(f"Loaded debug_hand_action with {target.shape[0]} samples at {source_hz:.3f} Hz from {resolved_path}")
        return target, source_hz, resolved_path

    def _next_debug_hand_action_target(self):
        """Return the next replayed hand target, resampled to the current RL rate."""
        if self._debug_hand_action_sequence is None:
            raise RuntimeError("debug_hand_action sequence is not loaded")

        sequence = self._debug_hand_action_sequence
        if sequence.shape[0] == 1:
            return sequence[0].copy()

        phase = self._debug_hand_action_phase % sequence.shape[0]
        lower_idx = int(np.floor(phase))
        upper_idx = (lower_idx + 1) % sequence.shape[0]
        alpha = phase - lower_idx
        blended_target = (1.0 - alpha) * sequence[lower_idx] + alpha * sequence[upper_idx]

        rl_rate = max(float(self.rl_rate), 1e-6)
        phase_step = self._debug_hand_action_source_hz / rl_rate
        self._debug_hand_action_phase = (phase + phase_step) % sequence.shape[0]
        return blended_target.astype(np.float32, copy=False)

    def _build_debug_hand_target(self, robot_state_data):
        """Hold the body at zero position and drive the hands with a debug target source."""
        target_q = self._debug_hand_body_hold_q.copy()

        if self._debug_hand_action_enabled:
            target_q[self._debug_hand_hand_idx] = self._next_debug_hand_action_target()
            return target_q.reshape(1, -1)

        if self._debug_hand_demo_enabled:
            return self._build_debug_hand_demo_target()

        current_q = np.asarray(robot_state_data[0, 7 : 7 + self.num_dofs], dtype=np.float32).copy()

        if not self._debug_hand_initialized:
            self._debug_hand_start_pos = current_q[self._debug_hand_hand_idx].copy()
            self._debug_hand_current_pos = self._debug_hand_start_pos.copy()
            self._debug_hand_step = 0
            self._debug_hand_is_opening = True
            self._debug_hand_initialized = True

        target_pos = self._debug_hand_open_pos if self._debug_hand_is_opening else self._debug_hand_grip_pos
        if self._debug_hand_step < self._debug_hand_half_cycle_steps:
            alpha = self._debug_hand_step / float(self._debug_hand_half_cycle_steps)
            self._debug_hand_current_pos = self._debug_hand_start_pos + alpha * (target_pos - self._debug_hand_start_pos)
            self._debug_hand_step += 1
        else:
            self._debug_hand_step = 0
            self._debug_hand_start_pos = self._debug_hand_current_pos.copy()
            self._debug_hand_is_opening = not self._debug_hand_is_opening

        target_q[self._debug_hand_hand_idx] = self._debug_hand_current_pos
        return target_q.reshape(1, -1)

    def policy_action(self):
        """Execute policy action and send commands to robot."""

        kp_override = None
        kd_override = None

        # Stage 1: Read State
        with self.latency_tracker.measure("read_state"):
            if self.num_dofs == 43:
                robot_state_data = self.interface.get_full_state_43dof()
            else:
                robot_state_data = self.interface.get_low_state()

        self._policy_step_hook(robot_state_data)

        # Safety Check: Monitor for unsafe conditions after policy starts
        if self.safety_enabled and self.use_policy_action:
            base_quat = robot_state_data[:, 3:7]
            v = np.array([[0, 0, -1]])
            projected_gravity = quat_rotate_inverse(base_quat, v)

            if self.is_unsafe(robot_state_data, projected_gravity):
                self.activate_damping_mode()
                return  # Skip rest of control loop

        # Stage 2: Pre-processing
        with self.latency_tracker.measure("preprocessing"):
            # Determine target joint positions
            if self.get_ready_state:
                q_target = self.get_init_target(robot_state_data)
                self.init_count = min(self.init_count, 500)
                # Use motor_kp/kd when moving to the default pose
                if self.robot_config.motor_kp is not None:
                    kp_override = np.asarray(self.robot_config.motor_kp, dtype=np.float32)
                if self.robot_config.motor_kd is not None:
                    kd_override = np.asarray(self.robot_config.motor_kd, dtype=np.float32)
            elif not self.use_policy_action:
                manual_cmd = self._get_manual_command(robot_state_data)
                if manual_cmd is not None:
                    q_target = manual_cmd["q"]
                    kp_override = manual_cmd.get("kp")
                    kd_override = manual_cmd.get("kd")
                else:
                    q_target = robot_state_data[:, 7 : 7 + self.num_dofs]
            else:
                # Prepare for inference - any preprocessing before RL inference
                q_target = None

        # Stage 3: Inference
        if self.use_policy_action and not self.get_ready_state:
            with self.latency_tracker.measure("inference"):
                scaled_policy_action = self.rl_inference(robot_state_data)

        # Stage 4: Post-processing
        with self.latency_tracker.measure("postprocessing"):
            if self.use_policy_action and not self.get_ready_state:
                if scaled_policy_action.shape[1] != self.num_dofs:
                    if not self.upper_body_controller:
                        scaled_policy_action = np.concatenate(
                            [np.zeros((1, self.num_dofs - scaled_policy_action.shape[1])), scaled_policy_action], axis=1
                        )
                    else:
                        raise NotImplementedError("Upper body controller not implemented")
                # Most policies output residual joint offsets around default pose.
                # Tracking policies can opt into absolute joint commands.
                if self.config.task.use_absolute_action:
                    q_target = scaled_policy_action
                else:
                    q_target = scaled_policy_action + self.default_dof_angles
                
                if self._debug_hand_enabled:
                    q_target = self._build_debug_hand_target(robot_state_data)
                    if self._debug_hand_kp is not None:
                        kp_override = self._debug_hand_kp
                    if self._debug_hand_kd is not None:
                        kd_override = self._debug_hand_kd

                # Prepare command (reuse pre-allocated arrays)
                self.cmd_q[:] = q_target[0]
                # Clip per-step delta to prevent sudden jumps
                if self._prev_cmd_q is not None:
                    delta = self.cmd_q - self._prev_cmd_q
                    np.clip(delta, -self._cmd_q_max_delta, self._cmd_q_max_delta, out=delta)
                    self.cmd_q[:] = self._prev_cmd_q + delta
                self._prev_cmd_q = self.cmd_q.copy()
            else:
                # Prepare command (reuse pre-allocated arrays)
                self.cmd_q[:] = q_target[0]

        # Stage 5: Action Pub
        with self.latency_tracker.measure("action_pub"):
            # Expand 29-DOF command to 43-DOF if needed (add hand defaults)
            if self.num_dofs == 29 and getattr(self.interface, "use_hands", False):
                # Robot supports 43-DOF but policy only controls 29-DOF
                # Insert hand defaults at correct positions in config order
                cmd_q_43 = np.zeros(43, dtype=np.float32)
                # Body joints (29): left_leg(6) + right_leg(6) + waist(3) + left_arm(4) + left_wrist(3) + right_arm(4) + right_wrist(3)
                cmd_q_43[:22] = self.cmd_q[:22]  # left_leg + right_leg + waist + left_arm + left_wrist
                cmd_q_43[22:29] = self.hand_default_pos[:7]  # left hand (7 joints)
                cmd_q_43[29:36] = self.cmd_q[22:29]  # right_arm + right_wrist
                cmd_q_43[36:43] = self.hand_default_pos[7:]  # right hand (7 joints)

                # Get current state for 43-DOF
                robot_state_43 = self.interface.get_full_state_43dof()

                # Send 43-DOF command
                kp_override = np.concatenate([
                    self._default_kp[:22],  # body
                    np.full(7, 5.0),  # left hand - lower gains for compliance
                    self._default_kp[22:29] if len(self._default_kp) >= 29 else np.full(7, 14.25),  # right arm
                    np.full(7, 5.0),  # right hand - lower gains for compliance
                ])
                kd_override = np.concatenate([
                    self._default_kd[:22],  # body
                    np.full(7, 0.5),  # left hand
                    self._default_kd[22:29] if len(self._default_kd) >= 29 else np.full(7, 0.91),  # right arm
                    np.full(7, 0.5),  # right hand
                ])

                self.interface.send_full_command_43dof(
                    cmd_q_43,
                    np.zeros(43),  # cmd_dq
                    np.zeros(43),  # cmd_tau
                    robot_state_43[0, 7:50],  # current q (43 DOF)
                    kp_override_43=kp_override,
                    kd_override_43=kd_override,
                )
            elif self.num_dofs == 43:
                kp_override, kd_override = self._resolve_command_gains(kp_override, kd_override)
                self.interface.send_full_command_43dof(
                    self.cmd_q,
                    self.cmd_dq,
                    self.cmd_tau,
                    robot_state_data[0, 7 : 7 + self.num_dofs],
                    kp_override_43=kp_override,
                    kd_override_43=kd_override,
                )
            else:
                self.interface.send_low_command(
                    self.cmd_q,
                    self.cmd_dq,
                    self.cmd_tau,
                    robot_state_data[0, 7 : 7 + self.num_dofs],
                    kp_override=kp_override,
                    kd_override=kd_override,
                )

    def _get_manual_command(self, robot_state_data):
        """Optional manual command when policy control is disabled."""
        return

    def _get_obs_phase_time(self):
        """Calculate phase time for gait."""
        cur_time = time.perf_counter() * self.stand_command[0, 0]
        phase_time = cur_time % self.gait_period / self.gait_period
        self.phase_time[:, 0] = phase_time
        return self.phase_time

    def update_phase_time(self):
        """Update phase time."""
        phase_tp1 = self.phase + self.phase_dt
        self.phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi

    # ============================================================================
    # Input Handler Methods
    # ============================================================================

    def start_key_listener(self):
        """Start keyboard listener thread."""

        def on_press(keycode):
            try:
                self.handle_keyboard_button(keycode)
            except AttributeError:
                pass  # Handle special keys if needed

        try:
            listener = listen_keyboard(on_press=on_press)
            listener.start()
            listener.join()
        except OSError as e:
            # Handle termios errors in non-TTY environments
            self.logger.warning("Could not start keyboard listener: %s", e)
            self.logger.warning("Keyboard input will not be available")

    def process_joystick_input(self):
        """Process joystick input and update commands using interface."""
        # Store previous key states for edge detection
        self.last_key_states = self.key_states.copy() if hasattr(self, "key_states") else {}

        # Process joystick input - returns (lin_vel, ang_vel, key_states)
        self.lin_vel_command, self.ang_vel_command, self.key_states = self.interface.process_joystick_input(
            self.lin_vel_command, self.ang_vel_command, self.stand_command, False
        )

        # Handle button presses (edge detection: only trigger on press, not hold)
        for key, is_pressed in self.key_states.items():
            if is_pressed and not self.last_key_states.get(key, False):
                self.handle_joystick_button(key)
                self._print_control_status()

    # ============================================================================
    # Button Handler Methods
    # ============================================================================

    def handle_keyboard_button(self, keycode):
        """Handle keyboard button presses."""
        if self._try_switch_policy_key(keycode):
            pass
        elif self._debug_hand_demo_enabled and keycode in {"1", "2", "3", "4", "5", "6"}:
            self._select_debug_hand_demo_finger(int(keycode))
        elif keycode == "]":
            self._handle_start_policy()
        elif keycode == "o":
            self._handle_stop_policy()
        elif keycode == "i":
            self._handle_init_state()
        elif keycode in ["v", "b", "f", "g", "r"]:
            self._handle_kp_control(keycode)

        self._print_control_status()

    def handle_joystick_button(self, cur_key):
        """Handle joystick button presses."""
        if cur_key == "A":
            self._handle_start_policy()
        elif cur_key == "B":
            self._handle_stop_policy()
        elif cur_key == "Y":
            self._handle_init_state()
        elif cur_key in ["up", "down", "left", "right", "F1"]:
            # TODO: Make this more intuitive
            self._handle_joystick_kp_control(cur_key)
        elif cur_key == "select":
            # Cycle to next policy
            next_index = (self.active_policy_index + 1) % len(self.model_paths)
            self._activate_policy(next_index)
        elif cur_key == "L1+R1":
            # Kill program, works on G1 joystick only.
            self.logger.info(colored("Killing program via joystick command", "red"))
            sys.exit(0)

    # ============================================================================
    # Control Action Methods
    # ============================================================================

    def _handle_start_policy(self):
        """Handle start policy action."""
        self.use_policy_action = True
        self.get_ready_state = False
        self.safety_enabled = True  # Enable safety checks when policy starts
        if self._debug_hand_enabled:
            self._reset_debug_hand_cycle()
            if self._debug_hand_action_enabled and self._debug_hand_action_path is not None:
                self.logger.info(
                    f"debug_hand_action enabled: body joints will hold zero position while hands replay "
                    f"{self._debug_hand_action_path}."
                )
            elif self._debug_hand_demo_enabled:
                finger_groups = self._get_debug_hand_demo_groups()
                finger_label, _ = finger_groups[self._debug_hand_demo_selected_finger]
                self.logger.info(
                    "debug_hand_demo enabled: body joints will hold zero position while finger groups "
                    f"1-6 drive the hands. Current finger: {self._debug_hand_demo_selected_finger} ({finger_label})."
                )
            else:
                self.logger.info("debug_hand enabled: body joints will hold zero position while hands repeat open/grip.")
        self.logger.info(colored("Using policy actions - Safety monitoring ENABLED", "blue"))
        self.phase = np.array([[0.0, np.pi]])
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0

    def _handle_stop_policy(self):
        """Handle stop policy action."""
        self.use_policy_action = False
        self.get_ready_state = False
        self.safety_enabled = False  # Disable safety checks when policy is stopped
        self.damping_mode_active = False  # Reset damping mode
        if self._debug_hand_enabled:
            self._reset_debug_hand_cycle()
        self.logger.info("Actions set to zero")
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 1

    def _handle_init_state(self):
        """Handle initialization state."""
        self.get_ready_state = True
        self.init_count = 0
        self.logger.info("Setting to init state")
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0

    def _handle_kp_control(self, keycode):
        """Handle keyboard KP control."""
        if keycode == "v":
            self.interface.kp_level -= 0.01
        elif keycode == "b":
            self.interface.kp_level += 0.01
        elif keycode == "f":
            self.interface.kp_level -= 0.1
        elif keycode == "g":
            self.interface.kp_level += 0.1
        elif keycode == "r":
            self.interface.kp_level = 1.0

    def _handle_joystick_kp_control(self, keycode):
        """Handle joystick KP control."""
        if keycode == "down":
            self.interface.kp_level -= 0.1
        elif keycode == "up":
            self.interface.kp_level += 0.1
        elif keycode == "left":
            self.interface.kp_level -= 0.01
        elif keycode == "right":
            self.interface.kp_level += 0.01
        elif keycode == "F1":
            self.interface.kp_level = 1.0

    def _print_control_status(self):
        """Print current control status."""
        self.logger.info("------------ Control Status ------------")
        if self.active_model_path:
            total = len(self.model_paths)
            name = Path(self.active_model_path).name
            debug_str = (
                f"Active policy [{self.active_policy_index + 1}/{total}]: {name} Kp level {self.interface.kp_level:.2f}"
            )
            if self._debug_hand_demo_enabled:
                finger_groups = self._get_debug_hand_demo_groups()
                finger_label, _ = finger_groups[self._debug_hand_demo_selected_finger]
                debug_str += f" debug_hand_demo={self._debug_hand_demo_selected_finger}:{finger_label}"
            self.logger.info(debug_str)

    # ============================================================================
    # Main Run Method
    # ============================================================================

    def run(self):
        """Main run loop for the policy."""
        try:
            for it in itertools.count():
                self.latency_tracker.start_cycle()

                if self.use_joystick and self.interface.get_joystick_msg() is not None:
                    self.process_joystick_input()
                if self.use_phase:
                    self.update_phase_time()

                self.policy_action()

                self.latency_tracker.end_cycle()

                if it % 50 == 0 and self.use_policy_action:
                    debug_str = f"RL FPS: {self.latency_tracker.get_fps():.2f} | {self.latency_tracker.get_stats_str()}"
                    self.logger.info(debug_str, flush=True)

                self.rate.sleep()

        except KeyboardInterrupt:
            pass

    def shutdown(self):
        """Optional cleanup hook for derived policies."""
        return

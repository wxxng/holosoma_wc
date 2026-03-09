from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import onnx
import onnxruntime
from loguru import logger
from scipy.spatial.transform import Rotation
from termcolor import colored

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.policies import BasePolicy
from holosoma_inference.utils.clock import ClockSub
from holosoma_inference.utils.math.quat import (
    quat_mul,
    quat_to_rpy,
    rpy_to_quat,
)

# Joint ordering mappings for G1 robot
# Config order (hardware/simulator): grouped by limb [L_leg, R_leg, waist, L_arm, L_hand, R_arm, R_hand]
# Asset order (model training): interleaved [LHP, RHP, WY, LHR, RHR, WR, LHY, RHY, WP, ...]
# G1_JOINT_NAMES order (motion clips): grouped by limb (29-DOF body, optional 43-DOF with hands)

# 43-DOF: Config -> Asset (for observations)
CONFIG_TO_ASSET_ORDER_43DOF = (
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 29, 4, 10, 16, 30, 5, 11, 17, 31, 18, 32,
    19, 33, 20, 34, 21, 35, 27, 25, 22, 41, 39, 36, 28, 26, 23, 42, 40, 37, 24, 38
)

# 43-DOF: Asset -> Config (for actions)
ASSET_TO_CONFIG_ORDER_43DOF = (
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 31,
    37, 41, 30, 36, 29, 35, 12, 16, 20, 22, 24, 26, 28, 34, 40, 42, 33, 39, 32, 38
)

# 29-DOF: Config -> Asset (for observations)
CONFIG_TO_ASSET_ORDER_29DOF = (
    12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 26, 27, 28
)

# 29-DOF: Asset -> Config (for actions)
ASSET_TO_CONFIG_ORDER_29DOF = (
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 26, 27, 28
)

ACTION_CONFIG_TO_ASSET_ORDER_29DOF = (
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18,
    25, 19, 26, 20, 27, 21, 28
)

ACTION_ASSET_TO_CONFIG_ORDER_29DOF = (
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27,
    12, 16, 20, 22, 24, 26, 28
)

# G1_JOINT_NAMES (29-DOF motion) -> Asset order (for motion commands)
G1_TO_ASSET_ORDER_29DOF = (
    12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 26, 27, 28
)

G1_MOTION_JOINT_NAMES_29 = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

G1_MOTION_JOINT_NAMES_43 = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
]

DEFAULT_MOTION_REL_PATH = Path("src/holosoma/holosoma/data/motions/motion_tracking/grab_omomo_selected_111_filtered.pkl")
DEFAULT_MOTION_CLIP_KEY = "GRAB_s10_cubemedium_pass_1"
GRAVITY_WORLD = np.array([0.0, 0.0, -1.0], dtype=np.float32)
LEFT_HAND_DEFAULTS = np.array([0.0, 1.0, 1.7, -1.57, -1.7, -1.57, -1.7], dtype=np.float32)
RIGHT_HAND_DEFAULTS = np.array([0.0, -1.0, -1.7, 1.57, 1.7, 1.57, 1.7], dtype=np.float32)
ASSET_TO_CONFIG_ORDER_29DOF_ARR = np.array(ASSET_TO_CONFIG_ORDER_29DOF, dtype=np.int64)
G1_TO_ASSET_ORDER_29DOF_ARR = np.array(G1_TO_ASSET_ORDER_29DOF, dtype=np.int64)


class MotionTrackingPolicy(BasePolicy):
    def __init__(self, config: InferenceConfig):
        self.motion_timestep = 0
        self.motion_clip_progressing = False
        self.motion_start_timestep = None
        self.stabilization_mode = False

        # Motion library
        self.motion_data = None
        self.motion_dof_pos = None  # [num_frames, num_dofs] in motion clip order
        self.motion_dof_vel = None  # [num_frames, num_dofs] calculated from pos
        self.motion_fps = None
        self.motion_length = 0
        self.motion_clip_key = None
        self.motion_to_asset_joint_map = None

        # Interpolation counter for smooth transition to first frame
        self.rl_interp_count = 0
        self.rl_interp_steps = 100

        # Global timestep counter
        self.global_timestep = 0

        self.timestep_interval_ms = 1000.0 / config.task.rl_rate

        self.clock_sub = ClockSub()
        self.clock_sub.start()
        self._last_clock_reading: int | None = None

        self.use_sim_time = config.task.use_sim_time
        self.motion_to_asset_joint_map = G1_TO_ASSET_ORDER_29DOF_ARR

        self._stiff_hold_active = True
        self.robot_yaw_offset = 0.0

        # Stiff hold interpolation
        self._stiff_hold_interp_count = 0
        self._stiff_hold_interp_steps = 50  # 1 second at 50Hz
        self._stiff_hold_start_q = None

        super().__init__(config)

        if config.robot.stiff_startup_pos is not None:
            self._stiff_hold_q = np.array(config.robot.stiff_startup_pos, dtype=np.float32).reshape(1, -1)
        else:
            self._stiff_hold_q = np.array(config.robot.default_dof_angles, dtype=np.float32).reshape(1, -1)

        if config.robot.stiff_startup_kp is not None:
            self._stiff_hold_kp = np.array(config.robot.stiff_startup_kp, dtype=np.float32)
        else:
            raise ValueError("Robot config must specify stiff_startup_kp for WBT policy")

        if config.robot.stiff_startup_kd is not None:
            self._stiff_hold_kd = np.array(config.robot.stiff_startup_kd, dtype=np.float32)
        else:
            raise ValueError("Robot config must specify stiff_startup_kd for WBT policy")

        if self._stiff_hold_q.shape[1] != self.num_dofs:
            raise ValueError("Stiff startup pose dimension mismatch with robot DOFs")

        if self.num_dofs == 43:
            self._config_to_asset_order = np.array(CONFIG_TO_ASSET_ORDER_43DOF, dtype=np.int64)
            self._action_asset_to_config_order = np.array(ASSET_TO_CONFIG_ORDER_43DOF, dtype=np.int64)
        else:
            self._config_to_asset_order = np.array(CONFIG_TO_ASSET_ORDER_29DOF, dtype=np.int64)
            self._action_asset_to_config_order = np.array(ACTION_ASSET_TO_CONFIG_ORDER_29DOF, dtype=np.int64)

        self._action_params_cache_29: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self._action_params_cache_43: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

        if config.robot.motor_kp is not None:
            self._default_kp = np.array(config.robot.motor_kp, dtype=np.float32)
        else:
            raise ValueError("Robot config must specify motor_kp for WBT policy")

        if config.robot.motor_kd is not None:
            self._default_kd = np.array(config.robot.motor_kd, dtype=np.float32)
        else:
            raise ValueError("Robot config must specify motor_kd for WBT policy")

        def _show_warning():
            logger.warning(
                colored(
                    "⚠️  Non-interactive mode detected - cannot prompt for stiff mode confirmation!",
                    "red",
                    attrs=["bold"],
                )
            )

        if sys.stdin.isatty():
            logger.info(colored("\n⚠️  Ready to enter stiff hold mode", "yellow", attrs=["bold"]))
            logger.info(colored("Press Enter to continue...", "yellow"))
            try:
                input()
                logger.info(colored("✓ Entering stiff hold mode", "green"))
            except EOFError:
                _show_warning()
        else:
            _show_warning()

    def _initialize_history_state(self):
        """Override to use listed (not sorted) obs term order to match training."""
        super()._initialize_history_state()
        for group, term_names in self.obs_dict.items():
            self.obs_terms_sorted[group] = list(term_names)
        # Rebuild obs_buf_dict with corrected order
        for group, term_names in self.obs_dict.items():
            history_len = self.history_length_dict.get(group, 1)
            self.obs_history_buffers[group] = {}
            flattened_terms = []
            for term in self.obs_terms_sorted[group]:
                term_dim = self.obs_dims[term]
                self.obs_history_buffers[group][term] = deque(maxlen=history_len)
                flattened_terms.append(np.zeros((1, term_dim * history_len), dtype=np.float32))
            self.obs_buf_dict[group] = np.concatenate(flattened_terms, axis=1) if flattened_terms else np.zeros((1, 0))

    def _load_motion_from_pkl(self, pkl_path: str, clip_key: str | None = None):
        """Load motion data from PKL file and prepare for inference."""
        logger.info(f"Loading motion from {pkl_path}, clip key: {clip_key}")

        self.motion_data = joblib.load(pkl_path)
        motion_keys = list(self.motion_data.keys())

        if clip_key is None:
            clip_key = motion_keys[0]
            logger.info(f"No clip key specified, using first clip: {clip_key}")

        if clip_key not in self.motion_data:
            clips_info = []
            for key in motion_keys:
                clip_data = self.motion_data[key]
                fps = clip_data["fps"]
                num_frames = clip_data["dof_pos"].shape[0]
                duration = num_frames / fps
                clips_info.append(f"  - '{key}': {num_frames} frames ({duration:.2f}s @ {fps}Hz)")
            raise ValueError(f"Clip key '{clip_key}' not found. Available clips:\n" + "\n".join(clips_info))

        clip_data = self.motion_data[clip_key]

        logger.info(f"Using clip: {clip_key}")
        logger.info(f"  FPS: {clip_data['fps']}")
        logger.info(f"  dof_pos shape: {clip_data['dof_pos'].shape}")

        self.motion_fps = clip_data["fps"]
        raw_dof_pos = clip_data["dof_pos"]
        raw_dof_dim = raw_dof_pos.shape[1]
        if raw_dof_dim == 43:
            body_indices = [G1_MOTION_JOINT_NAMES_43.index(name) for name in G1_MOTION_JOINT_NAMES_29]
            self.motion_dof_pos = raw_dof_pos[:, body_indices]
            logger.info("  Stripped hand joints from 43-DOF clip -> 29-DOF motion commands")
        elif raw_dof_dim == 29:
            self.motion_dof_pos = raw_dof_pos
        else:
            raise ValueError(f"Unexpected motion dof dimension: {raw_dof_dim} (expected 29 or 43)")

        self.motion_length = self.motion_dof_pos.shape[0]

        dof_pos_next = np.roll(self.motion_dof_pos, -1, axis=0)
        dof_pos_next[-1] = self.motion_dof_pos[-1]
        self.motion_dof_vel = (dof_pos_next - self.motion_dof_pos) * self.motion_fps
        self.motion_dof_vel[-1] = 0.0

        logger.info(
            f"Motion loaded: {self.motion_length} frames @ {self.motion_fps} Hz"
            f" = {self.motion_length / self.motion_fps:.2f}s"
        )

        self.motion_clip_key = clip_key

    def _get_motion_command_sequence(self, timestep: int) -> np.ndarray:
        """Get 10-frame motion sequence starting from timestep.

        Returns:
            np.ndarray: [1, 580] - 10 frames × 58 dims (29-DOF, G1_JOINT_NAMES order)
        """
        future_steps = np.arange(timestep, timestep + 10)
        future_steps = np.clip(future_steps, 0, self.motion_length - 1)

        pos_seq = self.motion_dof_pos[future_steps]  # [10, 29]
        vel_seq = self.motion_dof_vel[future_steps]  # [10, 29]

        command_seq = np.concatenate([pos_seq, vel_seq], axis=1)  # [10, 58]
        return command_seq.reshape(1, -1)  # [1, 580]

    def setup_policy(self, model_path):
        self.onnx_policy_session = onnxruntime.InferenceSession(model_path)
        self.onnx_input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        self.onnx_output_names = [out.name for out in self.onnx_policy_session.get_outputs()]

        onnx_model = onnx.load(model_path)
        metadata = {}
        for prop in onnx_model.metadata_props:
            metadata[prop.key] = json.loads(prop.value)

        self.onnx_kp = np.array(metadata["kp"]) if "kp" in metadata else None
        self.onnx_kd = np.array(metadata["kd"]) if "kd" in metadata else None

        if self.onnx_kp is not None:
            logger.info(f"Loaded KP/KD from ONNX metadata: {Path(model_path).name}")

        pkl_path, clip_key = self._resolve_motion_source()

        if pkl_path.exists():
            self._load_motion_from_pkl(str(pkl_path), clip_key=clip_key)
        else:
            logger.warning(f"Motion PKL file not found at {pkl_path}. Motion commands will be zero.")

        def policy_act(input_feed):
            output = self.onnx_policy_session.run(["actions"], input_feed)
            return output[0]

        self.policy = policy_act

    def _resolve_motion_source(self) -> tuple[Path, str]:
        """Resolve PKL path and clip key for tracking motion."""
        holosoma_root = Path(__file__).parent.parent.parent.parent.parent
        pkl_path = (
            Path(self.config.task.motion_pkl_path)
            if self.config.task.motion_pkl_path is not None
            else holosoma_root / DEFAULT_MOTION_REL_PATH
        )
        clip_key = self.config.task.motion_clip_key or DEFAULT_MOTION_CLIP_KEY
        return pkl_path, clip_key

    def _capture_policy_state(self):
        state = super()._capture_policy_state()
        state.update(
            {
                "motion_timestep": self.motion_timestep,
                "motion_clip_key": self.motion_clip_key,
            }
        )
        return state

    def _restore_policy_state(self, state):
        super()._restore_policy_state(state)
        self.motion_timestep = state.get("motion_timestep", 0)
        self.motion_clip_key = state.get("motion_clip_key", None)
        self.motion_clip_progressing = False
        self.stabilization_mode = False
        self.motion_start_timestep = None
        self._last_clock_reading = None
        self.robot_yaw_offset = 0.0

    def _on_policy_switched(self, model_path: str):
        super()._on_policy_switched(model_path)
        self.motion_clip_progressing = False
        self.stabilization_mode = False
        self.motion_timestep = 0
        self.motion_start_timestep = None
        self._last_clock_reading = None
        self._stiff_hold_active = True
        self.robot_yaw_offset = 0.0

    def get_init_target(self, robot_state_data):
        """Get initialization target joint positions."""
        dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
        if self.get_ready_state:
            if self.motion_data is not None:
                target_dof_pos = self._motion_to_config_order(self.motion_dof_pos[0]).reshape(1, -1)
            else:
                target_dof_pos = dof_pos
            q_target = dof_pos + (target_dof_pos - dof_pos) * (self.init_count / 500)
            self.init_count += 1
            return q_target
        return dof_pos

    def get_current_obs_buffer_dict(self, robot_state_data):
        current_obs_buffer_dict = {}

        dof_pos_config = robot_state_data[:, 7 : 7 + self.num_dofs]
        dof_vel_config = robot_state_data[:, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs]

        dof_pos_asset = dof_pos_config[:, self._config_to_asset_order]
        dof_vel_asset = dof_vel_config[:, self._config_to_asset_order]
        default_dof_angles_asset = self.default_dof_angles[self._config_to_asset_order]

        # obs in listed order: dof_pos, dof_vel, base_ang_vel, projected_gravity, actions, motion_command_sequence
        current_obs_buffer_dict["dof_pos"] = dof_pos_asset - default_dof_angles_asset
        current_obs_buffer_dict["dof_vel"] = dof_vel_asset
        current_obs_buffer_dict["base_ang_vel"] = robot_state_data[:, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6]

        base_quat_wxyz = robot_state_data[:, 3:7]
        w, x, y, z = base_quat_wxyz[0]
        rot = Rotation.from_quat([x, y, z, w])
        current_obs_buffer_dict["projected_gravity"] = rot.inv().apply(GRAVITY_WORLD).reshape(1, -1)

        current_obs_buffer_dict["actions"] = self.last_policy_action

        if self.motion_data is not None:
            command_timestep = 0 if self.stabilization_mode else self.motion_timestep
            current_obs_buffer_dict["motion_command_sequence"] = self._get_motion_command_sequence(command_timestep)
        else:
            current_obs_buffer_dict["motion_command_sequence"] = np.zeros((1, 580), dtype=np.float32)

        return current_obs_buffer_dict

    def rl_inference(self, robot_state_data):
        if not self.motion_clip_progressing and not self.stabilization_mode:
            self.motion_timestep = 0
            self.motion_start_timestep = None
            self._last_clock_reading = None

            # Smoothly interpolate to first motion frame before clip starts
            if self.motion_data is not None:
                if not getattr(self, "_interp_started_logged", False):
                    logger.info(colored("▶ Interpolating to first motion frame (motion_data loaded)", "green", attrs=["bold"]))
                    self._interp_started_logged = True
                dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
                target_pos_config = self._motion_to_config_order(self.motion_dof_pos[0]).reshape(1, -1)
                interp_ratio = min(self.rl_interp_count / self.rl_interp_steps, 1.0)
                interp_abs = dof_pos + (target_pos_config - dof_pos) * interp_ratio
                # Return absolute joint targets during interpolation.
                self.scaled_policy_action = interp_abs
                self.rl_interp_count += 1
                self.last_policy_action = np.zeros((1, self.num_dofs), dtype=np.float32)
                self.global_timestep += 1
                return self.scaled_policy_action
            else:
                if not getattr(self, "_inference_no_motion_logged", False):
                    logger.warning(colored(
                        "⚠ motion_data is None — skipping interpolation, running ONNX inference directly! "
                        "Check that the PKL file exists.",
                        "red", attrs=["bold"]
                    ))
                    self._inference_no_motion_logged = True

        if not getattr(self, "_onnx_inference_started_logged", False):
            logger.info(colored(
                f"▶ ONNX inference started "
                f"(motion_clip_progressing={self.motion_clip_progressing}, "
                f"stabilization_mode={self.stabilization_mode}, "
                f"motion_data={'loaded' if self.motion_data is not None else 'None'})",
                "cyan", attrs=["bold"]
            ))
            self._onnx_inference_started_logged = True

        obs = self.prepare_obs_for_rl(robot_state_data)
        if self.config.task.print_observations:
            self._print_observations(obs)

        input_feed = {"obs": obs["actor_obs"]}
        raw_action_asset = self.policy(input_feed)

        action_scale, action_offset, action_clip_min, action_clip_max = self._get_action_params_asset()
        policy_action_asset = np.clip(
            raw_action_asset * action_scale + action_offset,
            action_clip_min,
            action_clip_max,
        )

        self.scaled_policy_action = policy_action_asset[:, self._action_asset_to_config_order]

        self.last_policy_action = raw_action_asset
        self.global_timestep += 1

        if self.motion_clip_progressing and not self.stabilization_mode:
            if self.use_sim_time:
                self._update_clock()
            else:
                self.motion_timestep += 1

        return self.scaled_policy_action

    def _get_action_params_asset(self):
        """Return (scale, offset, clip_min, clip_max) in asset order."""
        cache = self._action_params_cache_43
        if cache is None:
            action_scale = np.array([
                0.5475, 0.5475, 0.5475, 0.3507, 0.3507, 0.4386, 0.5475, 0.5475, 0.4386,
                0.3507, 0.3507, 0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.4386,
                0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.0745, 0.0745,
                0.0745, 0.0745, 0.7000, 0.7000, 0.3063, 0.7000, 0.7000, 0.3063, 0.7000,
                0.7000, 0.7000, 0.7000, 0.7000, 0.7000, 0.7000, 0.7000,
            ], dtype=np.float32)

            action_offset = np.array([
                -0.3120, -0.3120, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.6690, 0.6690, 0.2000, 0.2000, -0.3630, -0.3630, 0.2000,
                -0.2000, 0.0000, 0.0000, 0.0000, 0.0000, 0.6000, 0.6000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -1.4000, -1.4000, 0.0000,
                1.4000, 1.4000, 0.0000, -1.5700, -1.5700, 0.3500, 1.5700, 1.5700,
                0.3500, 1.5700, -1.5700,
            ], dtype=np.float32)

            action_clip_min = np.array([
                -10.9509, -10.9509, -10.9509, -7.0132, -7.0132, -8.7715, -10.9509, -10.9509, -8.7715,
                -7.0132, -7.0132, -8.7715, -8.7715, -8.7715, -8.7715, -8.7715, -8.7715, -8.7715,
                -8.7715, -8.7715, -8.7715, -8.7715, -8.7715, -8.7715, -8.7715, -1.4900, -1.4900,
                -1.4900, -1.4900, -14.0000, -14.0000, -6.1250, -14.0000, -14.0000, -6.1250, -14.0000,
                -14.0000, -14.0000, -14.0000, -14.0000, -14.0000, -14.0000, -14.0000,
            ], dtype=np.float32)

            action_clip_max = np.array([
                10.9509, 10.9509, 10.9509, 7.0132, 7.0132, 8.7715, 10.9509, 10.9509, 8.7715,
                7.0132, 7.0132, 8.7715, 8.7715, 8.7715, 8.7715, 8.7715, 8.7715, 8.7715,
                8.7715, 8.7715, 8.7715, 8.7715, 8.7715, 8.7715, 8.7715, 1.4900, 1.4900,
                1.4900, 1.4900, 14.0000, 14.0000, 6.1250, 14.0000, 14.0000, 6.1250, 14.0000,
                14.0000, 14.0000, 14.0000, 14.0000, 14.0000, 14.0000, 14.0000,
            ], dtype=np.float32)
            cache = (action_scale, action_offset, action_clip_min, action_clip_max)
            self._action_params_cache_43 = cache

        if self.num_dofs == 29:
            if self._action_params_cache_29 is None:
                self._action_params_cache_29 = tuple(arr[:29] for arr in cache)
            return self._action_params_cache_29

        return cache

    def _motion_to_config_order(self, pos_motion_order: np.ndarray) -> np.ndarray:
        """Convert joint positions from G1_JOINT_NAMES order to config order."""
        pos_asset = pos_motion_order[self.motion_to_asset_joint_map]
        if self.num_dofs == 43:
            pos_config_29 = pos_asset[ASSET_TO_CONFIG_ORDER_29DOF_ARR]
            return np.concatenate([pos_config_29[:22], LEFT_HAND_DEFAULTS, pos_config_29[22:], RIGHT_HAND_DEFAULTS])
        return pos_asset[ASSET_TO_CONFIG_ORDER_29DOF_ARR]

    def _get_manual_command(self, robot_state_data):
        if not self._stiff_hold_active:
            return None

        dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]

        if self._stiff_hold_start_q is None:
            self._stiff_hold_start_q = dof_pos.copy()
            self._stiff_hold_interp_count = 0

        if self._stiff_hold_interp_count < self._stiff_hold_interp_steps:
            alpha = (self._stiff_hold_interp_count + 1) / self._stiff_hold_interp_steps
            q_target = self._stiff_hold_start_q + alpha * (self._stiff_hold_q - self._stiff_hold_start_q)
            self._stiff_hold_interp_count += 1
        else:
            q_target = self._stiff_hold_q.copy()

        return {
            "q": q_target,
            "kp": self._stiff_hold_kp,
            "kd": self._stiff_hold_kd,
        }

    def _handle_start_policy(self):
        super()._handle_start_policy()
        self._stiff_hold_active = False
        self.motion_clip_progressing = False
        self.rl_interp_count = 0
        self._capture_robot_yaw_offset()

    def _update_clock(self):
        current_clock = self.clock_sub.get_clock()
        if self.motion_start_timestep is None:
            self.motion_start_timestep = current_clock
        elif self._last_clock_reading is not None and current_clock < self._last_clock_reading:
            offset_ms = round(self.motion_timestep * self.timestep_interval_ms)
            self.logger.warning("Clock sync returned earlier timestamp; adjusting motion timing anchor.")
            self.motion_start_timestep = current_clock - offset_ms
        self._last_clock_reading = current_clock
        elapsed_ms = current_clock - self.motion_start_timestep
        if self.motion_timestep == 0 and int(elapsed_ms // self.timestep_interval_ms) > 1:
            self.logger.warning(
                "Still at the beginning but the clock jumped ahead: elapsed_ms={elapsed_ms}, "
                "timestep_interval_ms={timestep_interval_ms}, motion_timestep={motion_timestep}. "
                "Re-anchoring to the current timestamp so the motion always starts from frame 0.",
                elapsed_ms=elapsed_ms,
                timestep_interval_ms=self.timestep_interval_ms,
                motion_timestep=self.motion_timestep,
            )
            self.motion_start_timestep = current_clock
            self._last_clock_reading = current_clock
            self.motion_timestep = 0
            return
        previous_motion_timestep = self.motion_timestep
        self.motion_timestep = int(elapsed_ms // self.timestep_interval_ms)
        if self.motion_timestep != previous_motion_timestep:
            self.logger.info(
                "Motion timestep advanced from {previous_motion_timestep} to {motion_timestep}",
                previous_motion_timestep=previous_motion_timestep,
                motion_timestep=self.motion_timestep,
            )

    def _handle_stop_policy(self):
        """Handle stop policy action."""
        self.use_policy_action = False
        self.get_ready_state = False
        self._stiff_hold_active = True
        self.logger.info("Actions set to stiff startup command")
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0

        self.motion_clip_progressing = False
        self.stabilization_mode = False
        self.motion_timestep = 0
        self.motion_start_timestep = None
        self._last_clock_reading = None
        self.robot_yaw_offset = 0.0

    def _handle_start_stabilization(self):
        """Enter stabilization mode — policy runs with first motion frame command."""
        if not self.use_policy_action:
            self.logger.warning("Press ']' first to enable policy, then 'd' to start stabilization")
            return

        self.stabilization_mode = True
        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None
        self._last_clock_reading = None
        self.logger.info(colored("🛡️  Entering stabilization mode - policy active with first motion frame command", "yellow", attrs=["bold"]))
        self.logger.info(colored("    Robot will stabilize and touch ground. Press 's' when ready to start motion.", "yellow"))

    def _handle_start_motion_clip(self):
        """Handle start motion clip action."""
        if self.stabilization_mode:
            self.stabilization_mode = False
            self.logger.info(colored("Exiting stabilization mode", "cyan"))

        self.clock_sub.reset_origin()
        self.motion_clip_progressing = True
        self.motion_start_timestep = None
        self.motion_timestep = 0
        self._last_clock_reading = None
        self.logger.info(colored("Starting motion clip playback", "blue"))

    def handle_keyboard_button(self, keycode):
        if keycode == "d":
            self._handle_start_stabilization()
        elif keycode == "s":
            if self.use_policy_action:
                self._handle_start_motion_clip()
            else:
                self.logger.warning("Press ']' first to enable policy, then 's' to start motion")
        else:
            super().handle_keyboard_button(keycode)

    def handle_joystick_button(self, cur_key):
        if cur_key == "X":
            self._handle_start_stabilization()
        elif cur_key == "start":
            self._handle_start_motion_clip()
        else:
            super().handle_joystick_button(cur_key)
        super()._print_control_status()

    def _capture_robot_yaw_offset(self):
        """Capture robot yaw when policy starts to use as reference offset."""
        robot_state_data = self.interface.get_low_state()
        if robot_state_data is None:
            self.robot_yaw_offset = 0.0
            self.logger.warning("Unable to capture robot yaw offset - missing robot state.")
            return

        base_quat_wxyz = robot_state_data[:, 3:7]
        _, _, yaw = quat_to_rpy(base_quat_wxyz.reshape(-1, 4)[0])
        self.robot_yaw_offset = float(yaw)
        self.logger.info(colored(f"Robot yaw offset captured at {np.degrees(yaw):.1f} deg", "blue"))

    def _remove_yaw_offset(self, quat_wxyz: np.ndarray, yaw_offset: float) -> np.ndarray:
        """Remove stored yaw offset from robot orientation quaternion."""
        if abs(yaw_offset) < 1e-6:
            return quat_wxyz
        yaw_quat = rpy_to_quat((0.0, 0.0, -yaw_offset)).reshape(1, 4)
        yaw_quat = np.broadcast_to(yaw_quat, quat_wxyz.shape)
        return quat_mul(yaw_quat, quat_wxyz)

    @staticmethod
    def _quat_yaw(quat_wxyz: np.ndarray) -> float:
        """Extract yaw angle from quaternion array of shape (1, 4)."""
        quat_flat = quat_wxyz.reshape(-1, 4)[0]
        _, _, yaw = quat_to_rpy(quat_flat)
        return float(yaw)

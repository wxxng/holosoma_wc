import csv
import json
import pickle
import signal
import socket
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import onnx
import onnxruntime
import pinocchio as pin
from defusedxml import ElementTree
from loguru import logger
from termcolor import colored
from holosoma_inference.utils.trajectory_generator import TrajectoryGenerator

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_types.robot import RobotConfig
from holosoma_inference.config.config_values import observation as observation_values
from holosoma_inference.policies import BasePolicy
from holosoma_inference.utils.clock import ClockSub
from holosoma_inference.utils.math.misc import get_index_of_a_in_b
from holosoma_inference.utils.math.quat import (
    matrix_from_quat,
    quat_inverse,
    quat_apply,
    quat_rotate_inverse,
    quat_mul,
    quat_to_rpy,
    rpy_to_quat,
    subtract_frame_transforms,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)

# Joint ordering mappings for G1 robot
# Config order (hardware/simulator): grouped by limb [L_leg, R_leg, waist, L_arm, L_hand, R_arm, R_hand]
# Asset order (model training): interleaved [LHP, RHP, WY, LHR, RHR, WR, LHY, RHY, WP, ...]
# G1_JOINT_NAMES order (motion clips): grouped by limb, 29-DOF only

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

# 29-DOF: Config -> Asset (for observations, same as first 29 of 43-DOF)
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

# 29-DOF: Asset -> Config (for actions)
ACTION_ASSET_TO_CONFIG_ORDER_29DOF = (
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27,
    12, 16, 20, 22, 24, 26, 28
)

# G1_JOINT_NAMES (29-DOF motion) -> Asset order (for motion commands)
G1_TO_ASSET_ORDER_29DOF = (
    12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 26, 27, 28
)

# 43-DOF: Action order used for WBT observations and policy outputs
ACTION_ORDER_43DOF = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_hand_index_0_joint",
    "left_hand_middle_0_joint",
    "left_hand_thumb_0_joint",
    "right_hand_index_0_joint",
    "right_hand_middle_0_joint",
    "right_hand_thumb_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_1_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_hand_thumb_2_joint",
)

# 29-DOF: Stabilization (master) action output order
STABILIZATION_ACTION_ORDER_29DOF = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

# 29-DOF: Stabilization (master) observation order (waist-first, asset order)
STABILIZATION_OBS_ORDER_29DOF = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
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
)

# Motion clip joint order for 43-DOF DEX3 clips (per user-provided ordering)
G1_DEX3_JOINT_NAMES = (
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
)

# 29-DOF motion command joint order (G1_JOINT_NAMES)
G1_MOTION_JOINT_NAMES_29 = (
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
)


class PinocchioRobot:
    def __init__(self, robot_cfg: RobotConfig, urdf_text: str):
        # create pinocchio robot
        xml_text = self._create_xml_from_urdf(urdf_text)
        
        self.robot_model = pin.buildModelFromXML(xml_text, pin.JointModelFreeFlyer())
        self.robot_data = self.robot_model.createData()

        # get joint names in pinocchio robot and real robot
        joint_names_in_real_robot = robot_cfg.dof_names
        joint_names_in_pinocchio_robot = [
            name for name in self.robot_model.names if name not in ["universe", "root_joint"]
        ]
        assert len(joint_names_in_pinocchio_robot) == len(joint_names_in_real_robot), (
            "The number of joints in the pinocchio robot and the real robot are not the same"
        )
        self.real2pinocchio_index = get_index_of_a_in_b(joint_names_in_pinocchio_robot, joint_names_in_real_robot)

        # get ref body frame id in pinocchio robot
        self.ref_body_frame_id = self.robot_model.getFrameId(robot_cfg.motion["body_name_ref"][0])

    def fk_and_get_ref_body_orientation_in_world(self, configuration: np.ndarray) -> np.ndarray:
        # forward kinematics
        pin.framesForwardKinematics(self.robot_model, self.robot_data, configuration)

        # get ref body pose in world
        ref_body_pose_in_world = self.robot_data.oMf[self.ref_body_frame_id]
        quaternion = pin.Quaternion(ref_body_pose_in_world.rotation)  # (4, )

        return np.expand_dims(quaternion.coeffs(), axis=0)  # xyzw, (1, 4)

    def fk_and_get_ref_body_pose_in_world(
        self,
        root_pos: np.ndarray,
        root_quat_wxyz: np.ndarray,
        dof_pos_real: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        root_pos = np.asarray(root_pos, dtype=np.float32).reshape(3,)
        root_quat_wxyz = np.asarray(root_quat_wxyz, dtype=np.float32).reshape(1, 4)
        root_ori_xyzw = wxyz_to_xyzw(root_quat_wxyz)[0]

        dof_pos_real = np.asarray(dof_pos_real, dtype=np.float32).reshape(-1)
        dof_pos_in_pinocchio = dof_pos_real[self.real2pinocchio_index]

        configuration = np.concatenate([root_pos, root_ori_xyzw, dof_pos_in_pinocchio], axis=0)
        pin.framesForwardKinematics(self.robot_model, self.robot_data, configuration)

        ref_body_pose_in_world = self.robot_data.oMf[self.ref_body_frame_id]
        quat_xyzw = pin.Quaternion(ref_body_pose_in_world.rotation).coeffs()
        pos = np.asarray(ref_body_pose_in_world.translation, dtype=np.float32)
        quat_wxyz = xyzw_to_wxyz(np.asarray(quat_xyzw, dtype=np.float32).reshape(1, 4))[0]
        return pos, quat_wxyz

    @staticmethod
    def _create_xml_from_urdf(urdf_text: str) -> str:
        """Strip visuals/collisions from URDF text and return XML text."""
        root = ElementTree.fromstring(urdf_text)

        def _is_visual_or_collision(tag: str) -> bool:
            # Handle optional XML namespaces by only checking the suffix after '}'.
            return tag.split("}")[-1] in {"visual", "collision"}

        for parent in root.iter():
            for child in list(parent):
                if _is_visual_or_collision(child.tag):
                    parent.remove(child)

        xml_text = ElementTree.tostring(root, encoding="unicode")
        if not xml_text.lstrip().startswith("<?xml"):
            xml_text = '<?xml version="1.0"?>\n' + xml_text
        return xml_text


class WholeBodyTrackingPolicy(BasePolicy):
    def __init__(self, config: InferenceConfig):
        # initialize timestep
        self.motion_timestep = max(int(config.task.motion_start_timestep), 0)
        self.motion_clip_progressing = False
        self.motion_start_timestep = None
        self.stabilization_mode = False  # Pre-motion stabilization stage
        
        # Motion library
        self.motion_data = None
        self.motion_dof_pos = None  # [num_frames, num_dofs] in motion clip order
        self.motion_dof_vel = None  # [num_frames, num_dofs] calculated from pos
        self.motion_obj_pos = None  # [num_frames, 3] world
        self.motion_obj_rot = None  # [num_frames, 4] world (wxyz)
        self._motion_obj_pos_world = None  # [num_frames, 3] original world frame (preserved before alignment)
        self._motion_obj_rot_world_wxyz = None  # [num_frames, 4] original world frame wxyz
        self.motion_table_pos = None  # [num_frames, 3] world
        self.motion_table_rot_wxyz = None  # [num_frames, 4] world (wxyz)
        self.motion_obj_name = None
        self.motion_root_pos_w = None  # [num_frames, 3] world
        self.motion_root_quat_w = None  # [num_frames, 4] world (wxyz)
        self.motion_align_ref_pos_w = None  # [3] initial human pos used to align motion frames
        self.motion_align_ref_quat_wxyz = None  # [4] initial human quat used to align motion frames (wxyz)
        self.motion_obj_pos_aligned = None  # [num_frames, 3] motion obj pos aligned to initial human frame
        self.motion_obj_rot_aligned_wxyz = None  # [num_frames, 4] motion obj rot aligned (wxyz)
        self.motion_table_pos_aligned = None  # [num_frames, 3] motion table pos aligned to initial human frame
        self.motion_table_rot_aligned_wxyz = None  # [num_frames, 4] motion table rot aligned (wxyz)
        self.motion_root_pos_w_aligned = None  # [num_frames, 3] aligned motion root (torso) pos
        self.motion_root_quat_w_aligned = None  # [num_frames, 4] aligned motion root (torso) quat (wxyz)
        self.motion_fps = None
        self.motion_length = 0
        self.motion_clip_key = None  # Which clip key from PKL to use
        self._motion_pkl_path = None
        self.motion_to_robot_joint_map = None  # Reorder indices
        self._scene_object_name = None
        self._scene_reset_counter = None
        
        # Interpolation counter for smooth transition to the configured start frame.
        # Phase 1 (0..rl_interp_steps-1): all joints except elbows interpolate to target.
        # Phase 2 (rl_interp_steps..2*rl_interp_steps-1): elbow joints interpolate to target.
        self.rl_interp_count = 0
        self.rl_interp_steps = 100  # Steps per phase (total = 2 * rl_interp_steps)
        self._elbow_interp_start_pos = None  # Elbow positions captured at start of phase 2
        self.interp_complete_dof_pos_logged = False  # Flag to log dof_pos once when interp complete
        
        # Global timestep counter for logging
        self.global_timestep = 0

        # Calculate timestep interval from rl_rate (e.g., 50Hz = 20ms intervals)
        self.timestep_interval_ms = 1000.0 / config.task.rl_rate

        # Initialize clock subscriber for synchronization
        self.clock_sub = ClockSub()
        self.clock_sub.start()
        self._last_clock_reading: int | None = None
        
        # Object height tracking for table removal
        self._initial_object_height = None
        self._table_removed = False
        self._table_removal_grace_steps = 0  # Countdown; skip height checks while > 0

        # Read use_sim_time from config
        self.use_sim_time = config.task.use_sim_time

        self._stiff_hold_active = True
        self.robot_yaw_offset = 0.0
        self.robot_world_origin_w = None  # [3] robot start position (x,y, z=0) used as world origin
        self.use_motion_tracking = False # Toggle between motion tracking and policy

        # soft stop with base policy
        self._soft_stop_active = False
        self._soft_stop_target_q = None

        # Stiff hold interpolation (Enter key -> stiff_startup_pos)
        self._stiff_hold_interp_count = 0
        self._stiff_hold_interp_steps = 50  # 1 second at 50Hz
        self._stiff_hold_start_q = None  # Starting position for interpolation

        # Object state freeze (f)
        self.freeze_object_state = False
        self._cached_object_pos_w = None   # (1,3)
        self._cached_object_quat_w = None  # (1,4) wxyz

        # Stabilization (master) policy state (initialized lazily)
        self._stabilization_policy_session = None
        self._stabilization_policy_input_names = []
        self._stabilization_policy_output_names = []
        self._stabilization_policy_callable = None
        self._stabilization_policy_path = None

        self._master_obs_config = None
        self._master_obs_dict = {}
        self._master_obs_dims = {}
        self._master_obs_scales = {}
        self._master_history_length_dict = {}
        self._master_obs_terms_sorted: dict[str, list[str]] = {}
        self._master_obs_history_buffers: dict[str, dict[str, deque[np.ndarray]]] = {}
        self._master_last_action = None  # action order (43) for master policy
        self._stabilization_last_action_43 = None  # raw action (43) in ACTION_ORDER_43DOF

        # Motion command is always body-only 29-DOF
        self.motion_body_indices = None
        self.motion_command_dofs = 29
        self.startup_pos = None
        self._last_sent_table_frame = None
        self._table_pose_send_warned = False

        # BPS code cache (per object name) — must be initialized before super().__init__
        # because setup_policy -> _load_motion_from_pkl may call _load_bps_code
        self._bps_cache: dict[str, np.ndarray] = {}
        self._obj_bps: np.ndarray | None = None  # current object BPS code (512,)

        # Optional recorded object trajectory in world frame for BPS motion commands.
        self.record_traj = bool(getattr(config.task, "record_traj", False))
        self._record_traj_active = False
        self._recorded_obj_pos_w = None  # [num_frames, 3]
        self._recorded_obj_rot_wxyz = None  # [num_frames, 4]
        self._recorded_traj_time_buffer: list[float] = []
        self._recorded_traj_pos_buffer: list[np.ndarray] = []
        self._recorded_traj_rot_buffer: list[np.ndarray] = []
        self._recorded_traj_path: Path | None = None

        super().__init__(config)

        # Load stiff startup parameters from robot config
        if config.robot.stiff_startup_pos is not None:
            self._stiff_hold_q = np.array(config.robot.stiff_startup_pos, dtype=np.float32).reshape(1, -1)
        else:
            # Fallback to default_dof_angles if not specified
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
        
        if config.robot.motor_kp is not None:
            self._default_kp = np.array(config.robot.motor_kp, dtype=np.float32)
        else:
            raise ValueError("Robot config must specify motor_kp for WBT policy")
        
        if config.robot.motor_kd is not None:
            self._default_kd = np.array(config.robot.motor_kd, dtype=np.float32)
        else:
            raise ValueError("Robot config must specify motor_kd for WBT policy")
        
        self.use_gen_traj = config.task.use_gen_traj
        self._traj_gen = None
        self._gen_traj_initialized = False
        if self.use_gen_traj:
            self._traj_gen = TrajectoryGenerator(
                rl_rate=config.task.rl_rate,
                mode=config.task.gen_traj_mode,
            )
        
        # Prompt user before entering stiff mode (only if stdin is available)
        def _show_warning():
            logger.warning(
                colored(
                    "⚠️  Non-interactive mode detected - cannot prompt for stiff mode confirmation!",
                    "red",
                    attrs=["bold"],
                )
            )

        # RViz object trajectory UDP sender — init before input() so it sends immediately
        self._rviz_traj_enabled = bool(config.task.rviz_traj)
        self._rviz_traj_host = str(config.task.rviz_traj_host).strip()
        self._rviz_traj_port = int(config.task.rviz_traj_port)
        self._rviz_traj_dt = max(float(config.task.rviz_traj_dt), 1e-3)
        self._rviz_traj_sock = None
        self._rviz_traj_addr = None
        if self._rviz_traj_enabled:
            self._rviz_traj_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._rviz_traj_addr = (self._rviz_traj_host, self._rviz_traj_port)
            logger.info(
                colored(
                    f"RViz object trajectory UDP sender enabled on {self._rviz_traj_addr} "
                    f"(dt={self._rviz_traj_dt:.3f}s)",
                    "cyan",
                )
            )
            self._send_rviz_traj()

        if sys.stdin.isatty():
            logger.info(colored("\n⚠️  Ready to enter stiff hold mode", "yellow", attrs=["bold"]))
            logger.info(colored("Press Enter to continue...", "yellow"))
            try:
                input()
                logger.info(colored("✓ Entering stiff hold mode", "green"))
            except EOFError:
                # [drockyd] seems like in some cases, input() will raise EOFError even in interactive mode.
                _show_warning()
        else:
            _show_warning()
        
        self.max_motion_length = None  # Safety limit for motion clip length
        
        # Debug mode flag
        self.debug_mode = getattr(config.task, 'debug', False)
        if self.debug_mode:
            logger.info(colored("🐛 Debug mode enabled: will log dof_pos and scaled_policy_action to NPZ, print every 100 steps", "magenta", attrs=["bold"]))
        
        # Debug logging (dof_pos, scaled_policy_action)
        self.debug_log_data = None
        self.debug_log_path = None
        
        # Debug: print interface value
        logger.info(colored(f"DEBUG: config.task.interface = '{config.task.interface}'", "cyan", attrs=["bold"]))
        
        if config.task.interface in ["lo", "eth0", "enp123s0"]:
            self.max_motion_length = None  # ~3 seconds at 50 fps
            logger.info(colored(f"Safety mode: Motion limited to {self.max_motion_length} frames", "yellow", attrs=["bold"]))
        else:
            logger.info(colored(f"NPZ logging disabled (interface is '{config.task.interface}', not 'lo' or 'eth0')", "yellow"))
        
        # Initialize debug logging if debug mode is enabled
        if self.debug_mode:
            self._init_debug_logging()

        # Motion logging (per-term observations + poses, starts on 's' key)
        self._motion_log_data: list[dict] | None = None
        self._motion_log_path: Path | None = None

        # Register SIGINT handler so Ctrl+C saves the log before exiting
        self._prev_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._on_sigint)

        # Object observation state (t-1 robot frame)
        self._prev_robot_anchor_quat_w = None
        self._prev_robot_obj_pos_w = None

        # Object pointcloud cache
        self._mesh_points_cache: dict[str, np.ndarray] = {}
        self._mesh_points_warned: set[str] = set()

        # Debug: stream object point cloud (world frame) to MuJoCo viewer via UDP
        self._debug_obj_pcd_enabled = bool(getattr(config.task, "debug_obj_pcd", False))
        self._debug_obj_pcd_stride = max(1, int(getattr(config.task, "debug_obj_pcd_stride", 1)))
        self._debug_obj_pcd_interval = max(1, int(getattr(config.task, "debug_obj_pcd_interval", 1)))
        self._debug_obj_pcd_port = int(getattr(config.task, "debug_obj_pcd_port", 10004))
        self._debug_obj_pcd_step = 0
        self._debug_obj_pcd_sock = None
        self._debug_obj_pcd_addr = None
        if self._debug_obj_pcd_enabled and config.task.interface in ["lo", "eth0"]:
            self._debug_obj_pcd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._debug_obj_pcd_addr = ("127.0.0.1", self._debug_obj_pcd_port)
            logger.info(
                colored(
                    f"Debug obj_pcd UDP sender enabled on {self._debug_obj_pcd_addr} "
                    f"(stride={self._debug_obj_pcd_stride}, interval={self._debug_obj_pcd_interval})",
                    "cyan",
                )
            )

        # Debug: visualize future object trajectory in the MuJoCo viewer via UDP
        self._debug_traj_viz_enabled = bool(getattr(config.task, "debug_traj_viz", False))
        self._debug_traj_viz_port = int(getattr(config.task, "debug_traj_viz_port", 10006))
        self._debug_traj_viz_sock = None
        self._debug_traj_viz_addr = None
        if self._debug_traj_viz_enabled and config.task.interface == "lo":
            self._debug_traj_viz_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._debug_traj_viz_addr = ("127.0.0.1", self._debug_traj_viz_port)
            logger.info(
                colored(
                    f"Debug trajectory viz UDP sender enabled on {self._debug_traj_viz_addr}",
                    "cyan",
                )
            )

        # MuJoCo twin: real-time full-state streaming for live visualization
        self._mujoco_twin_enabled = bool(getattr(config.task, "mujoco_twin", False))
        self._mujoco_twin_host = str(getattr(config.task, "mujoco_twin_host", "127.0.0.1")).strip()
        self._mujoco_twin_port = int(getattr(config.task, "mujoco_twin_port", 10008))
        self._mujoco_twin_sock = None
        self._mujoco_twin_addr = None
        self._mujoco_twin_first_packet_logged = False
        self._mujoco_twin_send_error_logged = False
        if self._mujoco_twin_enabled:
            self._mujoco_twin_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._mujoco_twin_addr = (self._mujoco_twin_host, self._mujoco_twin_port)
            logger.info(
                colored(
                    f"MuJoCo twin UDP sender enabled on {self._mujoco_twin_addr}",
                    "cyan",
                )
            )

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

    def __del__(self):
        """Save log files on cleanup."""
        # Save debug log
        if hasattr(self, 'debug_log_data') and self.debug_log_data is not None and len(self.debug_log_data.get("dof_pos", [])) > 0:
            self._save_debug_log()
        # Save motion pkl log
        if hasattr(self, '_motion_log_data') and self._motion_log_data:
            self._save_motion_log()
    
    def _init_debug_logging(self):
        """Initialize debug logging for dof_pos and scaled_policy_action."""
        # Create logs directory if it doesn't exist
        log_dir = Path("logs/action_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create NPZ file path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.debug_log_path = log_dir / f"debug_log_{timestamp}.npz"
        
        # Initialize data storage
        self.debug_log_data = {
            "global_timestep": [],
            "motion_timestep": [],
            "dof_pos": [],  # Current joint positions (config order)
            "scaled_policy_action": [],  # Scaled policy action (config order)
        }
        
        logger.info(colored(f"✓ Debug logging initialized: {self.debug_log_path.resolve()}", "magenta", attrs=["bold"]))

    def _get_interface_log_tag(self) -> str:
        interface = str(getattr(self.config.task, "interface", "unknown")).strip()
        if not interface:
            interface = "unknown"
        return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in interface)

    def _save_debug_log(self):
        """Save debug log data to NPZ file."""
        if self.debug_log_path is None or self.debug_log_data is None:
            return
        if len(self.debug_log_data.get("dof_pos", [])) == 0:
            return
        
        try:
            # Convert lists to numpy arrays
            save_dict = {
                key: np.array(value) for key, value in self.debug_log_data.items()
            }
            
            # Save to NPZ file
            np.savez_compressed(self.debug_log_path, **save_dict)
            logger.info(colored(
                f"✓ Debug log saved: {self.debug_log_path.resolve()} "
                f"({len(self.debug_log_data['dof_pos'])} timesteps)",
                "magenta", attrs=["bold"]
            ))
        except Exception as e:
            logger.error(colored(f"Failed to save debug log: {e}", "red"))

    # -------------------------------------------------------------------------
    # Motion logging (observation terms + poses, collected after 's' key)
    # -------------------------------------------------------------------------

    def _on_sigint(self, signum, frame):
        """Save logs on Ctrl+C, then propagate signal."""
        logger.info(colored("Ctrl+C detected — saving logs...", "yellow", attrs=["bold"]))
        if self._motion_log_data:
            self._save_motion_log()
        # Restore previous handler and re-raise so the process exits normally
        signal.signal(signal.SIGINT, self._prev_sigint_handler)
        signal.raise_signal(signal.SIGINT)

    def _init_motion_log(self):
        """Initialize per-motion-clip pkl log buffer."""
        log_dir = Path("logs/sim2real/wbt")
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        interface_tag = self._get_interface_log_tag()
        self._motion_log_path = log_dir / f"wbt_log_{interface_tag}_{timestamp}.pkl"
        self._motion_log_data = []
        logger.info(colored(f"✓ Motion log initialized: {self._motion_log_path.resolve()}", "green", attrs=["bold"]))

    def _save_motion_log(self):
        """Save motion log to pkl file."""
        if not self._motion_log_data or self._motion_log_path is None:
            return
        try:
            with open(self._motion_log_path, "wb") as f:
                pickle.dump(self._motion_log_data, f)
            logger.info(colored(
                f"✓ Motion log saved: {self._motion_log_path.resolve()} "
                f"({len(self._motion_log_data)} timesteps)",
                "green", attrs=["bold"],
            ))
        except Exception as e:
            logger.error(colored(f"Failed to save motion log: {e}", "red"))

    @staticmethod
    def _pose_to_log_array(
        pos: np.ndarray | None,
        quat_wxyz: np.ndarray | None,
    ) -> np.ndarray | None:
        if pos is None or quat_wxyz is None:
            return None
        pos_arr = np.asarray(pos, dtype=np.float32).reshape(-1)
        quat_arr = np.asarray(quat_wxyz, dtype=np.float32).reshape(-1)
        if pos_arr.size != 3 or quat_arr.size != 4:
            return None
        return np.concatenate([pos_arr, quat_arr], axis=0).astype(np.float32, copy=False)

    def _init_recorded_traj_path(self) -> Path:
        log_dir = Path("logs/recorded_object_traj")
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return log_dir / f"recorded_object_traj_{timestamp}.pkl"

    def _start_record_traj(self) -> None:
        self._recorded_traj_time_buffer = []
        self._recorded_traj_pos_buffer = []
        self._recorded_traj_rot_buffer = []
        self._recorded_traj_path = self._init_recorded_traj_path()
        self._record_traj_active = True
        logger.info(
            colored(
                "⏺️  Started recording object trajectory in world frame. Press the same button again to stop.",
                "yellow",
                attrs=["bold"],
            )
        )

    def _stop_record_traj(self) -> None:
        self._record_traj_active = False
        if (
            not self._recorded_traj_time_buffer
            or not self._recorded_traj_pos_buffer
            or not self._recorded_traj_rot_buffer
        ):
            logger.warning("Recorded trajectory is empty; keeping the existing motion object trajectory.")
            return

        raw_time = np.asarray(self._recorded_traj_time_buffer, dtype=np.float64)
        raw_time = raw_time - raw_time[0]
        raw_pos = np.stack(self._recorded_traj_pos_buffer, axis=0).astype(np.float32, copy=False)
        raw_quat = self._normalize_quat_wxyz(
            np.stack(self._recorded_traj_rot_buffer, axis=0).astype(np.float32, copy=False)
        )

        resampled_time, pos, quat = self._resample_recorded_traj(
            raw_time,
            raw_pos,
            raw_quat,
            float(self.config.task.rl_rate),
        )
        source_fps = self._estimate_sample_rate(raw_time)
        self._recorded_obj_pos_w = pos
        self._recorded_obj_rot_wxyz = quat

        payload = {
            "created_at": datetime.now().isoformat(),
            "fps": float(self.config.task.rl_rate),
            "source_fps_estimate": None if source_fps is None else float(source_fps),
            "source_time_from_start_sec": raw_time.astype(np.float32, copy=False),
            "source_obj_pos_w": raw_pos,
            "source_obj_rot_wxyz": raw_quat,
            "resampled_time_from_start_sec": resampled_time,
            "obj_pos_w": pos,
            "obj_rot_wxyz": quat,
            "motion_clip_key": self.motion_clip_key,
            "motion_obj_name": self.motion_obj_name,
        }
        if self._recorded_traj_path is not None:
            try:
                with self._recorded_traj_path.open("wb") as f:
                    pickle.dump(payload, f)
                logger.info(
                    colored(
                        f"✓ Recorded object trajectory saved: {self._recorded_traj_path.resolve()} "
                        f"({raw_pos.shape[0]} raw frames -> {pos.shape[0]} frames @ {self.config.task.rl_rate:.1f}Hz)",
                        "green",
                        attrs=["bold"],
                    )
                )
            except Exception as exc:
                logger.error(colored(f"Failed to save recorded object trajectory: {exc}", "red"))

        logger.info(
            colored(
                "Using recorded world-frame object trajectory for BPS motion commands "
                f"({raw_pos.shape[0]} raw frames -> {pos.shape[0]} resampled frames).",
                "cyan",
                attrs=["bold"],
            )
        )

    def _toggle_record_traj(self) -> None:
        if not self.record_traj:
            return
        if self._record_traj_active:
            self._stop_record_traj()
        else:
            self._start_record_traj()

    def _record_traj_step(self) -> None:
        if not self._record_traj_active:
            return
        if not hasattr(self.interface, "get_object_pose_world_with_timestamp"):
            pos_w = self._get_object_pos_w()
            quat_wxyz = self._get_object_quat_w()
            pose_timestamp = None
        else:
            pose, pose_timestamp = self.interface.get_object_pose_world_with_timestamp()
            if pose is None or pose_timestamp is None:
                return
            pose = np.asarray(pose, dtype=np.float32).reshape(-1)
            if pose.size < 7:
                return
            pos_w = pose[:3].reshape(1, 3)
            quat_wxyz = xyzw_to_wxyz(pose[3:7].reshape(1, 4))

        if pos_w is None or quat_wxyz is None:
            return
        if pose_timestamp is None:
            pose_timestamp = len(self._recorded_traj_time_buffer) / max(float(self.config.task.rl_rate), 1e-6)
        if self._recorded_traj_time_buffer and pose_timestamp <= self._recorded_traj_time_buffer[-1]:
            return
        self._recorded_traj_time_buffer.append(float(pose_timestamp))
        self._recorded_traj_pos_buffer.append(pos_w[0].astype(np.float32, copy=True))
        self._recorded_traj_rot_buffer.append(self._normalize_quat_wxyz(quat_wxyz)[0].astype(np.float32, copy=True))

    @staticmethod
    def _estimate_sample_rate(sample_times: np.ndarray) -> float | None:
        if sample_times.size < 2:
            return None
        dt = np.diff(sample_times)
        dt = dt[dt > 1e-6]
        if dt.size == 0:
            return None
        return float(1.0 / np.median(dt))

    def _quat_slerp_wxyz(self, q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
        q0_n = self._normalize_quat_wxyz(np.asarray(q0, dtype=np.float32).reshape(1, 4))[0]
        q1_n = self._normalize_quat_wxyz(np.asarray(q1, dtype=np.float32).reshape(1, 4))[0]
        dot = float(np.dot(q0_n, q1_n))
        if dot < 0.0:
            q1_n = -q1_n
            dot = -dot
        dot = float(np.clip(dot, -1.0, 1.0))

        if dot > 0.9995:
            q = q0_n + float(alpha) * (q1_n - q0_n)
            return self._normalize_quat_wxyz(q.reshape(1, 4))[0]

        theta_0 = float(np.arccos(dot))
        sin_theta_0 = float(np.sin(theta_0))
        if sin_theta_0 < 1e-6:
            return q0_n

        theta = theta_0 * float(alpha)
        s0 = np.sin(theta_0 - theta) / sin_theta_0
        s1 = np.sin(theta) / sin_theta_0
        q = s0 * q0_n + s1 * q1_n
        return self._normalize_quat_wxyz(q.reshape(1, 4))[0]

    def _resample_recorded_traj(
        self,
        sample_times: np.ndarray,
        pos_seq: np.ndarray,
        rot_wxyz_seq: np.ndarray,
        target_fps: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sample_times = np.asarray(sample_times, dtype=np.float64).reshape(-1)
        pos_seq = np.asarray(pos_seq, dtype=np.float32)
        rot_wxyz_seq = self._normalize_quat_wxyz(np.asarray(rot_wxyz_seq, dtype=np.float32))

        if sample_times.size == 0 or pos_seq.shape[0] == 0 or rot_wxyz_seq.shape[0] == 0:
            return (
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 4), dtype=np.float32),
            )

        if sample_times.size == 1 or target_fps <= 0.0:
            return (
                np.array([0.0], dtype=np.float32),
                pos_seq[:1].astype(np.float32, copy=False),
                rot_wxyz_seq[:1].astype(np.float32, copy=False),
            )

        target_dt = 1.0 / target_fps
        duration = float(sample_times[-1])
        target_times = np.arange(0.0, duration + 0.5 * target_dt, target_dt, dtype=np.float64)
        if target_times.size == 0:
            target_times = np.array([0.0], dtype=np.float64)

        pos_resampled = np.stack(
            [np.interp(target_times, sample_times, pos_seq[:, axis]) for axis in range(3)],
            axis=1,
        ).astype(np.float32, copy=False)

        quat_resampled = np.zeros((target_times.shape[0], 4), dtype=np.float32)
        for idx, t in enumerate(target_times):
            right = int(np.searchsorted(sample_times, t, side="right"))
            if right <= 0:
                quat_resampled[idx] = rot_wxyz_seq[0]
            elif right >= sample_times.shape[0]:
                quat_resampled[idx] = rot_wxyz_seq[-1]
            else:
                left = right - 1
                t0 = sample_times[left]
                t1 = sample_times[right]
                alpha = 0.0 if t1 <= t0 else float((t - t0) / (t1 - t0))
                quat_resampled[idx] = self._quat_slerp_wxyz(
                    rot_wxyz_seq[left], rot_wxyz_seq[right], alpha
                )

        return target_times.astype(np.float32, copy=False), pos_resampled, quat_resampled

    def _policy_step_hook(self, robot_state_data):
        super()._policy_step_hook(robot_state_data)
        self._record_traj_step()

    def _collect_motion_log_step(
        self,
        robot_state_data: np.ndarray,
        current_obs_dict: dict[str, dict[str, np.ndarray]],
    ) -> None:
        """Append one timestep of per-term observations and key poses to the motion log."""
        if self._motion_log_data is None:
            return

        obj_pos_torso, obj_quat_torso_wxyz = self._get_object_pose_torso()
        obj_pos_world = self._get_object_pos_w()
        obj_quat_world_wxyz = self._get_object_quat_w()
        pelvis_pose_world = np.asarray(robot_state_data[0, :7], dtype=np.float32).copy()

        entry = {
            "motion_timestep": int(self.motion_timestep),
            "global_timestep": int(self.global_timestep),
            "observation_terms": {
                group: {
                    term: np.asarray(value, dtype=np.float32).reshape(-1).copy()
                    for term, value in term_dict.items()
                }
                for group, term_dict in current_obs_dict.items()
            },
            "scaled_policy_action": None
            if getattr(self, "scaled_policy_action", None) is None
            else np.asarray(self.scaled_policy_action, dtype=np.float32).reshape(-1).copy(),
            "object_pose_torso": self._pose_to_log_array(obj_pos_torso, obj_quat_torso_wxyz),
            "object_pose_world": self._pose_to_log_array(obj_pos_world, obj_quat_world_wxyz),
            "pelvis_pose_world": pelvis_pose_world,
        }
        self._motion_log_data.append(entry)

    # -------------------------------------------------------------------------
    # Stabilization (master) policy helpers
    # -------------------------------------------------------------------------

    def _ensure_master_obs_config(self) -> None:
        if self._master_obs_config is not None:
            return
        # Master stabilization policy expects 29-DOF tracking-style obs (673 dims).
        self._master_obs_config = observation_values.wbt_29dof_tracking
        self._master_obs_dict = self._master_obs_config.obs_dict
        self._master_obs_dims = self._master_obs_config.obs_dims
        self._master_obs_scales = self._master_obs_config.obs_scales
        self._master_history_length_dict = self._master_obs_config.history_length_dict

        self._master_obs_terms_sorted = {group: list(terms) for group, terms in self._master_obs_dict.items()}
        self._master_obs_history_buffers = {}
        for group, term_names in self._master_obs_terms_sorted.items():
            history_len = self._master_history_length_dict.get(group, 1)
            self._master_obs_history_buffers[group] = {
                term: deque(maxlen=history_len) for term in term_names
            }

        if self._master_last_action is None:
            self._master_last_action = np.zeros((1, self.motion_command_dofs), dtype=np.float32)

    def _reset_master_obs_history(self) -> None:
        self._ensure_master_obs_config()
        for group_buffers in self._master_obs_history_buffers.values():
            for buffer in group_buffers.values():
                buffer.clear()
        if self._master_last_action is not None:
            self._master_last_action.fill(0.0)

    def _init_stabilization_policy(self, holosoma_root: Path) -> None:
        if self._stabilization_policy_session is not None:
            return
        master_path = (
            holosoma_root
            / "src"
            / "holosoma_inference"
            / "holosoma_inference"
            / "models"
            / "wbt"
            / "tracking"
            / "master_policy.onnx"
        )
        self._stabilization_policy_path = master_path
        if not master_path.exists():
            logger.warning(f"Master policy not found: {master_path}")
            return

        self._stabilization_policy_session = onnxruntime.InferenceSession(str(master_path))
        self._stabilization_policy_input_names = [
            inp.name for inp in self._stabilization_policy_session.get_inputs()
        ]
        self._stabilization_policy_output_names = [
            out.name for out in self._stabilization_policy_session.get_outputs()
        ]

        def _policy_act(input_feed):
            outputs = self._stabilization_policy_session.run(
                self._stabilization_policy_output_names, input_feed
            )
            return outputs[0]

        self._stabilization_policy_callable = _policy_act
        self._ensure_master_obs_config()
        logger.info(colored(f"Loaded master policy for stabilization: {master_path}", "cyan"))

    def _get_obs_body_indices_config(self) -> np.ndarray:
        if not hasattr(self, "_obs_body_indices_config"):
            # Stabilization observations use waist-first 29-DOF order (asset order)
            self._obs_body_indices_config = np.array(
                get_index_of_a_in_b(list(STABILIZATION_OBS_ORDER_29DOF), self.dof_names),
                dtype=np.int64,
            )
        return self._obs_body_indices_config

    def _get_action_body_indices_config(self) -> np.ndarray:
        if not hasattr(self, "_action_body_indices_config"):
            # Stabilization actions use the policy output order
            self._action_body_indices_config = np.array(
                get_index_of_a_in_b(list(STABILIZATION_ACTION_ORDER_29DOF), self.dof_names),
                dtype=np.int64,
            )
        return self._action_body_indices_config

    def _get_stabilization_action_order_indices(self) -> np.ndarray:
        """Map stabilization action order (29) into ACTION_ORDER_43DOF indices for scaling."""
        if not hasattr(self, "_stabilization_action_order_indices"):
            self._stabilization_action_order_indices = np.array(
                get_index_of_a_in_b(list(STABILIZATION_ACTION_ORDER_29DOF), list(ACTION_ORDER_43DOF)),
                dtype=np.int64,
            )
        return self._stabilization_action_order_indices

    def _get_stabilization_action_to_obs_indices(self) -> np.ndarray:
        """Map stabilization action order into observation order."""
        if not hasattr(self, "_stabilization_action_to_obs_indices"):
            self._stabilization_action_to_obs_indices = np.array(
                get_index_of_a_in_b(list(STABILIZATION_OBS_ORDER_29DOF), list(STABILIZATION_ACTION_ORDER_29DOF)),
                dtype=np.int64,
            )
        return self._stabilization_action_to_obs_indices

    def _seed_obs_history_with_current(self, robot_state_data) -> None:
        """Fill main policy history buffers with the current observation (repeat)."""
        current_obs_buffer_dict = self.get_current_obs_buffer_dict(robot_state_data)
        current_obs_dict = self.parse_current_obs_dict(current_obs_buffer_dict)

        for group, term_names in self.obs_terms_sorted.items():
            history_len = self.history_length_dict.get(group, 1)
            for term in term_names:
                obs = np.asarray(current_obs_dict[group][term], dtype=np.float32, order="C")
                if obs.ndim == 1:
                    obs = obs.reshape(1, -1)
                buffer = self.obs_history_buffers[group][term]
                buffer.clear()
                for _ in range(history_len):
                    buffer.append(obs.copy())

        # Update flattened history buffers to match seeded history
        self.obs_buf_dict = {}
        for group, term_names in self.obs_terms_sorted.items():
            history_len = self.history_length_dict.get(group, 1)
            flattened_terms: list[np.ndarray] = []
            for term in term_names:
                obs = np.asarray(current_obs_dict[group][term], dtype=np.float32, order="C")
                if obs.ndim == 1:
                    obs = obs.reshape(1, -1)
                stacked = np.repeat(obs[:, None, :], history_len, axis=1)
                flattened_terms.append(stacked.reshape(obs.shape[0], -1))
            self.obs_buf_dict[group] = (
                np.concatenate(flattened_terms, axis=1).astype(np.float32, copy=False)
                if flattened_terms
                else np.zeros((1, 0), dtype=np.float32)
            )

    def _get_action_hand_indices_config(self) -> np.ndarray:
        if not hasattr(self, "_action_hand_indices_config"):
            if self.num_dofs <= self.motion_command_dofs:
                self._action_hand_indices_config = np.array([], dtype=np.int64)
            else:
                body_indices = self._get_action_body_indices_config()
                mask = np.ones(self.num_dofs, dtype=bool)
                mask[body_indices] = False
                self._action_hand_indices_config = np.nonzero(mask)[0].astype(np.int64)
        return self._action_hand_indices_config

    def _get_configured_motion_start_timestep(self) -> int:
        return max(int(self.config.task.motion_start_timestep), 0)

    def _get_motion_start_frame_index(self) -> int:
        start_timestep = self._get_configured_motion_start_timestep()
        if self.motion_length <= 0:
            return start_timestep
        return int(np.clip(start_timestep, 0, self.motion_length - 1))

    def _reset_motion_timing(self) -> None:
        self.motion_timestep = self._get_motion_start_frame_index()
        self.motion_start_timestep = None
        self._last_clock_reading = None

    def _validate_motion_timestep_config(self) -> None:
        if self.motion_length <= 0:
            return

        start_timestep = self._get_configured_motion_start_timestep()
        if start_timestep >= self.motion_length:
            raise ValueError(
                f"motion_start_timestep={start_timestep} is out of range for clip "
                f"'{self.motion_clip_key}' with {self.motion_length} frames"
            )

        end_timestep = self.config.task.motion_end_timestep
        if end_timestep is None:
            return
        if end_timestep <= start_timestep:
            raise ValueError(
                f"motion_end_timestep={end_timestep} must be greater than "
                f"motion_start_timestep={start_timestep}"
            )
        if end_timestep > self.motion_length:
            raise ValueError(
                f"motion_end_timestep={end_timestep} exceeds clip '{self.motion_clip_key}' "
                f"length {self.motion_length}"
            )

    def _resample_motion_arrays(self, speed: float) -> None:
        """Resample all loaded motion arrays to play at ``speed`` × the original rate.

        speed > 1 → fewer frames (faster);  speed < 1 → more frames (slower).
        Quaternion arrays are renormalized after interpolation.
        """
        old_len = self.motion_length
        new_len = max(1, int(round(old_len / speed)))
        old_t = np.linspace(0.0, 1.0, old_len)
        new_t = np.linspace(0.0, 1.0, new_len)

        def _interp(arr: np.ndarray) -> np.ndarray:
            return np.stack(
                [np.interp(new_t, old_t, arr[:, d]) for d in range(arr.shape[1])],
                axis=1,
            ).astype(np.float32)

        def _interp_quat(arr: np.ndarray) -> np.ndarray:
            out = _interp(arr)
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            return (out / np.maximum(norms, 1e-8)).astype(np.float32)

        self.motion_dof_pos = _interp(self.motion_dof_pos)
        if self.motion_obj_pos is not None:
            self.motion_obj_pos = _interp(self.motion_obj_pos)
        if self.motion_obj_rot is not None:
            self.motion_obj_rot = _interp_quat(self.motion_obj_rot)
        if self.motion_table_pos is not None:
            self.motion_table_pos = _interp(self.motion_table_pos)
        if self.motion_table_rot_wxyz is not None:
            self.motion_table_rot_wxyz = _interp_quat(self.motion_table_rot_wxyz)
        if self.motion_root_pos_w is not None:
            self.motion_root_pos_w = _interp(self.motion_root_pos_w)
        if self.motion_root_quat_w is not None:
            self.motion_root_quat_w = _interp_quat(self.motion_root_quat_w)
        self.motion_length = new_len
        logger.info(
            colored(
                f"Reference speed {speed:.3g}×: {old_len} → {new_len} frames "
                f"({new_len / self.motion_fps:.2f}s)",
                "yellow",
            )
        )

    def _get_target_pos_config_first_frame(self) -> np.ndarray | None:
        if self.motion_data is None or self.motion_dof_pos is None or self.motion_length == 0:
            return None
        target_pos_motion_order = self.motion_dof_pos[self._get_motion_start_frame_index()]

        if self.num_dofs == 43 and target_pos_motion_order.shape[0] == 43:
            # Motion clip is 43-DOF in G1_DEX3_JOINT_NAMES order; map directly to config order.
            motion_to_config = np.array(
                get_index_of_a_in_b(self.dof_names, list(G1_DEX3_JOINT_NAMES)), dtype=np.int64
            )
            target_pos_config = target_pos_motion_order[motion_to_config]
        else:
            # Fallback to 29-DOF motion handling (G1_JOINT_NAMES order)
            motion_to_asset = getattr(self, "motion_to_asset_joint_map", None)
            if motion_to_asset is None:
                motion_to_asset = np.array(G1_TO_ASSET_ORDER_29DOF)
            target_pos_asset = target_pos_motion_order[motion_to_asset]  # [29] in asset order

            reorder_map = np.array(ASSET_TO_CONFIG_ORDER_29DOF)
            target_pos_config_29 = target_pos_asset[reorder_map]  # [29] in 29-DOF config order

            if self.num_dofs == 43:
                # Hand default values (config order)
                left_hand_defaults = np.array(
                    [0.0, 1.0, 1.7, -1.57, -1.7, -1.57, -1.7], dtype=np.float32
                )
                right_hand_defaults = np.array(
                    [0.0, -1.0, -1.7, 1.57, 1.7, 1.57, 1.7], dtype=np.float32
                )

                target_pos_config = np.concatenate(
                    [
                        target_pos_config_29[:22],  # left_leg + right_leg + waist + left_arm + left_wrist
                        left_hand_defaults,  # left hand (7 joints)
                        target_pos_config_29[22:],  # right_arm + right_wrist (7 joints)
                        right_hand_defaults,  # right hand (7 joints)
                    ]
                )
            else:
                target_pos_config = target_pos_config_29

        return target_pos_config.reshape(1, -1).astype(np.float32, copy=False)

    def _get_master_obs_buffer_dict(self, robot_state_data) -> dict[str, np.ndarray]:
        self._ensure_master_obs_config()
        current_obs_buffer_dict: dict[str, np.ndarray] = {}

        dof_pos_config = robot_state_data[:, 7 : 7 + self.num_dofs]
        dof_vel_config = robot_state_data[
            :, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs
        ]

        # Use stabilization observation order (waist-first 29-DOF) for proprioception
        obs_body_indices = self._get_obs_body_indices_config()
        dof_pos_action = dof_pos_config[:, obs_body_indices]
        dof_vel_action = dof_vel_config[:, obs_body_indices]
        default_dof_angles_action = self.default_dof_angles[obs_body_indices]

        current_obs_buffer_dict["dof_pos"] = dof_pos_action - default_dof_angles_action
        current_obs_buffer_dict["dof_vel"] = dof_vel_action
        current_obs_buffer_dict["base_ang_vel"] = robot_state_data[
            :, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6
        ]

        base_quat_wxyz = robot_state_data[:, 3:7]
        gravity_world = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
        current_obs_buffer_dict["projected_gravity"] = quat_rotate_inverse(base_quat_wxyz, gravity_world)

        # Previous action: keep policy raw output order (no reordering)
        if self._master_last_action is None:
            self._master_last_action = np.zeros((1, self.motion_command_dofs), dtype=np.float32)
        current_obs_buffer_dict["actions"] = self._master_last_action

        # Stabilization uses the first motion frame only
        if self._soft_stop_active:
            current_obs_buffer_dict["motion_command_sequence"] = self._get_motion_command_sequence_soft_stop()
        else:
            current_obs_buffer_dict["motion_command_sequence"] = self._get_motion_command_sequence_first_frame()

        return current_obs_buffer_dict

    def _prepare_master_group_observations(self, robot_state_data) -> dict[str, np.ndarray]:
        current_obs_buffer_dict = self._get_master_obs_buffer_dict(robot_state_data)
        current_obs_dict: dict[str, dict[str, np.ndarray]] = {}
        for group, term_names in self._master_obs_terms_sorted.items():
            grouped_terms: dict[str, np.ndarray] = {}
            for term in term_names:
                if term not in current_obs_buffer_dict:
                    raise KeyError(f"Master obs term '{term}' missing from current observation buffer.")
                term_obs = current_obs_buffer_dict[term]
                if term_obs.ndim == 1:
                    term_obs = term_obs.reshape(1, -1)
                scale = self._master_obs_scales[term]
                grouped_terms[term] = (term_obs * scale).astype(np.float32, copy=False)
            current_obs_dict[group] = grouped_terms
        return self._update_master_obs_history(current_obs_dict)

    def _update_master_obs_history(
        self, current_obs_dict: dict[str, dict[str, np.ndarray]]
    ) -> dict[str, np.ndarray]:
        group_outputs: dict[str, np.ndarray] = {}

        for group, term_dict in current_obs_dict.items():
            history_len = self._master_history_length_dict.get(group, 1)
            flattened_terms: list[np.ndarray] = []

            for term in self._master_obs_terms_sorted[group]:
                obs = np.asarray(term_dict[term], dtype=np.float32, order="C")
                if obs.ndim == 1:
                    obs = obs.reshape(1, -1)

                buffer = self._master_obs_history_buffers[group][term]
                buffer.append(obs.copy())

                history = list(buffer)
                if len(history) < history_len:
                    missing = history_len - len(history)
                    if group in {"proprio_body", "proprio_hand"}:
                        pad_value = history[0] if history else obs
                        history = [pad_value.copy()] * missing + history
                    else:
                        history = [np.zeros_like(obs)] * missing + history

                stacked = np.stack(history[-history_len:], axis=1)
                flattened_terms.append(stacked.reshape(obs.shape[0], -1))

            group_outputs[group] = (
                np.concatenate(flattened_terms, axis=1).astype(np.float32, copy=False)
                if flattened_terms
                else np.zeros((1, 0), dtype=np.float32)
            )

        return group_outputs

    def _load_motion_from_pkl(self, pkl_path: str, clip_key: str = None):
        """Load motion data from PKL file and prepare for inference.
        
        Args:
            pkl_path: Path to PKL file containing motion clips
            clip_key: Which clip key to use. If None, uses the first clip.
        """
        logger.info(f"Loading motion from {pkl_path}, clip key: {clip_key}")
        
        # Load PKL file
        self.motion_data = joblib.load(pkl_path)
        motion_keys = list(self.motion_data.keys())
        
        if clip_key is None:
            clip_key = motion_keys[0]
            logger.info(f"No clip key specified, using first clip: {clip_key}")
        
        if clip_key not in self.motion_data:
            # Build detailed error message with motion lengths
            clips_info = []
            for key in motion_keys:
                clip_data = self.motion_data[key]
                fps = clip_data['fps']
                num_frames = clip_data['dof_pos'].shape[0]
                duration = num_frames / fps
                clips_info.append(f"  - '{key}': {num_frames} frames ({duration:.2f}s @ {fps}Hz)")
            
            error_msg = f"Clip key '{clip_key}' not found. Available clips:\n" + "\n".join(clips_info)
            raise ValueError(error_msg)
        
        clip_name = clip_key
        clip_data = self.motion_data[clip_name]
        
        logger.info(f"Using clip: {clip_name}")
        logger.info(f"  FPS: {clip_data['fps']}")
        logger.info(f"  dof_pos shape: {clip_data['dof_pos'].shape}")
        
        # Extract data
        self.motion_fps = clip_data['fps']
        self.motion_clip_key = clip_key
        self.motion_dof_pos = clip_data['dof_pos']
        self.motion_length = self.motion_dof_pos.shape[0]
        obj_pos_raw = clip_data.get("obj_pos")
        if obj_pos_raw is None:
            self.motion_obj_pos = None
        else:
            obj_pos = np.asarray(obj_pos_raw, dtype=np.float32)
            if obj_pos.ndim != 2 or obj_pos.shape[1] != 3:
                logger.warning(f"Invalid obj_pos shape {obj_pos.shape}; disabling motion obj pos.")
                self.motion_obj_pos = None
            else:
                self.motion_obj_pos = obj_pos

        obj_rot_raw = clip_data.get("obj_rot")
        if obj_rot_raw is None:
            self.motion_obj_rot = None
        else:
            obj_rot = np.asarray(obj_rot_raw, dtype=np.float32)
            if obj_rot.ndim != 2 or obj_rot.shape[1] != 4:
                logger.warning(f"Invalid obj_rot shape {obj_rot.shape}; disabling motion obj rot.")
                self.motion_obj_rot = None
            else:
                # Motion clip rotations are xyzw; convert to wxyz for internal use.
                self.motion_obj_rot = xyzw_to_wxyz(obj_rot)

        table_pos_raw = clip_data.get("table_pos")
        if table_pos_raw is None:
            self.motion_table_pos = None
        else:
            table_pos = np.asarray(table_pos_raw, dtype=np.float32)
            if table_pos.ndim != 2 or table_pos.shape[1] != 3:
                logger.warning(f"Invalid table_pos shape {table_pos.shape}; disabling motion table pos.")
                self.motion_table_pos = None
            else:
                self.motion_table_pos = table_pos

        table_rot_raw = clip_data.get("table_rot")
        if table_rot_raw is None:
            self.motion_table_rot_wxyz = None
        else:
            table_rot = np.asarray(table_rot_raw, dtype=np.float32)
            if table_rot.ndim != 2 or table_rot.shape[1] != 4:
                logger.warning(f"Invalid table_rot shape {table_rot.shape}; disabling motion table rot.")
                self.motion_table_rot_wxyz = None
            else:
                # Motion clip rotations are xyzw; convert to wxyz for internal use.
                self.motion_table_rot_wxyz = xyzw_to_wxyz(table_rot)
        self.motion_obj_name = clip_data.get("obj_name")
        if isinstance(self.motion_obj_name, (list, np.ndarray)):
            self.motion_obj_name = self.motion_obj_name[0]
        if self.motion_obj_name is not None:
            self.motion_obj_name = str(self.motion_obj_name).strip()

        # Load BPS code for the object (used by g1-43dof-object-bps policy)
        if "obj_bps" in self.obs_dims and self.motion_obj_name:
            self._obj_bps = self._load_bps_code(self.motion_obj_name)

        # Optional: motion root pose for motion-frame transforms
        root_pos_raw = None
        root_quat_raw = None
        root_quat_key = None

        # Preferred: use global_*_extend first body (robot root)
        if "global_translation_extend" in clip_data and "global_rotation_extend" in clip_data:
            root_pos_raw = np.asarray(clip_data["global_translation_extend"], dtype=np.float32)[:, 0, :]
            root_quat_raw = np.asarray(clip_data["global_rotation_extend"], dtype=np.float32)[:, 0, :]
            root_quat_key = "global_rotation_extend_xyzw"
        if root_pos_raw is not None:
            root_pos = np.asarray(root_pos_raw, dtype=np.float32)
            if root_pos.ndim == 2 and root_pos.shape[1] == 3:
                self.motion_root_pos_w = root_pos
            else:
                logger.warning(f"Invalid root_pos shape {root_pos.shape}; ignoring motion root position.")

        if root_quat_raw is not None:
            root_quat = np.asarray(root_quat_raw, dtype=np.float32)
            if root_quat.ndim == 2 and root_quat.shape[1] == 4:
                # If explicitly labeled xyzw, convert to wxyz; otherwise assume wxyz.
                if root_quat_key and "xyzw" in root_quat_key:
                    self.motion_root_quat_w = xyzw_to_wxyz(root_quat)
                else:
                    self.motion_root_quat_w = root_quat
            else:
                logger.warning(f"Invalid root_quat shape {root_quat.shape}; ignoring motion root orientation.")

        # Apply reference speed resampling before alignment and velocity calculations
        reference_speed = getattr(self.config.task, "reference_speed", 1.0)
        if reference_speed != 1.0:
            self._resample_motion_arrays(reference_speed)
        self._validate_motion_timestep_config()

        # Preserve original world-frame object poses before alignment (used by RViz traj sender)
        if self.motion_obj_pos is not None:
            self._motion_obj_pos_world = self.motion_obj_pos.copy()
        if self.motion_obj_rot is not None:
            self._motion_obj_rot_world_wxyz = self.motion_obj_rot.copy()

        # Align motion object poses to the same root transform applied to the robot (xy translation + yaw)
        self._align_motion_objects_to_root(
            root_pos_w=self.motion_root_pos_w,
            root_quat_wxyz=self.motion_root_quat_w,
        )
        self._align_motion_table_to_root(
            root_pos_w=self.motion_root_pos_w,
            root_quat_wxyz=self.motion_root_quat_w,
        )
        self._last_sent_table_frame = None
        self._table_pose_send_warned = False
        
        # Calculate velocities using finite differences
        # vel[t] = (pos[t+1] - pos[t]) * fps
        dof_pos_next = np.roll(self.motion_dof_pos, -1, axis=0)
        dof_pos_next[-1] = self.motion_dof_pos[-1]  # Last frame has zero velocity
        self.motion_dof_vel = (dof_pos_next - self.motion_dof_pos) * self.motion_fps
        self.motion_dof_vel[-1] = 0.0  # Ensure last frame has zero velocity
        
        # Create joint reordering map from motion clip order to robot order
        # Motion clip order: 29-DOF (G1_JOINT_NAMES) or 43-DOF (G1_DEX3_JOINT_NAMES)
        if self.motion_dof_pos.shape[1] == 43:
            motion_joint_names = list(G1_DEX3_JOINT_NAMES)
        else:
            motion_joint_names = [
                "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
                "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
                "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
                "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
                "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
                "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
                "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
                "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
            ]
        self.motion_joint_names = motion_joint_names

        # Map from G1_JOINT_NAMES (motion order) to asset order for 29-DOF motions
        # Used when converting 29-DOF motion frames into asset/config order.
        self.motion_to_asset_joint_map = np.array(G1_TO_ASSET_ORDER_29DOF)

        # Cache body-only indices for motion command sequence (29-DOF in motion order)
        try:
            self.motion_body_indices = np.array(
                get_index_of_a_in_b(list(G1_MOTION_JOINT_NAMES_29), motion_joint_names), dtype=np.int64
            )
        except AssertionError as exc:
            logger.warning(f"Failed to build motion body indices: {exc}. Falling back to first 29 joints.")
            self.motion_body_indices = np.arange(min(29, self.motion_dof_pos.shape[1]), dtype=np.int64)
        
        self.motion_timestep = self._get_motion_start_frame_index()

    def _align_motion_objects_to_root(
        self,
        root_pos_w: np.ndarray | None,
        root_quat_wxyz: np.ndarray | None,
    ) -> None:
        """Align motion object poses using root xy translation and yaw from the start frame.

        Expects motion object rotations in wxyz and keeps them in wxyz after alignment.
        """
        if self.motion_obj_pos is None or self.motion_obj_rot is None:
            return
        if root_pos_w is None or root_quat_wxyz is None:
            return
        if root_pos_w.shape[0] == 0 or root_quat_wxyz.shape[0] == 0:
            return

        start_idx = self._get_motion_start_frame_index()
        root_pos0 = np.asarray(root_pos_w[start_idx], dtype=np.float32).reshape(3,)
        root_quat0_wxyz = np.asarray(root_quat_wxyz[start_idx], dtype=np.float32).reshape(1, 4)
        _, _, yaw = quat_to_rpy(root_quat0_wxyz.reshape(-1))
        yaw_quat_wxyz = rpy_to_quat((0.0, 0.0, yaw)).reshape(1, 4)

        # Position: subtract xy translation, keep z, then remove yaw
        obj_pos_w = np.asarray(self.motion_obj_pos, dtype=np.float32)
        trans_xy = np.array([root_pos0[0], root_pos0[1], 0.0], dtype=np.float32)
        rel_w = obj_pos_w - trans_xy
        q_pos = np.repeat(yaw_quat_wxyz, rel_w.shape[0], axis=0)
        rel_h = quat_rotate_inverse(q_pos, rel_w)

        # Rotation: remove yaw from object orientation (wxyz)
        obj_rot_wxyz = np.asarray(self.motion_obj_rot, dtype=np.float32)
        q_w2h = quat_inverse(yaw_quat_wxyz)
        q_w2h_rep = np.repeat(q_w2h, obj_rot_wxyz.shape[0], axis=0)
        obj_rot_h_wxyz = quat_mul(q_w2h_rep, obj_rot_wxyz)
        obj_rot_h_wxyz = self._normalize_quat_wxyz(obj_rot_h_wxyz)

        self.motion_obj_pos = rel_h.astype(np.float32, copy=False)
        self.motion_obj_rot = obj_rot_h_wxyz.astype(np.float32, copy=False)
        self.motion_obj_pos_aligned = self.motion_obj_pos.copy()
        self.motion_obj_rot_aligned_wxyz = obj_rot_h_wxyz.copy()
        logger.info(f"Aligned motion object poses to root xy/yaw of start frame {start_idx}.")

    def _align_motion_table_to_root(
        self,
        root_pos_w: np.ndarray | None,
        root_quat_wxyz: np.ndarray | None,
    ) -> None:
        """Align motion table poses using root xy translation and yaw from the start frame."""
        if self.motion_table_pos is None:
            return
        if root_pos_w is None or root_quat_wxyz is None:
            return
        if root_pos_w.shape[0] == 0 or root_quat_wxyz.shape[0] == 0:
            return

        start_idx = self._get_motion_start_frame_index()
        root_pos0 = np.asarray(root_pos_w[start_idx], dtype=np.float32).reshape(3,)
        root_quat0_wxyz = np.asarray(root_quat_wxyz[start_idx], dtype=np.float32).reshape(1, 4)
        _, _, yaw = quat_to_rpy(root_quat0_wxyz.reshape(-1))
        yaw_quat_wxyz = rpy_to_quat((0.0, 0.0, yaw)).reshape(1, 4)

        table_pos_w = np.asarray(self.motion_table_pos, dtype=np.float32)
        trans_xy = np.array([root_pos0[0], root_pos0[1], 0.0], dtype=np.float32)
        rel_w = table_pos_w - trans_xy
        q_pos = np.repeat(yaw_quat_wxyz, rel_w.shape[0], axis=0)
        rel_h = quat_rotate_inverse(q_pos, rel_w)
        self.motion_table_pos = rel_h.astype(np.float32, copy=False)
        self.motion_table_pos_aligned = self.motion_table_pos.copy()

        if self.motion_table_rot_wxyz is not None:
            table_rot_wxyz = np.asarray(self.motion_table_rot_wxyz, dtype=np.float32)
            q_w2h = quat_inverse(yaw_quat_wxyz)
            q_w2h_rep = np.repeat(q_w2h, table_rot_wxyz.shape[0], axis=0)
            table_rot_h_wxyz = quat_mul(q_w2h_rep, table_rot_wxyz)
            table_rot_h_wxyz = self._normalize_quat_wxyz(table_rot_h_wxyz)
            self.motion_table_rot_wxyz = table_rot_h_wxyz.astype(np.float32, copy=False)
            self.motion_table_rot_aligned_wxyz = table_rot_h_wxyz.copy()

        logger.info(f"Aligned motion table poses to root xy/yaw of start frame {start_idx}.")

    def _convert_motion_root_to_torso(self) -> None:
        if self.motion_root_pos_w is None or self.motion_root_quat_w is None:
            return
        if self.motion_dof_pos is None or self.motion_length == 0:
            return
        if not hasattr(self, "pinocchio_robot") or self.pinocchio_robot is None:
            return
        motion_joint_names = getattr(self, "motion_joint_names", None)
        if not motion_joint_names:
            return

        num_frames = self.motion_dof_pos.shape[0]
        default_q = self.default_dof_angles.reshape(1, -1).astype(np.float32)
        dof_pos_config = np.repeat(default_q, num_frames, axis=0)
        motion_to_config = np.array(get_index_of_a_in_b(motion_joint_names, self.dof_names), dtype=np.int64)
        dof_pos_config[:, motion_to_config] = self.motion_dof_pos.astype(np.float32)

        torso_pos = np.zeros_like(self.motion_root_pos_w, dtype=np.float32)
        torso_quat = np.zeros_like(self.motion_root_quat_w, dtype=np.float32)

        for i in range(num_frames):
            pos, quat = self.pinocchio_robot.fk_and_get_ref_body_pose_in_world(
                root_pos=self.motion_root_pos_w[i],
                root_quat_wxyz=self.motion_root_quat_w[i],
                dof_pos_real=dof_pos_config[i],
            )
            torso_pos[i] = pos
            torso_quat[i] = quat

        self.motion_root_pos_w = torso_pos
        self.motion_root_quat_w = torso_quat
        logger.info("Converted motion root pose from pelvis to torso_link frame using FK.")

    def _get_motion_command_sequence(self, timestep: int) -> np.ndarray:
        """Get 10-frame motion sequence starting from timestep.
        
        Returns:
            np.ndarray: [1, 580] - always 10 frames × 58 dims (29-DOF motion only)
                       Motion clips are 29-DOF, kept in G1_JOINT_NAMES order (motion command order)
        """
        # Get 10 future timesteps, clamping to motion length
        future_steps = np.arange(timestep, timestep + 10)
        future_steps = np.clip(future_steps, 0, self.motion_length - 1)
        
        # Fetch positions and velocities for 10 frames in motion clip order
        try:
            pos_seq = self.motion_dof_pos[future_steps]
        except:
            breakpoint()
        vel_seq = self.motion_dof_vel[future_steps]

        # Always use body-only 29-DOF for motion commands (drop hands if present)
        if pos_seq.shape[1] != self.motion_command_dofs:
            if self.motion_body_indices is not None:
                pos_seq = pos_seq[:, self.motion_body_indices]
                vel_seq = vel_seq[:, self.motion_body_indices]
            else:
                pos_seq = pos_seq[:, : self.motion_command_dofs]
                vel_seq = vel_seq[:, : self.motion_command_dofs]

        # Keep in motion clip order (G1_JOINT_NAMES) - this is the motion command order the model expects
        # Do NOT reorder to asset order here - motion commands are separate from proprioception
        
        # NOTE: Motion clips are always 29-DOF (body only, no hands)
        # Do NOT pad for 43-DOF robots - hand joints have no motion data
        
        # Concatenate pos and vel: [10, 58] in G1_JOINT_NAMES order (motion command order)
        command_seq = np.concatenate([pos_seq, vel_seq], axis=1)
        
        # Flatten to [1, 580]
        return command_seq.reshape(1, -1)

    def _get_motion_command_sequence_first_frame(self) -> np.ndarray:
        """Get a 10-frame sequence repeating the configured start motion frame (body-only)."""
        if self.motion_dof_pos is None or self.motion_length == 0:
            return np.zeros((1, 10 * self.motion_command_dofs * 2), dtype=np.float32)

        start_idx = self._get_motion_start_frame_index()
        pos0 = self.motion_dof_pos[start_idx]
        vel0 = self.motion_dof_vel[start_idx] if self.motion_dof_vel is not None else np.zeros_like(pos0)

        if pos0.shape[0] != self.motion_command_dofs:
            if self.motion_body_indices is not None:
                pos0 = pos0[self.motion_body_indices]
                vel0 = vel0[self.motion_body_indices]
            else:
                pos0 = pos0[: self.motion_command_dofs]
                vel0 = vel0[: self.motion_command_dofs]

        frame = np.concatenate([pos0, vel0], axis=0).reshape(1, -1)  # [1, 58]
        seq = np.repeat(frame, 10, axis=0)  # [10, 58]
        return seq.reshape(1, -1).astype(np.float32, copy=False)
    
    def _get_motion_command_sequence_first_pos(self) -> np.ndarray:
        """Get a 10-frame sequence repeating the configured start motion position."""
        if self.motion_dof_pos is None or self.motion_length == 0:
            return np.zeros((1, 10 * self.motion_command_dofs * 2), dtype=np.float32)

        start_idx = self._get_motion_start_frame_index()
        pos0 = self.motion_dof_pos[start_idx].copy()
        pos0[:22] = self.startup_pos[:22]
        pos0[29:36] = self.startup_pos[29:36]
        vel0 = self.motion_dof_vel[start_idx] if self.motion_dof_vel is not None else np.zeros_like(pos0)

        if pos0.shape[0] != self.motion_command_dofs:
            if self.motion_body_indices is not None:
                pos0 = pos0[self.motion_body_indices]
                vel0 = vel0[self.motion_body_indices]
            else:
                pos0 = pos0[: self.motion_command_dofs]
                vel0 = vel0[: self.motion_command_dofs]

        frame = np.concatenate([pos0, vel0], axis=0).reshape(1, -1)  # [1, 58]
        seq = np.repeat(frame, 10, axis=0)  # [10, 58]
        return seq.reshape(1, -1).astype(np.float32, copy=False)
    
    def _get_ref_body_orientation_in_world(self, robot_state_data):
        # Create configuration for pinocchio robot
        # Note:
        # 1. pinocchio quaternion is in xyzw format, robot_state_data is in wxyz format
        # 2. joint sequences in pinocchio robot and real robot are different

        # free base pos, does not matter
        root_pos = robot_state_data[0, :3]

        # free base ori, wxyz -> xyzw
        root_ori_xyzw = wxyz_to_xyzw(robot_state_data[:, 3:7])[0]

        # dof pos in real robot -> pinocchio robot
        num_dofs = self.num_dofs
        dof_pos_in_real = robot_state_data[0, 7 : 7 + num_dofs]
        dof_pos_in_pinocchio = dof_pos_in_real[self.pinocchio_robot.real2pinocchio_index]

        configuration = np.concatenate([root_pos, root_ori_xyzw, dof_pos_in_pinocchio], axis=0)

        ref_ori_xyzw = self.pinocchio_robot.fk_and_get_ref_body_orientation_in_world(configuration)
        return xyzw_to_wxyz(ref_ori_xyzw)

    def _get_ref_body_pose_in_world(self, robot_state_data) -> tuple[np.ndarray, np.ndarray]:
        root_pos = robot_state_data[0, :3]
        root_quat_wxyz = robot_state_data[0, 3:7]

        dof_pos_in_real = robot_state_data[0, 7 : 7 + self.num_dofs]
        if hasattr(self, "pinocchio_robot") and self.pinocchio_robot is not None:
            pos, quat = self.pinocchio_robot.fk_and_get_ref_body_pose_in_world(
                root_pos=root_pos,
                root_quat_wxyz=root_quat_wxyz,
                dof_pos_real=dof_pos_in_real,
            )
            return pos.reshape(1, 3), quat.reshape(1, 4)

        return root_pos.reshape(1, 3), root_quat_wxyz.reshape(1, 4)

    def _log_robot_object_pose_snapshot(self, tag: str, robot_state_data=None) -> None:
        """Log robot/object world poses for quick runtime verification."""
        if robot_state_data is None:
            robot_state_data = (
                self.interface.get_full_state_43dof() if self.num_dofs == 43 else self.interface.get_low_state()
            )

        if robot_state_data is None:
            self.logger.warning(colored(f"[{tag}] robot_state_data unavailable", "yellow"))
            return

        # Robot root pose from low/full state
        root_pos_w = np.asarray(robot_state_data[:, :3], dtype=np.float32).reshape(1, 3)
        root_quat_wxyz = np.asarray(robot_state_data[:, 3:7], dtype=np.float32).reshape(1, 4)

        # Reference body (torso) pose used by object-relative observations
        try:
            ref_pos_w, ref_quat_wxyz = self._get_ref_body_pose_in_world(robot_state_data)
        except Exception as exc:
            ref_pos_w, ref_quat_wxyz = None, None
            self.logger.warning(colored(f"[{tag}] failed to compute ref body pose: {exc}", "yellow"))

        obj_pos_w = self._get_object_pos_w()
        obj_quat_wxyz = self._get_object_quat_w()

        self.logger.info(
            colored(
                f"[{tag}] root_pos_w={root_pos_w.flatten().tolist()} "
                f"root_quat_wxyz={root_quat_wxyz.flatten().tolist()}",
                "cyan",
            )
        )

        if ref_pos_w is not None and ref_quat_wxyz is not None:
            self.logger.info(
                colored(
                    f"[{tag}] ref_pos_w={ref_pos_w.flatten().tolist()} "
                    f"ref_quat_wxyz={ref_quat_wxyz.flatten().tolist()}",
                    "cyan",
                )
            )

        self.logger.info(
            colored(
                f"[{tag}] object_pos_w={None if obj_pos_w is None else obj_pos_w.flatten().tolist()} "
                f"object_quat_wxyz={None if obj_quat_wxyz is None else obj_quat_wxyz.flatten().tolist()}",
                "magenta",
            )
        )

    def setup_policy(self, model_path):
        self.onnx_policy_session = onnxruntime.InferenceSession(model_path)
        self.onnx_input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        self.onnx_output_names = [out.name for out in self.onnx_policy_session.get_outputs()]

        # Extract KP/KD from ONNX metadata (same as base class)
        onnx_model = onnx.load(model_path)
        metadata = {}
        for prop in onnx_model.metadata_props:
            metadata[prop.key] = json.loads(prop.value)

        holosoma_root = Path(__file__).parent.parent.parent.parent.parent

        # Extract URDF text from ONNX metadata or load from config
        if "robot_urdf" in metadata:
            urdf_text = metadata["robot_urdf"]
            logger.info("Loaded URDF from ONNX metadata")
        else:
            # Fallback: construct URDF path from robot type
            # Construct path based on robot type: src/holosoma/holosoma/data/robots/{robot}/{robot_type}.urdf
            robot_name = self.config.robot.robot  # e.g., "g1"
            robot_type = self.config.robot.robot_type  # e.g., "g1_43dof"
            urdf_path = holosoma_root / "src" / "holosoma" / "holosoma" / "data" / "robots" / robot_name / f"{robot_type}.urdf"
            
            if urdf_path.exists():
                urdf_text = urdf_path.read_text()
                logger.info(f"Loaded URDF from: {urdf_path}")
            else:
                raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        
        self.pinocchio_robot = PinocchioRobot(self.config.robot, urdf_text)

        self.onnx_kp = np.array(metadata["kp"]) if "kp" in metadata else None
        self.onnx_kd = np.array(metadata["kd"]) if "kd" in metadata else None

        if self.onnx_kp is not None:
            logger.info(f"Loaded KP/KD from ONNX metadata: {Path(model_path).name}")
        else:
            # Use training KP/KD values (in config order - 43 DOF)
            # Config order: [L_leg(6), R_leg(6), waist(3), L_arm(4), L_hand(7), R_arm(4), R_hand(7)]
            # self.onnx_kp = np.array([
            #     # Left leg (6 joints): hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
            #     28.5012, 28.5012, 28.5012, 28.5012, 28.5012, 28.5012,
            #     # Right leg (6 joints): hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
            #     28.5012, 28.5012, 28.5012, 28.5012, 28.5012, 28.5012,
            #     # Waist (3 joints): yaw, roll, pitch
            #     40.1792, 28.5012, 28.5012,
            #     # Left arm (4 joints): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
            #     14.2506, 14.2506, 14.2506, 14.2506,
            #     # Left wrist (3 joints): roll, pitch, yaw
            #     14.2506, 16.7783, 16.7783,
            #     # Left hand (7 joints): thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1
            #     2.0000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
            #     # Right arm (4 joints): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
            #     14.2506, 14.2506, 14.2506, 14.2506,
            #     # Right wrist (3 joints): roll, pitch, yaw
            #     14.2506, 16.7783, 16.7783,
            #     # Right hand (7 joints): thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1
            #     2.0000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
            # ], dtype=np.float32)

            self.onnx_kp = np.array([
                # Left leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
                40.17923737, 99.09842682, 40.17923737, 99.09842682, 28.50124550, 28.50124550,
                # Right leg
                40.17923737, 99.09842682, 40.17923737, 99.09842682, 28.50124550, 28.50124550,
                # Waist: yaw, roll, pitch
                40.17923737, 28.50124550, 28.50124550,
                # Left arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
                14.25062275, 14.25062275, 14.25062275, 14.25062275,
                # Left wrist: roll, pitch, yaw
                14.25062275, 16.77832794, 16.77832794,
                # Left hand
                2.0000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                # Right arm
                14.25062275, 14.25062275, 14.25062275, 14.25062275,
                # Right wrist
                14.25062275, 16.77832794, 16.77832794,
                # Right hand
                2.0000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
            ], dtype=np.float32)

            self.onnx_kd = np.array([
                # Left leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
                2.55788970, 6.30880185, 2.55788970, 6.30880185, 1.81444573, 1.81444573,
                # Right leg
                2.55788970, 6.30880185, 2.55788970, 6.30880185, 1.81444573, 1.81444573,
                # Waist: yaw, roll, pitch
                2.55788970, 1.81444573, 1.81444573,
                # Left arm
                0.90722287, 0.90722287, 0.90722287, 0.90722287,
                # Left wrist: roll, pitch, yaw
                0.90722287, 1.06814146, 1.06814146,
                # Left hand
                0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000,
                # Right arm
                0.90722287, 0.90722287, 0.90722287, 0.90722287,
                # Right wrist
                0.90722287, 1.06814146, 1.06814146,
                # Right hand
                0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000,
            ], dtype=np.float32)

            # self.onnx_kd = np.array([
            #     # Left leg (6 joints)
            #     1.8144, 1.8144, 1.8144, 1.8144, 1.8144, 1.8144,
            #     # Right leg (6 joints)
            #     1.8144, 1.8144, 1.8144, 1.8144, 1.8144, 1.8144,
            #     # Waist (3 joints): yaw, roll, pitch
            #     2.5579, 1.8144, 1.8144,
            #     # Left arm (4 joints)
            #     0.9072, 0.9072, 0.9072, 0.9072,
            #     # Left wrist (3 joints): roll, pitch, yaw
            #     0.9072, 1.0681, 1.0681,
            #     # Left hand (7 joints)
            #     0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000,
            #     # Right arm (4 joints)
            #     0.9072, 0.9072, 0.9072, 0.9072,
            #     # Right wrist (3 joints): roll, pitch, yaw
            #     0.9072, 1.0681, 1.0681,
            #     # Right hand (7 joints)
            #     0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000,
            # ], dtype=np.float32)
            
            logger.info(colored("Using training KP/KD values (43-DOF config order)", "cyan"))

        # Load motion data from PKL file
        # Prefer CLI task config overrides, then fall back to bundled defaults.
        model_dir = Path(model_path).parent
        holosoma_root = Path(__file__).parent.parent.parent.parent.parent  # Go up to holosoma root

        config_motion_pkl_path = getattr(self.config.task, "motion_pkl_path", None)
        config_motion_clip_key = getattr(self.config.task, "motion_clip_key", None)

        if config_motion_pkl_path:
            pkl_path = Path(config_motion_pkl_path)
            if not pkl_path.exists():
                raise FileNotFoundError(f"Configured motion PKL path does not exist: {pkl_path}")
            pkl_search_paths = [pkl_path]
        else:
            pkl_search_paths = [
                model_dir / "simple_8_motions.pkl",  # Same directory as model
                holosoma_root / "src/holosoma/holosoma/data/motions/motion_tracking/cubemediums_12_0116_with_text_traj.pkl",  # Training data location
            ]

            pkl_path = None
            for path in pkl_search_paths:
                if path.exists():
                    pkl_path = path
                    break

        self._motion_pkl_path = str(pkl_path) if pkl_path is not None else None

        if pkl_path:
            clip_key = config_motion_clip_key or "GRAB_s1_cubemedium_pass_1"
            self._load_motion_from_pkl(str(pkl_path), clip_key=clip_key)
            logger.info(colored(f"📊 Motion Clip Summary:", "cyan", attrs=["bold"]))
            logger.info(colored(f"   Clip Key: {self.motion_clip_key}", "yellow"))
            logger.info(colored(f"   Motion Length: {self.motion_length} frames", "yellow", attrs=["bold"]))
            logger.info(colored(f"   Duration: {self.motion_length / self.motion_fps:.2f} seconds", "yellow"))
            logger.info(colored(f"   FPS: {self.motion_fps}", "yellow"))
        else:
            logger.warning(f"Motion PKL file not found. Searched: {[str(p) for p in pkl_search_paths]}")

        # Policy function - simplified since motion is loaded separately
        def policy_act(input_feed):
            output = self.onnx_policy_session.run(["actions"], input_feed)
            action = output[0]
            return action

        self.policy = policy_act
        self._init_stabilization_policy(holosoma_root)

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
        self.motion_timestep = state.get("motion_timestep", self._get_configured_motion_start_timestep())
        self.motion_clip_key = state.get("motion_clip_key", None)
        self.motion_clip_progressing = False
        self.stabilization_mode = False
        self.motion_start_timestep = None
        self._last_clock_reading = None
        self.robot_yaw_offset = 0.0

    @staticmethod
    def _normalize_object_name(obj_name: str | None) -> str | None:
        if obj_name is None:
            return None
        normalized = str(obj_name).strip()
        return normalized or None

    def _apply_object_identity(self, obj_name: str | None, source: str) -> None:
        obj_name = self._normalize_object_name(obj_name)
        if obj_name is None:
            return
        previous_name = self.motion_obj_name
        self.motion_obj_name = obj_name
        if "obj_bps" in self.obs_dims:
            self._obj_bps = self._load_bps_code(obj_name)
        if obj_name != previous_name:
            logger.info(colored(f"Scene object set to '{obj_name}' via {source}", "cyan"))

    def _find_clip_key_for_object(self, obj_name: str | None) -> str | None:
        obj_name = self._normalize_object_name(obj_name)
        if obj_name is None or self.motion_data is None:
            return None

        for clip_key, clip_data in self.motion_data.items():
            clip_obj_name = clip_data.get("obj_name", None)
            if isinstance(clip_obj_name, (list, np.ndarray)):
                clip_obj_array = np.asarray(clip_obj_name).reshape(-1)
                clip_obj_name = clip_obj_array[0] if clip_obj_array.size > 0 else None
            clip_obj_name = self._normalize_object_name(clip_obj_name)
            if clip_obj_name == obj_name:
                return clip_key

        for clip_key in self.motion_data.keys():
            if obj_name in str(clip_key):
                return clip_key
        return None

    def _sync_scene_object_name(self) -> None:
        if not hasattr(self.interface, "get_scene_object_name"):
            return

        if hasattr(self.interface, "get_scene_reset_counter"):
            scene_reset_counter = self.interface.get_scene_reset_counter()
            if scene_reset_counter is not None and scene_reset_counter != self._scene_reset_counter:
                self._scene_reset_counter = scene_reset_counter
                if self._table_removed or self._initial_object_height is not None:
                    self.logger.info(
                        "Scene reset detected from simulator reset counter; restoring table-tracking state."
                    )
                self._table_removed = False
                self._initial_object_height = None
                self._last_sent_table_frame = None
                self._table_removal_grace_steps = 0

        scene_object_name = self._normalize_object_name(self.interface.get_scene_object_name())
        if scene_object_name is None:
            return

        self._scene_object_name = scene_object_name
        can_switch_clip = (
            self._motion_pkl_path is not None
            and not self.motion_clip_progressing
            and not self.stabilization_mode
        )
        if can_switch_clip:
            matching_clip_key = self._find_clip_key_for_object(scene_object_name)
            if matching_clip_key is not None and matching_clip_key != self.motion_clip_key:
                self._load_motion_from_pkl(self._motion_pkl_path, clip_key=matching_clip_key)
                logger.info(
                    colored(
                        f"Auto-switched motion clip to '{matching_clip_key}' for scene object '{scene_object_name}'",
                        "cyan",
                    )
                )

        if self.motion_obj_name != scene_object_name:
            self._apply_object_identity(scene_object_name, source="scene UDP")

    def _on_policy_switched(self, model_path: str):
        super()._on_policy_switched(model_path)
        self.motion_clip_progressing = False
        self.stabilization_mode = False
        self._reset_motion_timing()
        self._stiff_hold_active = True
        self.robot_yaw_offset = 0.0
        self._prev_robot_anchor_quat_w = None
        self._prev_robot_obj_pos_w = None

    def get_init_target(self, robot_state_data):
        """Get initialization target joint positions."""
        dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
        if self.get_ready_state:
            # Interpolate from current dof_pos to first pose in motion
            if self.motion_data is not None:
                # Get configured start frame from motion data.
                target_pos_motion_order = self.motion_dof_pos[self._get_motion_start_frame_index()]

                if self.num_dofs == 43 and target_pos_motion_order.shape[0] == 43:
                    # Motion clip is 43-DOF in G1_DEX3_JOINT_NAMES order; map directly to config order.
                    motion_to_config = np.array(
                        get_index_of_a_in_b(self.dof_names, list(G1_DEX3_JOINT_NAMES)), dtype=np.int64
                    )
                    target_pos_config = target_pos_motion_order[motion_to_config]
                else:
                    # Fallback to existing 29-DOF motion handling (G1_JOINT_NAMES order)
                    target_pos_asset = target_pos_motion_order[self.motion_to_asset_joint_map]  # [29] in asset order
                    if self.num_dofs == 43:
                        # For 43-DOF: reorder body joints and insert hand defaults at correct positions
                        reorder_map = np.array(ASSET_TO_CONFIG_ORDER_29DOF)
                        target_pos_config_29 = target_pos_asset[reorder_map]  # [29] in 29-DOF config order

                        # Hand default values (config order)
                        left_hand_defaults = np.array(
                            [0.0, 1.0, 1.7, -1.57, -1.7, -1.57, -1.7], dtype=np.float32
                        )
                        right_hand_defaults = np.array(
                            [0.0, -1.0, -1.7, 1.57, 1.7, 1.57, 1.7], dtype=np.float32
                        )

                        # Interleave hands in correct positions
                        target_pos_config = np.concatenate(
                            [
                                target_pos_config_29[:22],  # left_leg + right_leg + waist + left_arm + left_wrist
                                left_hand_defaults,  # left hand (7 joints)
                                target_pos_config_29[22:],  # right_arm + right_wrist (7 joints)
                                right_hand_defaults,  # right hand (7 joints)
                            ]
                        )
                    else:
                        # For 29-DOF: just reorder
                        reorder_map = np.array(ASSET_TO_CONFIG_ORDER_29DOF)
                        target_pos_config = target_pos_asset[reorder_map]  # [29] in config order

                target_dof_pos = target_pos_config.reshape(1, -1)
            else:
                # Fallback: stay at current position
                target_dof_pos = dof_pos

            q_target = dof_pos + (target_dof_pos - dof_pos) * (self.init_count / 500)
            self.init_count += 1
            return q_target
        return dof_pos

    def _get_action_order_mappings(self) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "_action_order_to_config"):
            self._action_order_to_config = np.array(
                get_index_of_a_in_b(list(ACTION_ORDER_43DOF), self.dof_names), dtype=np.int64
            )
            self._config_to_action_order = np.argsort(self._action_order_to_config)
        return self._action_order_to_config, self._config_to_action_order

    def get_current_obs_buffer_dict(self, robot_state_data):

        current_obs_buffer_dict = {}
        self._sync_scene_object_name()
        command_timestep = self._get_motion_start_frame_index() if self.stabilization_mode else self.motion_timestep
        self._send_debug_traj_viz(command_timestep)

        # Extract joint data from robot_state_data (in config order)
        dof_pos_config = robot_state_data[:, 7 : 7 + self.num_dofs]  # [1, num_dofs]
        dof_vel_config = robot_state_data[
            :, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs
        ]  # [1, num_dofs]

        if self.num_dofs == 43:
            if not hasattr(self, "_proprio_hand_indices"):
                hand_joint_names = [
                    "left_hand_thumb_0_joint",
                    "left_hand_thumb_1_joint",
                    "left_hand_thumb_2_joint",
                    "left_hand_middle_0_joint",
                    "left_hand_middle_1_joint",
                    "left_hand_index_0_joint",
                    "left_hand_index_1_joint",
                    "right_hand_thumb_0_joint",
                    "right_hand_thumb_1_joint",
                    "right_hand_thumb_2_joint",
                    "right_hand_middle_0_joint",
                    "right_hand_middle_1_joint",
                    "right_hand_index_0_joint",
                    "right_hand_index_1_joint",
                ]
                body_joint_names = [
                    "waist_yaw_joint",
                    "waist_roll_joint",
                    "waist_pitch_joint",
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
                self._proprio_hand_indices = get_index_of_a_in_b(hand_joint_names, self.dof_names)
                self._proprio_body_indices = get_index_of_a_in_b(body_joint_names, self.dof_names)

            hand_idx = self._proprio_hand_indices
            body_idx = self._proprio_body_indices

            dof_pos_rel = dof_pos_config - self.default_dof_angles.reshape(1, -1)
            current_obs_buffer_dict["joint_pos_hand"] = dof_pos_rel[:, hand_idx]
            current_obs_buffer_dict["joint_vel_hand"] = dof_vel_config[:, hand_idx]
            current_obs_buffer_dict["joint_pos_body"] = dof_pos_rel[:, body_idx]
            current_obs_buffer_dict["joint_vel_body"] = dof_vel_config[:, body_idx]

            current_obs_buffer_dict["base_lin_vel"] = robot_state_data[:, 7 + self.num_dofs : 7 + self.num_dofs + 3]
            current_obs_buffer_dict["base_ang_vel"] = robot_state_data[
                :, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6
            ]

            base_quat_wxyz = robot_state_data[:, 3:7]
            gravity_world = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
            current_obs_buffer_dict["projected_gravity"] = quat_rotate_inverse(base_quat_wxyz, gravity_world)
        else:
            reorder_map = np.array(CONFIG_TO_ASSET_ORDER_29DOF)
            dof_pos_asset = dof_pos_config[:, reorder_map]
            dof_vel_asset = dof_vel_config[:, reorder_map]
            default_dof_angles_asset = self.default_dof_angles[reorder_map]

            current_obs_buffer_dict["dof_pos"] = dof_pos_asset - default_dof_angles_asset
            current_obs_buffer_dict["dof_vel"] = dof_vel_asset

            current_obs_buffer_dict["base_ang_vel"] = robot_state_data[
                :, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6
            ]
            base_quat_wxyz = robot_state_data[:, 3:7]
            gravity_world = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
            current_obs_buffer_dict["projected_gravity"] = quat_rotate_inverse(base_quat_wxyz, gravity_world)

        # obj_pos_diff_b: delta object pos in previous robot frame (torso-based)
        obj_pos_diff_b = self._compute_obj_pos_diff_b(robot_state_data)
        current_obs_buffer_dict["obj_pos_diff_b"] = obj_pos_diff_b

        # BPS policy observation terms (used by g1-43dof-object-bps)
        _bps_yaw_quat = None
        if "obj_pos_rel" in self.obs_dims or "obj_rot_6d" in self.obs_dims:
            _bps_obj_pos, _bps_obj_rot_6d, _bps_yaw_quat = self._compute_bps_obj_state(robot_state_data)
            current_obs_buffer_dict["obj_pos_rel"] = _bps_obj_pos
            current_obs_buffer_dict["obj_rot_6d"] = _bps_obj_rot_6d
        if "obj_bps" in self.obs_dims:
            if self._obj_bps is not None:
                current_obs_buffer_dict["obj_bps"] = self._obj_bps.reshape(1, -1)
            else:
                current_obs_buffer_dict["obj_bps"] = np.zeros((1, 512), dtype=np.float32)
        if "motion_obj_pos_short" in self.obs_dims or "motion_obj_ori_short" in self.obs_dims:
            pos_short, ori_short, pos_long, ori_long = self._compute_bps_motion_commands(
                robot_state_data, command_timestep, _bps_yaw_quat
            )
            current_obs_buffer_dict["motion_obj_pos_short"] = pos_short
            current_obs_buffer_dict["motion_obj_ori_short"] = ori_short
            current_obs_buffer_dict["motion_obj_pos_long"] = pos_long
            current_obs_buffer_dict["motion_obj_ori_long"] = ori_long

        # Motion object commands (reference motion only)
        if self.motion_data is not None and self.motion_obj_pos is not None and self.motion_obj_rot is not None:
            _motion_obj_pos_rel = self._compute_motion_obj_pos_rel_all(robot_state_data, command_timestep)
            _motion_obj_ori_rel = self._compute_motion_obj_ori_rel_all(robot_state_data, command_timestep)
            current_obs_buffer_dict["motion_obj_pos_rel_all"] = _motion_obj_pos_rel
            current_obs_buffer_dict["motion_obj_ori_rel_all"] = _motion_obj_ori_rel
        else:
            current_obs_buffer_dict["motion_obj_pos_rel_all"] = np.zeros((1, 42), dtype=np.float32)
            current_obs_buffer_dict["motion_obj_ori_rel_all"] = np.zeros((1, 84), dtype=np.float32)

        # Object pointcloud (current object pose in robot heading frame)
        obj_pcd = self._compute_obj_pcd(robot_state_data)
        current_obs_buffer_dict["obj_pcd"] = obj_pcd

        # Object pointcloud (motion command t+1 pose in robot heading frame)
        motion_obj_pcd = self._compute_motion_obj_pcd(
            robot_state_data, command_timestep if self.motion_data is not None else 0
        )
        current_obs_buffer_dict["motion_obj_pcd"] = motion_obj_pcd

        # actions
        current_obs_buffer_dict["actions"] = self.last_policy_action

        # Command: 10-frame motion sequence
        if self.motion_data is not None:
            # In stabilization mode, keep using the configured start frame for motion command.
            if self.stabilization_mode:
                current_obs_buffer_dict["motion_command_sequence"] = self._get_motion_command_sequence_first_frame()
            # else:
            #     current_obs_buffer_dict["motion_command_sequence"] = self._get_motion_command_sequence(command_timestep)
        else:
            # Fallback: use zeros if motion not loaded
            seq_dim = 10 * self.motion_command_dofs * 2  # 580 for 29-DOF (body-only)
            current_obs_buffer_dict["motion_command_sequence"] = np.zeros((1, seq_dim), dtype=np.float32)
            if not hasattr(self, '_motion_warning_shown'):
                logger.warning("Motion data not loaded, using zero motion command sequence")
                self._motion_warning_shown = True

        return current_obs_buffer_dict

    @staticmethod
    def _normalize_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
        quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32)
        norm = np.linalg.norm(quat_wxyz, axis=-1, keepdims=True)
        norm = np.where(norm == 0.0, 1.0, norm)
        return quat_wxyz / norm

    @staticmethod
    def _motion_short_indices(base_index: int) -> np.ndarray:
        # t+1..t+10 (inclusive) -> length 10
        return base_index + np.arange(0, 10, dtype=np.int64)

    @staticmethod
    def _motion_long_indices(base_index: int) -> np.ndarray:
        # t+20, t+40, t+60, t+80, t+100 -> base_index + [19,39,59,79,99]
        return base_index + np.array([19, 39, 59, 79, 99], dtype=np.int64)

    def _compute_motion_obj_pos_rel_all(self, robot_state_data, timestep: int) -> np.ndarray:
        if self.motion_obj_pos is None or self.motion_length == 0:
            return np.zeros((1, 42), dtype=np.float32)
        if self.motion_obj_pos.ndim != 2 or self.motion_obj_pos.shape[1] != 3:
            return np.zeros((1, 42), dtype=np.float32)

        base_index = int(np.clip(timestep + 1, 0, self.motion_length - 1))
        short_idx = self._motion_short_indices(base_index)
        long_idx = self._motion_long_indices(base_index)
        short_idx = np.clip(short_idx, 0, self.motion_length - 1)
        long_idx = np.clip(long_idx, 0, self.motion_length - 1)
        
        short_pos_w = self.motion_obj_pos[short_idx]  # (10, 3)
        long_pos_w = self.motion_obj_pos[long_idx]    # (5, 3)

        # Use current robot heading (yaw-only) frame
        ref_quat_wxyz = self._get_ref_body_orientation_in_world(robot_state_data)
        _, _, yaw = quat_to_rpy(ref_quat_wxyz.reshape(-1))
        yaw_quat = rpy_to_quat((0.0, 0.0, yaw)).reshape(1, 4)

        base_pos_w = short_pos_w[0].reshape(1, 3)  # t+1
        rel_s_w = short_pos_w[1:] - base_pos_w     # (9, 3)
        rel_l_w = long_pos_w - base_pos_w          # (5, 3)

        q_s = np.repeat(yaw_quat, rel_s_w.shape[0], axis=0)
        q_l = np.repeat(yaw_quat, rel_l_w.shape[0], axis=0)
        rel_s_h = quat_rotate_inverse(q_s, rel_s_w)
        rel_l_h = quat_rotate_inverse(q_l, rel_l_w)

        out = np.concatenate([rel_s_h, rel_l_h], axis=0)  # (14, 3)
        return out.reshape(1, -1).astype(np.float32, copy=False)

    def _update_table_pose_from_motion(self, timestep: int) -> None:
        """Policy inference must not modify simulator table pose outside reset."""
        return

    def _compute_motion_obj_ori_rel_all(self, robot_state_data, timestep: int) -> np.ndarray:
        if self.motion_obj_rot is None or self.motion_length == 0:
            return np.zeros((1, 84), dtype=np.float32)
        if self.motion_obj_rot.ndim != 2 or self.motion_obj_rot.shape[1] != 4:
            return np.zeros((1, 84), dtype=np.float32)

        base_index = int(np.clip(timestep + 1, 0, self.motion_length - 1))
        short_idx = self._motion_short_indices(base_index)
        long_idx = self._motion_long_indices(base_index)
        short_idx = np.clip(short_idx, 0, self.motion_length - 1)
        long_idx = np.clip(long_idx, 0, self.motion_length - 1)

        short_rot_wxyz = self._normalize_quat_wxyz(self.motion_obj_rot[short_idx])  
        long_rot_wxyz = self._normalize_quat_wxyz(self.motion_obj_rot[long_idx])    
        # Use current robot heading (yaw-only) frame
        ref_quat_wxyz = self._get_ref_body_orientation_in_world(robot_state_data)
        _, _, yaw = quat_to_rpy(ref_quat_wxyz.reshape(-1))
        yaw_quat = rpy_to_quat((0.0, 0.0, yaw)).reshape(1, 4)
        q_w2h = quat_inverse(yaw_quat)

        # base orientation (t+1) in heading frame
        q0_h = quat_mul(q_w2h, short_rot_wxyz[:1])
        q0_h = self._normalize_quat_wxyz(q0_h)
        q0_h_inv = quat_inverse(q0_h)

        # short horizon: relative to base, skip t+1
        q_w2h_s = np.repeat(q_w2h, short_rot_wxyz.shape[0] - 1, axis=0)
        qt_h_s = quat_mul(q_w2h_s, short_rot_wxyz[1:])
        qt_h_s = self._normalize_quat_wxyz(qt_h_s)
        q0_h_inv_s = np.repeat(q0_h_inv, qt_h_s.shape[0], axis=0)
        dq_s = quat_mul(q0_h_inv_s, qt_h_s)
        dq_s = self._normalize_quat_wxyz(dq_s)

        # long horizon: relative to base, include all
        q_w2h_l = np.repeat(q_w2h, long_rot_wxyz.shape[0], axis=0)
        qt_h_l = quat_mul(q_w2h_l, long_rot_wxyz)
        qt_h_l = self._normalize_quat_wxyz(qt_h_l)
        q0_h_inv_l = np.repeat(q0_h_inv, qt_h_l.shape[0], axis=0)
        dq_l = quat_mul(q0_h_inv_l, qt_h_l)
        dq_l = self._normalize_quat_wxyz(dq_l)

        dq = np.concatenate([dq_s, dq_l], axis=0)  # (14, 4)
        mat = matrix_from_quat(dq)                 # (14, 3, 3)
        out = mat[..., :2].reshape(1, -1)
        return out.astype(np.float32, copy=False)

    def _get_object_quat_w(self):
        """Get object world orientation (wxyz) if available."""
        if self.freeze_object_state and self._cached_object_quat_w is not None:
            return self._cached_object_quat_w

        quat = None

        if hasattr(self, "current_object_quat_w") and self.current_object_quat_w is not None:
            quat = np.asarray(self.current_object_quat_w, dtype=np.float32).reshape(1, 4)

        elif hasattr(self.interface, "get_object_state"):
            state = self.interface.get_object_state()
            if state is not None:
                state = np.asarray(state, dtype=np.float32).reshape(-1)
                if state.size >= 7:
                    quat = xyzw_to_wxyz(state[3:7].reshape(1, 4))

        if quat is not None:
            self._cached_object_quat_w = quat.copy()
            return quat

        return self._cached_object_quat_w

    def _get_object_pos_w(self):
        """Best-effort access to the simulator object's world position."""
        if self.freeze_object_state and self._cached_object_pos_w is not None:
            return self._cached_object_pos_w

        pos = None

        if hasattr(self, "current_object_pos_w") and self.current_object_pos_w is not None:
            pos = np.asarray(self.current_object_pos_w, dtype=np.float32).reshape(1, 3)

        elif hasattr(self.interface, "get_object_state"):
            state = self.interface.get_object_state()
            if state is not None:
                state = np.asarray(state, dtype=np.float32).reshape(-1)
                if state.size >= 3:
                    pos = state[:3].reshape(1, 3)

        elif hasattr(self.interface, "get_object_pos_w"):
            p = self.interface.get_object_pos_w()
            if p is not None:
                pos = np.asarray(p, dtype=np.float32).reshape(1, 3)

        if pos is not None:
            self._cached_object_pos_w = pos.copy()
            return pos

        return self._cached_object_pos_w

    def _get_object_pose_torso(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Best-effort access to object pose in torso_link frame."""
        if not hasattr(self.interface, "get_object_pose_torso"):
            return None, None

        pose = self.interface.get_object_pose_torso()
        if pose is None:
            return None, None

        pose = np.asarray(pose, dtype=np.float32).reshape(-1)
        if pose.size < 7:
            return None, None

        pos_torso = pose[:3].reshape(1, 3)
        quat_torso_wxyz = xyzw_to_wxyz(pose[3:7].reshape(1, 4))
        return pos_torso, quat_torso_wxyz

    def _check_and_request_table_removal(self):
        """Policy inference must not request simulator scene changes."""
        return
    
    def _compute_bps_obj_state(
        self, robot_state_data
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute current object pos and 6D orientation in robot heading frame.

        Returns:
            obj_pos_rel  : (1, 3)  object position relative to torso in heading frame
            obj_rot_6d   : (1, 6)  object 6D rotation in heading frame (first 2 columns of R)
            yaw_quat     : (1, 4)  heading (yaw-only) quaternion wxyz for reuse
        """
        torso_pos_w, ref_quat_wxyz = self._get_ref_body_pose_in_world(robot_state_data)
        _, _, yaw = quat_to_rpy(ref_quat_wxyz.reshape(-1))
        yaw_quat = rpy_to_quat((0.0, 0.0, yaw)).reshape(1, 4)

        obj_pos_torso, obj_quat_torso_wxyz = self._get_object_pose_torso()
        if obj_pos_torso is not None:
            # Convert torso-frame measurement into the policy heading frame.
            delta_w = quat_apply(ref_quat_wxyz, obj_pos_torso)
            obj_pos_rel = quat_rotate_inverse(yaw_quat, delta_w).astype(np.float32)
        else:
            obj_pos_w = self._get_object_pos_w()
            if obj_pos_w is not None:
                delta = obj_pos_w - torso_pos_w.reshape(1, 3)
                obj_pos_rel = quat_rotate_inverse(yaw_quat, delta).astype(np.float32)
            else:
                obj_pos_rel = np.zeros((1, 3), dtype=np.float32)

        if obj_quat_torso_wxyz is not None:
            obj_quat_w = quat_mul(ref_quat_wxyz, self._normalize_quat_wxyz(obj_quat_torso_wxyz))
        else:
            obj_quat_w = self._get_object_quat_w()

        if obj_quat_w is not None:
            q_w2h = quat_inverse(yaw_quat)
            q_obj_h = quat_mul(q_w2h, self._normalize_quat_wxyz(obj_quat_w))
            R = matrix_from_quat(q_obj_h)  # (1, 3, 3)
            obj_rot_6d = R[0, :, :2].flatten().reshape(1, 6).astype(np.float32)
        else:
            obj_rot_6d = np.zeros((1, 6), dtype=np.float32)

        return obj_pos_rel, obj_rot_6d, yaw_quat

    def _compute_bps_motion_commands(
        self,
        robot_state_data,
        timestep: int,
        yaw_quat: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute short- and long-horizon object trajectory in heading frame for BPS policy.

        Short-horizon: t+1 .. t+10 (10 frames), absolute positions / orientations.
        Long-horizon : t+20, t+40, t+60, t+80, t+100 (5 frames).

        Returns:
            pos_short  : (1, 30)  10×3 positions
            ori_short  : (1, 60)  10×6 6D orientations
            pos_long   : (1, 15)   5×3 positions
            ori_long   : (1, 30)   5×6 6D orientations
        """
        use_recorded_world_traj = (
            self._recorded_obj_pos_w is not None
            and self._recorded_obj_rot_wxyz is not None
            and self._recorded_obj_pos_w.shape[0] > 0
            and self._recorded_obj_rot_wxyz.shape[0] > 0
        )

        if not use_recorded_world_traj and (self.motion_obj_pos is None or self.motion_obj_rot is None):
            return (
                np.zeros((1, 30), dtype=np.float32),
                np.zeros((1, 60), dtype=np.float32),
                np.zeros((1, 15), dtype=np.float32),
                np.zeros((1, 30), dtype=np.float32),
            )

        if yaw_quat is None:
            torso_pos_w, ref_quat_wxyz = self._get_ref_body_pose_in_world(robot_state_data)
            _, _, yaw = quat_to_rpy(ref_quat_wxyz.reshape(-1))
            yaw_quat = rpy_to_quat((0.0, 0.0, yaw)).reshape(1, 4)
        else:
            torso_pos_w, _ = self._get_ref_body_pose_in_world(robot_state_data)

        if use_recorded_world_traj:
            traj_len = min(self._recorded_obj_pos_w.shape[0], self._recorded_obj_rot_wxyz.shape[0])
            short_idx = np.clip(timestep + np.arange(1, 11), 0, traj_len - 1)
            long_idx = np.clip(timestep + np.array([20, 40, 60, 80, 100]), 0, traj_len - 1)

            torso_pos = torso_pos_w.reshape(1, 3)

            def _pos_to_heading(pos_w_batch: np.ndarray) -> np.ndarray:
                delta = pos_w_batch - torso_pos
                q_yaw = np.repeat(yaw_quat, delta.shape[0], axis=0)
                return quat_rotate_inverse(q_yaw, delta).astype(np.float32)

            pos_short = _pos_to_heading(self._recorded_obj_pos_w[short_idx]).reshape(1, -1)
            pos_long = _pos_to_heading(self._recorded_obj_pos_w[long_idx]).reshape(1, -1)

            q_w2h = quat_inverse(yaw_quat)

            def _ori_to_6d(rot_wxyz_batch: np.ndarray) -> np.ndarray:
                n = rot_wxyz_batch.shape[0]
                q_rep = np.repeat(q_w2h, n, axis=0)
                q_h = self._normalize_quat_wxyz(quat_mul(q_rep, self._normalize_quat_wxyz(rot_wxyz_batch)))
                R = matrix_from_quat(q_h)
                return R[..., :2].reshape(1, -1).astype(np.float32)

            ori_short = _ori_to_6d(self._recorded_obj_rot_wxyz[short_idx])
            ori_long = _ori_to_6d(self._recorded_obj_rot_wxyz[long_idx])
        else:
            torso_pos = torso_pos_w.reshape(1, 3)
            short_idx = np.clip(timestep + np.arange(1, 11), 0, self.motion_length - 1)
            long_idx  = np.clip(timestep + np.array([20, 40, 60, 80, 100]), 0, self.motion_length - 1)

            # Positions: (N, 3) world → heading frame relative to torso
            def _pos_to_heading(pos_w_batch: np.ndarray) -> np.ndarray:
                delta = pos_w_batch - torso_pos  # (N, 3)
                q = np.repeat(yaw_quat, delta.shape[0], axis=0)
                return quat_rotate_inverse(q, delta).astype(np.float32)

            pos_short = _pos_to_heading(self.motion_obj_pos[short_idx]).reshape(1, -1)
            pos_long  = _pos_to_heading(self.motion_obj_pos[long_idx]).reshape(1, -1)

            # Orientations: (N, 4) world wxyz → heading frame → 6D
            q_w2h = quat_inverse(yaw_quat)

            def _ori_to_6d(rot_wxyz_batch: np.ndarray) -> np.ndarray:
                n = rot_wxyz_batch.shape[0]
                q_rep = np.repeat(q_w2h, n, axis=0)
                q_h = self._normalize_quat_wxyz(quat_mul(q_rep, self._normalize_quat_wxyz(rot_wxyz_batch)))
                R = matrix_from_quat(q_h)  # (N, 3, 3)
                return R[..., :2].reshape(1, -1).astype(np.float32)  # (1, N*6)

            ori_short = _ori_to_6d(self.motion_obj_rot[short_idx])
            ori_long  = _ori_to_6d(self.motion_obj_rot[long_idx])

        return pos_short, ori_short, pos_long, ori_long

    def _compute_obj_pos_diff_b(self, robot_state_data):
        """Compute object position delta in the previous robot frame."""
        obj_pos_w = self._get_object_pos_w()
        if obj_pos_w is None:
            return np.zeros((1, 3), dtype=np.float32)

        # Use robot heading (yaw-only) orientation for object-relative obs
        ref_quat_wxyz = self._get_ref_body_orientation_in_world(robot_state_data)
        _, _, yaw = quat_to_rpy(ref_quat_wxyz.reshape(-1))
        ref_quat_wxyz = rpy_to_quat((0.0, 0.0, yaw)).reshape(1, 4)

        if self._prev_robot_obj_pos_w is None or self._prev_robot_anchor_quat_w is None:
            self._prev_robot_obj_pos_w = obj_pos_w.copy()
            self._prev_robot_anchor_quat_w = ref_quat_wxyz.copy()
            return np.zeros((1, 3), dtype=np.float32)

        delta_pos_w = obj_pos_w - self._prev_robot_obj_pos_w
        obj_pos_diff_b = quat_rotate_inverse(self._prev_robot_anchor_quat_w, delta_pos_w)

        self._prev_robot_obj_pos_w = obj_pos_w.copy()
        self._prev_robot_anchor_quat_w = ref_quat_wxyz.copy()
        return obj_pos_diff_b.astype(np.float32, copy=False)

    def _load_bps_code(self, obj_name: str) -> np.ndarray | None:
        """Load BPS shape encoding (512,) for the given object. Returns None on failure."""
        if obj_name in self._bps_cache:
            return self._bps_cache[obj_name]
        bps_root = Path("src/holosoma/holosoma/data/objects_new/objects_new")
        bps_path = bps_root / obj_name / f"{obj_name}_bps.pkl"
        if not bps_path.exists():
            logger.warning(f"BPS file not found: {bps_path}")
            return None
        bps_data = joblib.load(str(bps_path))
        code = bps_data["bps_code"]
        if hasattr(code, "numpy"):
            code = code.numpy()
        code = np.asarray(code, dtype=np.float32).flatten()  # (512,)
        self._bps_cache[obj_name] = code
        logger.info(f"Loaded BPS code {code.shape} for '{obj_name}'")
        return code

    def _load_mesh_points(self, obj_name: str) -> np.ndarray | None:
        if obj_name in self._mesh_points_cache:
            return self._mesh_points_cache[obj_name]
        objects_root = Path("src/holosoma/holosoma/data/objects_new/objects_new")
        mesh_points_path = objects_root / obj_name / f"{obj_name}_sample_points_1024.pkl"
        if not mesh_points_path.exists():
            if obj_name not in self._mesh_points_warned:
                logger.warning(f"Mesh points file not found: {mesh_points_path}")
                self._mesh_points_warned.add(obj_name)
            return None
        mesh_points_data = joblib.load(mesh_points_path)
        mesh_points_offset = np.asarray(mesh_points_data["points"], dtype=np.float32)
        self._mesh_points_cache[obj_name] = mesh_points_offset
        return mesh_points_offset

    def _send_debug_obj_pcd(self, points_world: np.ndarray) -> None:
        if not self._debug_obj_pcd_enabled or self._debug_obj_pcd_sock is None:
            return
        if points_world.size == 0:
            return
        self._debug_obj_pcd_step += 1
        if self._debug_obj_pcd_step % self._debug_obj_pcd_interval != 0:
            return
        points = points_world
        if self._debug_obj_pcd_stride > 1:
            points = points[:: self._debug_obj_pcd_stride]
        try:
            self._debug_obj_pcd_sock.sendto(
                points.astype(np.float32, copy=False).tobytes(),
                self._debug_obj_pcd_addr,
            )
        except OSError:
            # Best-effort debug channel; ignore send failures
            return

    def _get_motion_obj_traj_world(self, timestep: int) -> tuple[np.ndarray, np.ndarray] | None:
        if self.motion_obj_pos is None or self.motion_length == 0:
            return None
        if self.motion_obj_pos.ndim != 2 or self.motion_obj_pos.shape[1] != 3:
            return None

        short_idx = np.clip(timestep + np.arange(1, 11), 0, self.motion_length - 1)
        long_idx = np.clip(timestep + np.array([20, 40, 60, 80, 100]), 0, self.motion_length - 1)
        return self.motion_obj_pos[short_idx], self.motion_obj_pos[long_idx]

    def _send_debug_traj_viz(self, timestep: int) -> None:
        if not self._debug_traj_viz_enabled or self._debug_traj_viz_sock is None:
            return
        traj_world = self._get_motion_obj_traj_world(timestep)
        if traj_world is None:
            return
        short_pos_w, long_pos_w = traj_world
        packet = np.concatenate(
            [
                np.asarray(short_pos_w, dtype=np.float32).reshape(-1),
                np.asarray(long_pos_w, dtype=np.float32).reshape(-1),
            ],
            axis=0,
        )
        try:
            self._debug_traj_viz_sock.sendto(packet.tobytes(), self._debug_traj_viz_addr)
        except OSError:
            return

    def _send_mujoco_twin_state(
        self,
        robot_state_data: np.ndarray,
        current_obs_dict: dict,
    ) -> None:
        """Stream full robot+object state to the mujoco_twin.py live viewer via UDP.

        Packet layout (all float32, 149 floats = 596 bytes):
            [0]        num_dofs  (43.0)
            [1]        motion_timestep
            [2:9]      pelvis_pose_world  (pos_xyz + quat_wxyz)
            [9:52]     dof_pos_config     (43, config order, absolute)
            [52:95]    dof_vel_config     (43, config order)
            [95:98]    obj_pos_rel        (heading frame, relative to torso)
            [98:104]   obj_rot_6d         (heading frame)
            [104:134]  short_pos          (10×3, heading frame)
            [134:149]  long_pos           (5×3, heading frame)
        """
        if not self._mujoco_twin_enabled or self._mujoco_twin_sock is None:
            return
        if self.num_dofs != 43:
            return
        try:
            dof_pos_config = np.asarray(
                robot_state_data[0, 7 : 7 + self.num_dofs], dtype=np.float32
            )
            dof_vel_config = np.asarray(
                robot_state_data[0, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs],
                dtype=np.float32,
            )
            pelvis_pose_world = np.asarray(robot_state_data[0, :7], dtype=np.float32)

            task_terms = current_obs_dict.get("task", {})
            obj_pos_rel = np.asarray(
                task_terms.get("obj_pos_rel", np.zeros(3)), dtype=np.float32
            ).reshape(3)
            obj_rot_6d = np.asarray(
                task_terms.get("obj_rot_6d", np.zeros(6)), dtype=np.float32
            ).reshape(6)

            cmd_terms = current_obs_dict.get("command", {})
            short_pos = np.asarray(
                cmd_terms.get("motion_obj_pos_short", np.zeros(30)), dtype=np.float32
            ).reshape(30)
            long_pos = np.asarray(
                cmd_terms.get("motion_obj_pos_long", np.zeros(15)), dtype=np.float32
            ).reshape(15)

            packet = np.concatenate([
                np.array([float(self.num_dofs), float(self.motion_timestep)], dtype=np.float32),
                pelvis_pose_world,
                dof_pos_config,
                dof_vel_config,
                obj_pos_rel,
                obj_rot_6d,
                short_pos,
                long_pos,
            ])
            self._mujoco_twin_sock.sendto(packet.tobytes(), self._mujoco_twin_addr)
            if not self._mujoco_twin_first_packet_logged:
                logger.info(
                    colored(
                        f"MuJoCo twin first packet sent to {self._mujoco_twin_addr} "
                        f"({packet.size} float32 / {packet.nbytes} bytes)",
                        "cyan",
                    )
                )
                self._mujoco_twin_first_packet_logged = True
        except OSError as exc:
            if not self._mujoco_twin_send_error_logged:
                logger.warning(
                    colored(
                        f"MuJoCo twin UDP send failed to {self._mujoco_twin_addr}: "
                        f"{exc.__class__.__name__}: {exc}",
                        "yellow",
                    )
                )
                self._mujoco_twin_send_error_logged = True
            return

    def _get_sampled_motion_obj_traj_for_rviz(self) -> np.ndarray | None:
        # Use original world-frame object poses (preserved before alignment).
        # For gen_traj the trajectory is already in world frame and stored directly
        # in motion_obj_pos / motion_obj_rot, so fall back to those.
        pos_seq = self._motion_obj_pos_world if self._motion_obj_pos_world is not None else self.motion_obj_pos
        rot_seq = (
            self._motion_obj_rot_world_wxyz
            if self._motion_obj_rot_world_wxyz is not None
            else self.motion_obj_rot
        )
        if pos_seq is None or rot_seq is None or self.motion_fps is None:
            return None

        pos_arr = np.asarray(pos_seq, dtype=np.float32)
        rot_arr = np.asarray(rot_seq, dtype=np.float32)
        if pos_arr.ndim != 2 or pos_arr.shape[1] != 3:
            return None
        if rot_arr.ndim != 2 or rot_arr.shape[1] != 4:
            return None

        traj_len = min(pos_arr.shape[0], rot_arr.shape[0])
        if traj_len == 0:
            return None

        # Express trajectory relative to the object's initial (first-frame) pose
        init_pos = pos_arr[0:1]  # (1, 3)
        init_rot = rot_arr[0:1]  # (1, 4) wxyz
        init_rot_inv = quat_inverse(init_rot)  # (1, 4)

        rel_pos = quat_apply(
            np.repeat(init_rot_inv, traj_len, axis=0),
            pos_arr - init_pos,
        )
        rel_rot = self._normalize_quat_wxyz(
            quat_mul(np.repeat(init_rot_inv, traj_len, axis=0), rot_arr)
        )

        sample_step = max(1, int(round(self._rviz_traj_dt * float(self.motion_fps))))
        sample_idx = np.arange(0, traj_len, sample_step, dtype=np.int64)
        if sample_idx.size == 0 or sample_idx[-1] != traj_len - 1:
            sample_idx = np.concatenate([sample_idx, np.array([traj_len - 1], dtype=np.int64)], axis=0)

        packet = np.concatenate(
            [
                rel_pos[sample_idx],
                rel_rot[sample_idx],
            ],
            axis=1,
        )
        return packet.astype(np.float32, copy=False)

    def _send_rviz_traj(self) -> None:
        if not self._rviz_traj_enabled or self._rviz_traj_sock is None:
            return

        packet = self._get_sampled_motion_obj_traj_for_rviz()
        if packet is None:
            return

        try:
            self._rviz_traj_sock.sendto(
                packet.reshape(-1).astype(np.float32, copy=False).tobytes(),
                self._rviz_traj_addr,
            )
        except OSError:
            return

    def _compute_obj_pcd(self, robot_state_data) -> np.ndarray:
        obj_name = self.motion_obj_name
        if not obj_name:
            return np.zeros((1, 1024 * 3), dtype=np.float32)
        mesh_offset = self._load_mesh_points(obj_name)
        if mesh_offset is None:
            return np.zeros((1, 1024 * 3), dtype=np.float32)
        obj_pos = self._get_object_pos_w()
        obj_rot_wxyz = self._get_object_quat_w()
        if obj_pos is None or obj_rot_wxyz is None:
            return np.zeros((1, 1024 * 3), dtype=np.float32)

        # Rotation already in wxyz format
        mesh_offset_flat = mesh_offset.reshape(-1, 3)
        obj_rot_flat = np.repeat(obj_rot_wxyz, mesh_offset_flat.shape[0], axis=0)
        mesh_rel_flat = quat_apply(obj_rot_flat, mesh_offset_flat)
        mesh_rel = mesh_rel_flat.reshape(1, -1, 3)
        mesh_world = obj_pos.reshape(1, 1, 3) + mesh_rel

        # Convert to robot heading frame (torso)
        mesh_world_flat = mesh_world.reshape(-1, 3)
        # self._send_debug_obj_pcd(mesh_world_flat)
        torso_pos_w, ref_quat_wxyz = self._get_ref_body_pose_in_world(robot_state_data)

        rel_w = mesh_world_flat - torso_pos_w
        _, _, yaw = quat_to_rpy(ref_quat_wxyz.reshape(-1))
        yaw_quat = rpy_to_quat((0.0, 0.0, yaw)).reshape(1, 4)

        q = np.repeat(yaw_quat, rel_w.shape[0], axis=0)
        rel_h = quat_rotate_inverse(q, rel_w)

        return rel_h.reshape(1, -1).astype(np.float32, copy=False)

    def _compute_motion_obj_pcd(self, robot_state_data, timestep: int) -> np.ndarray:
        obj_name = self.motion_obj_name
        if not obj_name:
            return np.zeros((1, 1024 * 3), dtype=np.float32)
        mesh_offset = self._load_mesh_points(obj_name)
        if mesh_offset is None:
            return np.zeros((1, 1024 * 3), dtype=np.float32)
        if self.motion_obj_pos is None or self.motion_obj_rot is None or self.motion_length == 0:
            return np.zeros((1, 1024 * 3), dtype=np.float32)

        base_index = int(np.clip(timestep + 1, 0, self.motion_length - 1))
        obj_pos = self.motion_obj_pos[base_index].reshape(1, 3)
        obj_rot_wxyz = self.motion_obj_rot[base_index].reshape(1, 4)
        mesh_offset_flat = mesh_offset.reshape(-1, 3)
        obj_rot_flat = np.repeat(obj_rot_wxyz, mesh_offset_flat.shape[0], axis=0)
        mesh_rel_flat = quat_apply(obj_rot_flat, mesh_offset_flat)
        mesh_rel = mesh_rel_flat.reshape(1, -1, 3)
        mesh_world = obj_pos.reshape(1, 1, 3) + mesh_rel

        # Convert to current robot heading frame (torso)
        mesh_world_flat = mesh_world.reshape(-1, 3)
        torso_pos_w, torso_quat_wxyz = self._get_ref_body_pose_in_world(robot_state_data)

        rel_w = mesh_world_flat - torso_pos_w
        _, _, yaw = quat_to_rpy(torso_quat_wxyz.reshape(-1))
        yaw_quat = rpy_to_quat((0.0, 0.0, yaw)).reshape(1, 4)

        q = np.repeat(yaw_quat, rel_w.shape[0], axis=0)
        rel_h = quat_rotate_inverse(q, rel_w)

        return rel_h.reshape(1, -1).astype(np.float32, copy=False)

    def _stabilization_inference(self, robot_state_data):
        """Run master policy during stabilization to hold the initial pose."""
        if self._stabilization_policy_callable is None:
            # Fallback: hold current pose
            dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
            self.scaled_policy_action = dof_pos.copy()
            return self.scaled_policy_action

        group_outputs = self._prepare_master_group_observations(robot_state_data)
        if "obs" in self._stabilization_policy_input_names:
            if "actor_obs" in group_outputs:
                input_feed = {"obs": group_outputs["actor_obs"]}
            else:
                concat_obs = np.concatenate(
                    [group_outputs[group] for group in self._master_obs_dict.keys() if group in group_outputs],
                    axis=1,
                )
                input_feed = {"obs": concat_obs}
        else:
            input_feed = {name: group_outputs[name] for name in self._stabilization_policy_input_names}

        raw_action_order = self._stabilization_policy_callable(input_feed)

        # Action scaling/offset/clip (use same training parameters as main policy)
        action_scale = np.array([
            0.5475, 0.5475, 0.5475, 0.3507, 0.3507, 0.4386, 0.5475, 0.5475, 0.4386,
            0.3507, 0.3507, 0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.4386,
            0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.0745, 0.0745,
            0.0745, 0.0745, 0.7000, 0.7000, 0.3063, 0.7000, 0.7000, 0.3063, 0.7000,
            0.7000, 0.7000, 0.7000, 0.7000, 0.7000, 0.7000, 0.7000
        ], dtype=np.float32)

        action_offset = np.array([
            -0.3120, -0.3120, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.6690, 0.6690, 0.2000, 0.2000, -0.3630, -0.3630, 0.2000,
            -0.2000, 0.0000, 0.0000, 0.0000, 0.0000, 0.6000, 0.6000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -1.4000, -1.4000, 0.0000,
            1.4000, 1.4000, 0.0000, -1.5700, -1.5700, 0.3500, 1.5700, 1.5700,
            0.3500, 1.5700, -1.5700
        ], dtype=np.float32)

        action_clip_min = np.array([
            -10.9509, -10.9509, -10.9509, -7.0132, -7.0132, -8.7715, -10.9509, -10.9509, -8.7715,
            -7.0132, -7.0132, -8.7715, -8.7715, -8.7715, -8.7715, -8.7715, -8.7715, -8.7715,
            -8.7715, -8.7715, -8.7715, -8.7715, -8.7715, -8.7715, -8.7715, -1.4900, -1.4900,
            -1.4900, -1.4900, -14.0000, -14.0000, -6.1250, -14.0000, -14.0000, -6.1250, -14.0000,
            -14.0000, -14.0000, -14.0000, -14.0000, -14.0000, -14.0000, -14.0000
        ], dtype=np.float32)

        action_clip_max = np.array([
            10.9509, 10.9509, 10.9509, 7.0132, 7.0132, 8.7715, 10.9509, 10.9509, 8.7715,
            7.0132, 7.0132, 8.7715, 8.7715, 8.7715, 8.7715, 8.7715, 8.7715, 8.7715,
            8.7715, 8.7715, 8.7715, 8.7715, 8.7715, 8.7715, 8.7715, 1.4900, 1.4900,
            1.4900, 1.4900, 14.0000, 14.0000, 6.1250, 14.0000, 14.0000, 6.1250, 14.0000,
            14.0000, 14.0000, 14.0000, 14.0000, 14.0000, 14.0000, 14.0000
        ], dtype=np.float32)

        # Keep full action params (43-DOF, ACTION_ORDER_43DOF) for hand raw reconstruction
        action_scale_full = action_scale
        action_offset_full = action_offset
        action_clip_min_full = action_clip_min
        action_clip_max_full = action_clip_max

        # Master policy outputs 29-DOF body-only actions in stabilization order
        stabilization_order_indices = self._get_stabilization_action_order_indices()
        action_scale = action_scale_full[stabilization_order_indices]
        action_offset = action_offset_full[stabilization_order_indices]
        action_clip_min = action_clip_min_full[stabilization_order_indices]
        action_clip_max = action_clip_max_full[stabilization_order_indices]

        policy_action_order = raw_action_order * action_scale + action_offset
        policy_action_order = np.clip(policy_action_order, action_clip_min, action_clip_max)

        # Map stabilization action order -> config order
        action_body_indices = self._get_action_body_indices_config()
        if self.num_dofs == 43:
            # Expand to 43-DOF by holding hand joints at current positions
            dof_pos_config = robot_state_data[:, 7 : 7 + self.num_dofs]
            full_action_config = dof_pos_config.copy()
            full_action_config[:, action_body_indices] = policy_action_order
            # Override hand joints with target pose from the first motion frame (if available)
            target_pos_config = self._get_target_pos_config_first_frame()
            if target_pos_config is not None:
                hand_indices = self._get_action_hand_indices_config()
                if hand_indices.size > 0:
                    full_action_config[:, hand_indices] = target_pos_config[:, hand_indices]
        else:
            full_action_config = np.zeros((1, self.num_dofs), dtype=policy_action_order.dtype)
            full_action_config[:, action_body_indices] = policy_action_order

        # Update master last action in action order (29-DOF) for next observation
        self._master_last_action = raw_action_order.astype(np.float32, copy=False)

        # Build 43-DOF raw action (ACTION_ORDER_43DOF) to hand off after stabilization
        if self.num_dofs == 43:
            raw_action_order_43 = np.zeros((1, len(ACTION_ORDER_43DOF)), dtype=raw_action_order.dtype)
            raw_action_order_43[:, stabilization_order_indices] = raw_action_order

            action_order_to_config, _ = self._get_action_order_mappings()
            target_pos_action_order = full_action_config[:, action_order_to_config]
            raw_from_target = (target_pos_action_order - action_offset_full) / action_scale_full

            hand_indices = np.arange(self.motion_command_dofs, len(ACTION_ORDER_43DOF), dtype=np.int64)
            raw_action_order_43[:, hand_indices] = raw_from_target[:, hand_indices]

            self._stabilization_last_action_43 = raw_action_order_43.astype(np.float32, copy=False)
            self.last_policy_action = np.zeros((1, self.num_dofs), dtype=policy_action_order.dtype) #self._stabilization_last_action_43
        else:
            self.last_policy_action = raw_action_order.astype(np.float32, copy=False)

        # Do not touch main policy action history
        self.scaled_policy_action = full_action_config
        return self.scaled_policy_action

    def rl_inference(self, robot_state_data):
        self._stiff_startup_active = False
        # prepare obs, run policy inference
        if self.stabilization_mode:
            scaled_action = self._stabilization_inference(robot_state_data)
            current_obs_buffer_dict = self.get_current_obs_buffer_dict(robot_state_data)
            current_obs_dict = self.parse_current_obs_dict(current_obs_buffer_dict)
            self._send_mujoco_twin_state(robot_state_data, current_obs_dict)
            self.global_timestep += 1
            return scaled_action
        if not self.motion_clip_progressing and not self.stabilization_mode:
            # Keep motion index pinned at the configured start frame while waiting to trigger the clip.
            self._reset_motion_timing()
            
            # Get current and target positions
            dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
            
            # Get the configured start frame from motion data and interpolate to it.
            if self.motion_data is not None:
                self._stiff_startup_active = True
                target_pos_motion_order = self.motion_dof_pos[self._get_motion_start_frame_index()]
                # self.startup_pos = np.array([
                #     -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # left leg
                #     -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # right leg
                #     0.0, 0.0, 0.0,  # waist
                #     0.2, 0.2, 0.0, 0.3, 0.0, 0.0, 0.0,  # left arm
                #     0.0, 1.0, 1.7, -1.57, -1.7, -1.57, -1.7,  # left hand
                #     0.2, -0.2, 0.0, 0.3, 0.0, 0.0, 0.0,  # right arm
                #     0.0, -1.0, -1.7, 1.57, 1.7, 1.57, 1.7,  # right hand
                # ], dtype=np.float32)
                # target_pos_motion_order[:22] = self.startup_pos[:22]
                # target_pos_motion_order[29:36] = self.startup_pos[29:36]

                if self.num_dofs == 43 and target_pos_motion_order.shape[0] == 43:
                    # Motion clip is 43-DOF in G1_DEX3_JOINT_NAMES order; map directly to config order.
                    motion_to_config = np.array(
                        get_index_of_a_in_b(self.dof_names, list(G1_DEX3_JOINT_NAMES)), dtype=np.int64
                    )
                    target_pos_config = target_pos_motion_order[motion_to_config]
                else:
                    # Fallback to existing 29-DOF motion handling (G1_JOINT_NAMES order)
                    target_pos_asset = target_pos_motion_order[self.motion_to_asset_joint_map]  # [29] in asset order

                    if self.num_dofs == 43:
                        # For 43-DOF: reorder body joints and insert hand defaults at correct positions
                        reorder_map = np.array(ASSET_TO_CONFIG_ORDER_29DOF)
                        target_pos_config_29 = target_pos_asset[reorder_map]  # [29] in 29-DOF config order

                        # Hand default values (config order)
                        left_hand_defaults = np.array(
                            [0.0, 1.0, 1.7, -1.57, -1.7, -1.57, -1.7], dtype=np.float32
                        )
                        right_hand_defaults = np.array(
                            [0.0, -1.0, -1.7, 1.57, 1.7, 1.57, 1.7], dtype=np.float32
                        )

                        # Interleave hands in correct positions
                        # Config order: left_leg(6) + right_leg(6) + waist(3) + left_arm(4) + left_wrist(3) = 22
                        # Then: left_hand(7), right_arm(4) + right_wrist(3) = 7, right_hand(7)
                        target_pos_config = np.concatenate(
                            [
                                target_pos_config_29[:22],  # left_leg + right_leg + waist + left_arm + left_wrist
                                left_hand_defaults,  # left hand (7 joints)
                                target_pos_config_29[22:],  # right_arm + right_wrist (7 joints)
                                right_hand_defaults,  # right hand (7 joints)
                            ]
                        )
                    else:
                        # For 29-DOF: just reorder
                        reorder_map = np.array(ASSET_TO_CONFIG_ORDER_29DOF)
                        target_pos_config = target_pos_asset[reorder_map]  # [29] in config order
                
                target_pos_config = target_pos_config.reshape(1, -1)  # [1, num_dofs]

                # Two-phase interpolation to avoid elbow collision:
                # Phase 1: all joints except elbows move to target.
                # Phase 2: elbows move to target after phase 1 is complete.
                left_elbow_idx = self.dof_names.index("left_elbow_joint")
                right_elbow_idx = self.dof_names.index("right_elbow_joint")
                elbow_indices = [left_elbow_idx, right_elbow_idx]

                # Capture elbow start position once at the very beginning (step 0).
                # Use the last commanded position if available so the delta clamp in
                # base.py doesn't fight a sensor-vs-command mismatch.
                if self._elbow_interp_start_pos is None:
                    if self._prev_cmd_q is not None:
                        self._elbow_interp_start_pos = self._prev_cmd_q[elbow_indices].copy()
                    else:
                        self._elbow_interp_start_pos = dof_pos[0, elbow_indices].copy()

                non_elbow_mask = np.ones(self.num_dofs, dtype=bool)
                non_elbow_mask[elbow_indices] = False

                interpolated_pos = dof_pos.copy()
                if self.rl_interp_count < self.rl_interp_steps:
                    # Phase 1: only non-elbow joints move; elbows stay at their captured start position
                    ratio = (self.rl_interp_count + 1) / self.rl_interp_steps
                    interpolated_pos[0, non_elbow_mask] = (
                        dof_pos[0, non_elbow_mask]
                        + (target_pos_config[0, non_elbow_mask] - dof_pos[0, non_elbow_mask]) * ratio
                    )
                    interpolated_pos[0, elbow_indices] = self._elbow_interp_start_pos
                else:
                    # Phase 2: non-elbow joints hold at target; elbows interpolate from their captured start
                    ratio = (self.rl_interp_count - self.rl_interp_steps + 1) / self.rl_interp_steps
                    ratio = min(ratio, 1.0)
                    interpolated_pos[0, non_elbow_mask] = target_pos_config[0, non_elbow_mask]
                    interpolated_pos[0, elbow_indices] = (
                        self._elbow_interp_start_pos
                        + (target_pos_config[0, elbow_indices] - self._elbow_interp_start_pos) * ratio
                    )

                self.rl_interp_count += 1

                # Log dof_pos when both phases are complete (once)
                if self.rl_interp_count >= self.rl_interp_steps * 2 and not self.interp_complete_dof_pos_logged:
                    logger.info(colored("\n" + "="*80, "green", attrs=["bold"]))
                    logger.info(colored("🎯 Interpolation Complete! Current dof_pos (config order):", "green", attrs=["bold"]))
                    dof_pos_str = "[" + ", ".join(f"{v:.2f}" for v in dof_pos[0]) + "]"
                    logger.info(colored(f"   {dof_pos_str}", "cyan"))
                    logger.info(colored("="*80 + "\n", "green", attrs=["bold"]))
                    self.interp_complete_dof_pos_logged = True
                
                # Store as scaled_policy_action for compatibility with base policy
                self.scaled_policy_action = interpolated_pos #- self.default_dof_angles
                
                # Set last_policy_action to ZERO for the first observation (before motion starts)
                # This ensures the model receives zero action history at the initial pose
                self.last_policy_action = np.zeros((1, self.num_dofs), dtype=np.float32)
                
                return self.scaled_policy_action

        current_obs_buffer_dict = self.get_current_obs_buffer_dict(robot_state_data)
        current_obs_dict = self.parse_current_obs_dict(current_obs_buffer_dict)
        group_outputs = self._update_obs_history(current_obs_dict)

        # Collect motion log entry only during active motion playback (not stabilization).
        if self.motion_clip_progressing and not self.stabilization_mode:
            self._collect_motion_log_step(robot_state_data, current_obs_dict)

        # Stream full state to live MuJoCo twin viewer (always, not just during playback).
        self._send_mujoco_twin_state(robot_state_data, current_obs_dict)

        if "obs" in self.onnx_input_names:
            if "actor_obs" in group_outputs:
                input_feed = {"obs": group_outputs["actor_obs"]}
            else:
                concat_obs = np.concatenate(
                    [group_outputs[group] for group in self.obs_dict.keys() if group in group_outputs],
                    axis=1,
                )
                input_feed = {"obs": concat_obs}
        else:
            input_feed = {name: group_outputs[name] for name in self.onnx_input_names}

        raw_action_order = self.policy(input_feed)  # Raw action in action order

        # Action scaling/offset from training (in action order)
        # action = raw_action * scale + offset
        action_scale = np.array([
            0.5475, 0.5475, 0.5475, 0.3507, 0.3507, 0.4386, 0.5475, 0.5475, 0.4386,
            0.3507, 0.3507, 0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.4386,
            0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.0745, 0.0745,
            0.0745, 0.0745, 0.7000, 0.7000, 0.3063, 0.7000, 0.7000, 0.3063, 0.7000,
            0.7000, 0.7000, 0.7000, 0.7000, 0.7000, 0.7000, 0.7000
        ], dtype=np.float32)
        
        action_offset = np.array([
            -0.3120, -0.3120, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.6690, 0.6690, 0.2000, 0.2000, -0.3630, -0.3630, 0.2000,
            -0.2000, 0.0000, 0.0000, 0.0000, 0.0000, 0.6000, 0.6000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000
        ], dtype=np.float32)
        
        # Action clip limits from training (in action order)
        action_clip_min = np.array([
            -10.9509, -10.9509, -10.9509, -7.0132, -7.0132, -8.7715, -10.9509, -10.9509, -8.7715,
            -7.0132, -7.0132, -8.7715, -8.7715, -8.7715, -8.7715, -8.7715, -8.7715, -8.7715,
            -8.7715, -8.7715, -8.7715, -8.7715, -8.7715, -8.7715, -8.7715, -1.4900, -1.4900,
            -1.4900, -1.4900, -14.0000, -14.0000, -6.1250, -14.0000, -14.0000, -6.1250, -14.0000,
            -14.0000, -14.0000, -14.0000, -14.0000, -14.0000, -14.0000, -14.0000
        ], dtype=np.float32)
        
        action_clip_max = np.array([
            10.9509, 10.9509, 10.9509, 7.0132, 7.0132, 8.7715, 10.9509, 10.9509, 8.7715,
            7.0132, 7.0132, 8.7715, 8.7715, 8.7715, 8.7715, 8.7715, 8.7715, 8.7715,
            8.7715, 8.7715, 8.7715, 8.7715, 8.7715, 8.7715, 8.7715, 1.4900, 1.4900,
            1.4900, 1.4900, 14.0000, 14.0000, 6.1250, 14.0000, 14.0000, 6.1250, 14.0000,
            14.0000, 14.0000, 14.0000, 14.0000, 14.0000, 14.0000, 14.0000
        ], dtype=np.float32)
        
        # Handle 29-DOF vs 43-DOF
        if self.num_dofs == 29:
            action_scale = action_scale[:29]
            action_offset = action_offset[:29]
            action_clip_min = action_clip_min[:29]
            action_clip_max = action_clip_max[:29]
        
        # Apply scale and offset (in action order)
        policy_action_order = raw_action_order * action_scale + action_offset
        
        # Clip policy action using training clip limits
        policy_action_order = np.clip(policy_action_order, action_clip_min, action_clip_max)
        
        # Reorder action from action order to config order for simulator
        if self.num_dofs == 43:
            _, config_to_action = self._get_action_order_mappings()
            policy_action_config = policy_action_order[:, config_to_action]
        else:
            policy_action_config = policy_action_order[:, np.array(ACTION_ASSET_TO_CONFIG_ORDER_29DOF)]
        
        self.scaled_policy_action = policy_action_config
        
        # Store last policy action for observation (in action order)
        self.last_policy_action = raw_action_order
        
        # Debug mode: log dof_pos and scaled_policy_action, print every 100 steps
        if self.debug_mode:
            dof_pos_debug = robot_state_data[:, 7 : 7 + self.num_dofs].flatten()
            scaled_action_flat = self.scaled_policy_action.flatten()
            
            # Log to NPZ
            if self.debug_log_data is not None:
                self.debug_log_data["global_timestep"].append(self.global_timestep)
                self.debug_log_data["motion_timestep"].append(self.motion_timestep)
                self.debug_log_data["dof_pos"].append(dof_pos_debug.copy())
                self.debug_log_data["scaled_policy_action"].append(scaled_action_flat.copy())
            
            # Print every 100 steps
            if self.global_timestep % 100 == 0:
                # Find top 2 indices with largest difference
                diff = np.abs(scaled_action_flat - dof_pos_debug)
                top2_indices = np.argsort(diff)[-2:][::-1]  # Descending order
                
                logger.info(colored(f"\n[DEBUG timestep={self.global_timestep}]", "magenta", attrs=["bold"]))
                logger.info(colored(f"  scaled_policy_action: {np.array2string(scaled_action_flat, precision=4, suppress_small=True, max_line_width=150)}", "magenta"))
                logger.info(colored(f"  dof_pos:              {np.array2string(dof_pos_debug, precision=4, suppress_small=True, max_line_width=150)}", "magenta"))
                logger.info(colored(f"  Top 2 diff indices: [{top2_indices[0]}] diff={diff[top2_indices[0]]:.4f}, [{top2_indices[1]}] diff={diff[top2_indices[1]]:.4f}", "yellow", attrs=["bold"]))
        
        # Increment global timestep counter
        self.global_timestep += 1
        # update motion timestep (only when motion is actually progressing, not stabilization)
        if self.motion_clip_progressing and not self.stabilization_mode:
            # Check if we've hit the safety limit
            if self.max_motion_length is not None and self.motion_timestep >= self.max_motion_length:
                # Freeze at last action - don't increment timestep
                logger.warning(colored(
                    f"Motion timestep frozen at {self.motion_timestep} (max: {self.max_motion_length}). "
                    "Holding last action for safety. Set max_motion_length=None to disable.",
                    "yellow"
                ))
                return self.scaled_policy_action
            dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
            dof_command = self.cmd_q[:]

            if self.use_sim_time:
                self._update_clock()
            else:
                self.motion_timestep += 1

            if (end := self.config.task.motion_end_timestep) is not None and self.motion_timestep >= end:
                self.motion_timestep = end
                self.motion_clip_progressing = False
                self.logger.info(colored(f"Reached motion_end_timestep={end}, stopping motion clip", "yellow"))
        return self.scaled_policy_action

    def _get_motion_command_sequence_soft_stop(self) -> np.ndarray:
        """Get a 10-frame sequence repeating the soft stop target position (body-only)."""
        if self._soft_stop_target_q is None:
            return self._get_motion_command_sequence_first_frame()
        
        # Convert soft stop target from config order to motion command order (G1_JOINT_NAMES)
        target_q_config = self._soft_stop_target_q[0]  # [num_dofs] in config order
        
        # Extract body joints in config order (29-DOF only, exclude hands)
        if self.num_dofs == 43:
            # Get body joint names in G1_MOTION_JOINT_NAMES_29 order
            body_joint_names = list(G1_MOTION_JOINT_NAMES_29)
            # Map these to config order indices
            body_indices_config = np.array(
                get_index_of_a_in_b(body_joint_names, self.dof_names), dtype=np.int64
            )
            target_q_body_config = target_q_config[body_indices_config]  # [29] in G1_MOTION order
            # Already in G1_JOINT_NAMES order!
            target_q_motion = target_q_body_config
        else:
            # For 29-DOF robot, convert config -> G1_JOINT_NAMES
            body_joint_names = list(G1_MOTION_JOINT_NAMES_29)
            body_indices_config = np.array(
                get_index_of_a_in_b(body_joint_names, self.dof_names), dtype=np.int64
            )
            target_q_motion = target_q_config[body_indices_config]
        
        # Zero velocity for holding position
        vel0 = np.zeros_like(target_q_motion)
        
        # Concatenate pos and vel
        frame = np.concatenate([target_q_motion, vel0], axis=0).reshape(1, -1)  # [1, 58]
        seq = np.repeat(frame, 10, axis=0)  # [10, 58]
        return seq.reshape(1, -1).astype(np.float32, copy=False)

    def _get_manual_command(self, robot_state_data):
        # TODO: instead of adding kp/kd_override in def _set_motor_command,
        # just use the motor_kp/motor_kd when calling it in _fill_motor_commands
        if not self._stiff_hold_active:
            return None
        
        # Get current joint positions
        dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
        
        # Initialize start position on first call
        if self._stiff_hold_start_q is None:
            self._stiff_hold_start_q = dof_pos.copy()
            self._stiff_hold_interp_count = 0
        
        # Smooth interpolation over 1 second
        if self._stiff_hold_interp_count < self._stiff_hold_interp_steps:
            alpha = (self._stiff_hold_interp_count + 1) / self._stiff_hold_interp_steps
            q_target = self._stiff_hold_start_q + alpha * (self._stiff_hold_q - self._stiff_hold_start_q)
            self._stiff_hold_interp_count += 1
        else:
            # Interpolation complete, hold at target
            q_target = self._stiff_hold_q.copy()
        
        return {
            "q": q_target,
            "kp": self._stiff_hold_kp,
            "kd": self._stiff_hold_kd,
        }

    def _handle_start_policy(self):
        super()._handle_start_policy()
        self._stiff_hold_active = False
        # Don't start motion clip automatically - wait for 's' key
        self.motion_clip_progressing = False
        self.rl_interp_count = 0  # Reset interpolation counter
        self._elbow_interp_start_pos = None
        self.interp_complete_dof_pos_logged = False  # Reset logging flag
        self._capture_robot_yaw_offset()

    def _update_clock(self):
        # Use synchronized clock with motion-relative timing
        start_frame = self._get_motion_start_frame_index()
        current_clock = self.clock_sub.get_clock()
        if self.motion_start_timestep is None:
            # Motion just started; anchor to the first received clock tick.
            self.motion_start_timestep = current_clock
        elif self._last_clock_reading is not None and current_clock < self._last_clock_reading:
            # Simulator clock jumped backwards (e.g., reset). Re-anchor start time while preserving progress.
            progress_frames = max(self.motion_timestep - start_frame, 0)
            offset_ms = round(progress_frames * self.timestep_interval_ms)
            self.logger.warning("Clock sync returned earlier timestamp; adjusting motion timing anchor.")
            self.motion_start_timestep = current_clock - offset_ms
        self._last_clock_reading = current_clock
        elapsed_ms = current_clock - self.motion_start_timestep
        if self.motion_timestep == start_frame and int(elapsed_ms // self.timestep_interval_ms) > 1:
            self.logger.warning(
                "Still at the beginning but the clock jumped ahead: elapsed_ms={elapsed_ms}, self.timestep_interval_ms="
                "{timestep_interval_ms}, self.motion_timestep={motion_timestep}. "
                "Re-anchoring to the current timestamp so the motion always starts from the configured frame.",
                elapsed_ms=elapsed_ms,
                timestep_interval_ms=self.timestep_interval_ms,
                motion_timestep=self.motion_timestep,
            )
            self.motion_start_timestep = current_clock
            self._last_clock_reading = current_clock
            self.motion_timestep = start_frame
            return
        previous_motion_timestep = self.motion_timestep
        self.motion_timestep = start_frame + int(elapsed_ms // self.timestep_interval_ms)
        if self.motion_timestep != previous_motion_timestep:
            self.logger.info(
                "Motion timestep advanced from {previous_motion_timestep} to {motion_timestep}",
                previous_motion_timestep=previous_motion_timestep,
                motion_timestep=self.motion_timestep,
            )

    def _handle_stop_policy(self):
        """Handle stop policy action with two-stage stop (soft stop -> stiff stop)."""
        if self._record_traj_active:
            self.logger.warning("Stopping active world-frame trajectory recording without saving.")
            self._record_traj_active = False
            self._recorded_traj_time_buffer = []
            self._recorded_traj_pos_buffer = []
            self._recorded_traj_rot_buffer = []
        
        # First stop press: soft stop (hold current position with stabilization policy)
        if not self._soft_stop_active:
            self.use_policy_action = True  # Keep inference running!
            self.get_ready_state = False
            self._soft_stop_active = True
            self._stiff_hold_active = False
            self._stiff_startup_active = False
            
            # Capture current joint positions as target
            if self.num_dofs == 43:
                robot_state_data = self.interface.get_full_state_43dof()
            else:
                robot_state_data = self.interface.get_low_state()
            
            self._soft_stop_target_q = robot_state_data[:, 7 : 7 + self.num_dofs].copy()
            
            # Enter stabilization mode
            self.stabilization_mode = True
            self.motion_clip_progressing = False
            self._reset_motion_timing()
            self.robot_yaw_offset = 0.0
            # Reset table tracking so the next motion run starts fresh
            self._table_removed = False
            self._initial_object_height = None
            self._last_sent_table_frame = None
            self._table_removal_grace_steps = 0

            self.logger.info(colored("🟡 Soft stop activated - holding current position with stabilization policy", "yellow", attrs=["bold"]))
            self.logger.info(colored("   Press 'o' again to enter stiff mode", "yellow"))
            
            if hasattr(self.interface, "no_action"):
                self.interface.no_action = 0
        
        # Second stop press: stiff stop
        else:
            self.use_policy_action = False  # Stop inference
            self._soft_stop_active = False
            self._soft_stop_target_q = None
            self._stiff_hold_active = True
            self._stiff_startup_active = False
            self.stabilization_mode = False
            
            # Common cleanup for stiff mode
            self.motion_clip_progressing = False
            self._reset_motion_timing()
            self.robot_yaw_offset = 0.0
            # Reset table tracking so the next run starts fresh
            self._table_removed = False
            self._initial_object_height = None
            self._last_sent_table_frame = None
            self._table_removal_grace_steps = 0

            self.logger.info(colored("🔴 Stiff mode activated", "red", attrs=["bold"]))
            
            if hasattr(self.interface, "no_action"):
                self.interface.no_action = 0
            
            if self.debug_log_data is not None and len(self.debug_log_data.get("dof_pos", [])) > 0:
                self._save_debug_log()
                logger.info(colored("✓ Debug log saved on policy stop", "magenta", attrs=["bold"]))

            if self._motion_log_data:
                self._save_motion_log()

    def _handle_start_stabilization(self):
        """Handle start stabilization mode using the configured motion start frame."""
        if not self.use_policy_action:
            self.logger.warning("Press ']' first to enable policy, then 'd' to start stabilization")
            return
        
        self._stiff_startup_active = False
        self.stabilization_mode = True
        self.motion_clip_progressing = False
        self._reset_motion_timing()
        self._reset_master_obs_history()
        self.logger.info(
            colored("🛡️  Entering stabilization mode - policy active with configured start-frame command", "yellow", attrs=["bold"])
        )
        self.logger.info(colored("    Robot will stabilize and touch ground. Press 's' when ready to start motion.", "yellow"))

    def _initialize_robot_to_motion_frame(self) -> None:
        """Initialize robot joint angles and velocities to the configured motion start frame.
        
        This ensures the robot starts at the exact pose from the motion clip at motion_start_timestep,
        allowing smooth motion playback without interpolation artifacts.
        """
        if self.motion_data is None or self.motion_dof_pos is None:
            self.logger.debug("Cannot initialize robot to motion frame: no motion data")
            return
        
        start_frame_idx = self._get_motion_start_frame_index()
        if start_frame_idx < 0 or start_frame_idx >= self.motion_length:
            self.logger.warning(f"Start frame index {start_frame_idx} out of range [0, {self.motion_length})")
            return
        
        # Get target pose from motion
        target_pos_config = self._get_target_pos_config_first_frame()
        if target_pos_config is None:
            self.logger.warning("Cannot get target position from motion frame")
            return
        
        # Initialize interpolation counter to trigger smooth interpolation to start frame
        self.rl_interp_count = 0
        self.interp_complete_dof_pos_logged = False
        
        self.logger.info(
            colored(
                f"🎯 Initializing robot to motion frame {start_frame_idx}: "
                f"target_pos={target_pos_config[0, :5].tolist()}... (first 5 joints)",
                "green",
                attrs=["bold"],
            )
        )

    def _handle_start_motion_clip(self):
        """Handle start motion clip action."""
        self._stiff_startup_active = False
        if self._record_traj_active:
            self.logger.warning("Finish trajectory recording first before starting the motion clip.")
            return
        # Snapshot world poses at motion-start trigger ('s' first press).
        self._log_robot_object_pose_snapshot("s:start_motion")
        # Initialize pkl log for this motion clip
        self._init_motion_log()
        # Exit stabilization mode if active
        if self.stabilization_mode:
            self.stabilization_mode = False
            self.logger.info(colored("Exiting stabilization mode", "cyan"))
        if self.use_gen_traj and (not self._gen_traj_initialized) and (self._traj_gen is not None):
            obj_pos_w = self._get_object_pos_w()
            obj_quat_w = self._get_object_quat_w()
            if obj_pos_w is not None and obj_quat_w is not None:
                # obj_pos_w: [1,3], obj_quat_w: [1,4] (wxyz)
                pos_seq_w, quat_seq_wxyz = self._traj_gen.build_gen_traj(
                    start_pos=obj_pos_w[0],
                    start_quat_wxyz=obj_quat_w[0],
                )

                # Plug into "motion" containers used by obs pipeline
                self.motion_obj_pos = pos_seq_w.astype(np.float32, copy=False)
                self.motion_obj_rot = quat_seq_wxyz.astype(np.float32, copy=False)
                self.motion_obj_pos_aligned = self.motion_obj_pos.copy()
                self.motion_obj_rot_aligned_wxyz = self.motion_obj_rot.copy()
                # gen_traj is already in world frame
                self._motion_obj_pos_world = self.motion_obj_pos.copy()
                self._motion_obj_rot_world_wxyz = self.motion_obj_rot.copy()

                self.motion_fps = self.config.task.rl_rate
                self.motion_length = int(self.motion_obj_pos.shape[0])

                self._gen_traj_initialized = True
                logger.info(
                    colored(
                        f"Gen traj enabled: length={self.motion_length} frames "
                        f"({self.motion_length/self.motion_fps:.2f}s), mode={self._traj_gen.mode}",
                        "cyan",
                    )
                )
            else:
                logger.warning("Gen traj requested but object pose not available; keeping original motion object traj.")
        elif self.record_traj and self._recorded_obj_pos_w is None:
            logger.warning("record_traj is enabled but no recorded world-frame trajectory is available; using clip object trajectory.")
        self._send_rviz_traj()
        self.clock_sub.reset_origin()
        self.motion_clip_progressing = True
        # Initialize robot to the motion start frame pose
        self._initialize_robot_to_motion_frame()
        # Capture motion-specific start timestep for policy-level timing control
        self._reset_motion_timing()
        # Reset table tracking so each motion run starts with a fresh baseline.
        # Without this, a previous run's _table_removed=True would prevent
        # _update_table_pose_from_motion from ever sending table commands, and a
        # stale _initial_object_height would cause incorrect height-delta checks.
        self._table_removed = False
        self._initial_object_height = None
        self._last_sent_table_frame = None
        # Grace period: let the table settle at its motion-clip position before
        # monitoring object height.  Without this, _update_table_pose_from_motion
        # repositions the table on the first few ticks, causing the object to
        # bounce, which exceeds the 0.03 m threshold and triggers premature
        # table removal.
        self._table_removal_grace_steps = int(self.rl_rate * 2)  # 2 seconds
        # Seed main policy history buffers with the current observation for smooth transition
        robot_state_data = (
            self.interface.get_full_state_43dof() if self.num_dofs == 43 else self.interface.get_low_state()
        )
        if robot_state_data is not None:
            self._seed_obs_history_with_current(robot_state_data)
        start_idx = self._get_motion_start_frame_index()
        if self.config.task.motion_end_timestep is not None:
            self.logger.info(
                colored(
                    f"🎬 Starting motion clip playback from timestep {start_idx} to {self.config.task.motion_end_timestep}",
                    "blue",
                    attrs=["bold"],
                )
            )
        else:
            self.logger.info(colored(f"🎬 Starting motion clip playback from timestep {start_idx}", "blue", attrs=["bold"]))

    def handle_keyboard_button(self, keycode):
        """Add new keyboard button to start and end the motion clips"""
        if keycode in ["\r", "\n", "", "return", "enter"]:
            # Print current dof_pos when Enter is pressed
            robot_state_data = self.interface.get_low_state()
            if robot_state_data is not None:
                dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
                
                # Calculate target_pos_config for the configured motion start frame.
                target_pos_config = None
                max_diff_idx = None
                max_diff_val = None
                joint_name = None
                
                if self.motion_data is not None:
                    target_pos_motion_order = self.motion_dof_pos[self._get_motion_start_frame_index()]
                    
                    # Reorder from G1_JOINT_NAMES to asset order
                    target_pos_asset = target_pos_motion_order[self.motion_to_asset_joint_map]  # [29] in asset order
                    
                    # Reorder from asset order to config order
                    if self.num_dofs == 43:
                        # For 43-DOF: reorder body joints and insert hand defaults at correct positions
                        reorder_map = np.array(ASSET_TO_CONFIG_ORDER_29DOF)
                        target_pos_config_29 = target_pos_asset[reorder_map]  # [29] in 29-DOF config order
                        
                        # Hand default values (config order)
                        left_hand_defaults = np.array([0.0, 1.0, 1.7, -1.57, -1.7, -1.57, -1.7], dtype=np.float32)
                        right_hand_defaults = np.array([0.0, -1.0, -1.7, 1.57, 1.7, 1.57, 1.7], dtype=np.float32)
                        
                        # Interleave hands in correct positions
                        target_pos_config = np.concatenate([
                            target_pos_config_29[:22],  # left_leg + right_leg + waist + left_arm + left_wrist
                            left_hand_defaults,          # left hand (7 joints)
                            target_pos_config_29[22:],   # right_arm + right_wrist (7 joints)
                            right_hand_defaults          # right hand (7 joints)
                        ])
                    else:
                        # For 29-DOF: just reorder
                        reorder_map = np.array(ASSET_TO_CONFIG_ORDER_29DOF)
                        target_pos_config = target_pos_asset[reorder_map]  # [29] in config order
                    
                    # Calculate difference
                    diff = np.abs(dof_pos[0] - target_pos_config)
                    max_diff_idx = np.argmax(diff)
                    max_diff_val = diff[max_diff_idx]
                    
                    # Get joint name
                    joint_names = self.config.robot.dof_names
                    joint_name = joint_names[max_diff_idx]
                
                logger.info(colored("\n" + "="*80, "cyan", attrs=["bold"]))
                logger.info(colored(f"📊 Current dof_pos (config order) - rl_interp_count: {self.rl_interp_count}", "cyan", attrs=["bold"]))
                logger.info(colored(f"   {dof_pos[0].tolist()}", "yellow"))
                
                if target_pos_config is not None:
                    logger.info(colored(f"\n🎯 Target pos (first motion frame):", "cyan", attrs=["bold"]))
                    logger.info(colored(f"   {target_pos_config.tolist()}", "yellow"))
                    logger.info(colored(f"\n📏 Max difference:", "red", attrs=["bold"]))
                    logger.info(colored(f"   Joint [{max_diff_idx}]: {joint_name}", "red", attrs=["bold"]))
                    logger.info(colored(f"   Difference: {max_diff_val:.6f} rad ({np.degrees(max_diff_val):.2f} deg)", "red", attrs=["bold"]))
                    logger.info(colored(f"   Current: {dof_pos[0][max_diff_idx]:.6f}, Target: {target_pos_config[max_diff_idx]:.6f}", "yellow"))
                
                logger.info(colored("="*80 + "\n", "cyan", attrs=["bold"]))
            else:
                logger.warning("Unable to get robot state")
        elif keycode == "d":
            # Start stabilization mode - policy runs with first motion frame
            self._handle_start_stabilization()
        elif keycode == "t":
            self._toggle_record_traj()
        elif keycode == "s":
            # 's' behavior:
            # - If motion not started yet: start motion clip
            # - If motion already progressing: toggle object-state freeze (use cached pose)
            if not self.use_policy_action:
                self.logger.warning("Press ']' first to enable policy, then 's' to start motion")
                return

            if self.motion_clip_progressing:
                # Toggle freeze while motion is running
                self.freeze_object_state = not self.freeze_object_state

                if self.freeze_object_state:
                    # Snapshot once (populate cache) before freezing
                    pos = self._get_object_pos_w()
                    quat = self._get_object_quat_w()
                    self.logger.info(
                        colored(
                            f"🧊 Object state frozen. cached_pos={None if pos is None else pos.flatten().tolist()} "
                            f"cached_quat={None if quat is None else quat.flatten().tolist()}",
                            "cyan",
                            attrs=["bold"],
                        )
                    )
                else:
                    self.logger.info(colored("💧 Object state unfrozen (live updates)", "cyan", attrs=["bold"]))
                return

            # First 's': start motion
            self.clock_sub.reset_origin()
            self._handle_start_motion_clip()
        elif keycode == "m":
            # Toggle between motion tracking and policy
            self.use_motion_tracking = not self.use_motion_tracking
            mode = "MOTION TRACKING" if self.use_motion_tracking else "POLICY"
            self.logger.info(colored(f"Switched to {mode} mode", "cyan", attrs=["bold"]))
        else:
            super().handle_keyboard_button(keycode)

    def handle_joystick_button(self, cur_key):
        """Handle joystick button presses for WBT-specific controls."""
        if cur_key == "Y" and self.record_traj:
            self._toggle_record_traj()
        elif cur_key == "X":
            # Start stabilization mode
            self._handle_start_stabilization()
        elif cur_key == "start":
            # Start playing motion clip
            self._handle_start_motion_clip()
        else:
            # Delegate all other buttons to base class
            super().handle_joystick_button(cur_key)
        super()._print_control_status()

    def _capture_robot_yaw_offset(self):
        """Capture robot yaw when policy starts to use as reference offset."""
        robot_state_data = self.interface.get_low_state()
        if robot_state_data is None:
            self.robot_yaw_offset = 0.0
            self.logger.warning("Unable to capture robot yaw offset - missing robot state.")
            return

        robot_ref_ori = self._get_ref_body_orientation_in_world(robot_state_data)  # wxyz
        yaw = self._quat_yaw(robot_ref_ori)
        self.robot_yaw_offset = yaw
        root_pos = robot_state_data[0, :3].astype(np.float32, copy=False)
        self.robot_world_origin_w = np.array([root_pos[0], root_pos[1], 0.0], dtype=np.float32)
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

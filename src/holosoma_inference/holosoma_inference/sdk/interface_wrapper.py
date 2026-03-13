import json
import os
import socket
import numpy as np
from termcolor import colored

from holosoma_inference.config.config_types import RobotConfig
from holosoma_inference.sdk.command_sender import create_command_sender
from holosoma_inference.sdk.state_processor import create_state_processor

class InterfaceWrapper:
    """
    Wrapper for robot control supporting multiple backends:
    - sdk2py: uses Python SDK for both unitree and booster robots
    - unitree: uses C++/pybind11 binding for unitree robots only

    Backend selection based on `robot_config.sdk_type`:
      - 'booster': uses sdk2py (booster robots)
      - 'unitree': uses C++/pybind11 binding (unitree robots only)
    Provides a unified interface for get_low_state, send_low_command, and joystick input.
    Optionally supports Dex3-1 hand interfaces for robots with hands.
    """

    # ============================================================================
    # Initialization
    # ============================================================================

    def __init__(self, robot_config: RobotConfig, domain_id=0, interface_str=None, use_joystick=True, use_hands=False):
        self.use_joystick = use_joystick
        self.use_hands = use_hands
        self.robot_config = robot_config
        self.domain_id = domain_id
        self.interface_str = interface_str
        self.sdk_type = robot_config.sdk_type
        self.backend = None

        # Initialize gain levels for binding backend
        self._kp_level = 1.0
        self._kd_level = 1.0

        # Initialize sdk components
        self._init_sdk_components()

    def _init_sdk_components(self):
        """Initialize the appropriate backend based on SDK type."""
        if self.sdk_type == "booster":
            # Use sdk2py Python interface for booster
            self.backend = "sdk2py"
            self.command_sender = create_command_sender(self.robot_config)
            self.state_processor = create_state_processor(self.robot_config)
        elif self.sdk_type == "unitree":
            try:
                import unitree_interface
            except ImportError as e:
                raise ImportError("unitree_interface python binding not found.") from e
            # Use C++/pybind11 binding (unitree only)
            self.backend = "binding"
            # Parse robot type
            robot_type_map = {
                "G1": unitree_interface.RobotType.G1,
                "H1": unitree_interface.RobotType.H1,
                "H1_2": unitree_interface.RobotType.H1_2,
                "GO2": unitree_interface.RobotType.GO2,
            }
            # Parse message type
            message_type_map = {"HG": unitree_interface.MessageType.HG, "GO2": unitree_interface.MessageType.GO2}
            self.unitree_interface = unitree_interface.create_robot(
                self.interface_str,
                robot_type_map[self.robot_config.robot.upper()],
                message_type_map[self.robot_config.message_type.upper()],
            )
            # Set control mode to PR (Pitch/Roll)
            self.unitree_interface.set_control_mode(unitree_interface.ControlMode.PR)
            control_mode = self.unitree_interface.get_control_mode()
            print(f"Control mode set to: {'PR' if control_mode == unitree_interface.ControlMode.PR else 'AB'}")

            # GO2 SDK motor order differs from our joint order; override when using the binding.
            if self.robot_config.robot.lower() == "go2":
                # SDK motor order: FR, FL, RR, RL (each hip, thigh, calf)
                self._unitree_motor_order = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)
        else:
            raise ValueError(f"Unsupported SDK_TYPE: {self.sdk_type}")

        # Setup wireless controller
        if self.use_joystick:
            self._setup_wireless_controller()

        # Setup hand interfaces
        if self.use_hands:
            self._setup_hand_interfaces()

    # ============================================================================
    # Robot State and Command Interface
    # ============================================================================

    def get_low_state(self):
        """Get the latest low-level robot state as a numpy array."""
        if self.backend == "sdk2py":
            return self.state_processor.get_robot_state_data()
        if self.backend == "binding":
            return self._convert_binding_state_to_array()
        raise RuntimeError("InterfaceWrapper not initialized correctly.")

    def _convert_binding_state_to_array(self):

        state = self.unitree_interface.read_low_state()
        # Compose array: [base_pos(3), quat(4), joint_pos(N), base_lin_vel(3), base_ang_vel(3), joint_vel(N), ...]
        base_pos = np.zeros(3)
        quat = np.array(state.imu.quat)
        motor_pos = np.array(state.motor.q)
        base_lin_vel = np.zeros(3)
        base_ang_vel = np.array(state.imu.omega)
        motor_vel = np.array(state.motor.dq)

        # Determine how many motors are available in the body SDK
        num_available_motors = len(motor_pos)
        
        # For 43 DOF robots, the body SDK only has 29 motors (body only, no hands)
        # We return a 29-joint state array here. Hands are read separately via hand interfaces.
        if self.robot_config.num_joints == 43:
            # Robot has hands - only read body joints (29 DOF)
            num_body_joints = 29
            joint_pos = np.zeros(num_body_joints)
            joint_vel = np.zeros(num_body_joints)
            
            # For body motors, use identity mapping (motor i -> joint i for i=0..28)
            for j_id in range(num_body_joints):
                if j_id < num_available_motors:
                    joint_pos[j_id] = float(motor_pos[j_id])
                    joint_vel[j_id] = float(motor_vel[j_id])
        else:
            # Standard robot without hands - read all joints
            joint_pos = np.zeros(self.robot_config.num_joints)
            joint_vel = np.zeros(self.robot_config.num_joints)
            motor_order = self._unitree_motor_order or self.robot_config.joint2motor
            
            for j_id in range(self.robot_config.num_joints):
                m_id = motor_order[j_id]
                if m_id < num_available_motors:
                    joint_pos[j_id] = float(motor_pos[m_id])
                    joint_vel[j_id] = float(motor_vel[m_id])
                else:
                    raise RuntimeError(
                        f"Motor ID {m_id} for joint {j_id} exceeds available motors {num_available_motors}"
                    )
        return np.concatenate(
            [
                base_pos,
                quat,
                joint_pos,
                base_lin_vel,
                base_ang_vel,
                joint_vel,
            ]
        ).reshape(1, -1)

    def send_low_command(
        self,
        cmd_q,
        cmd_dq,
        cmd_tau,
        dof_pos_latest=None,
        kp_override=None,
        kd_override=None,
    ):
        if self.backend == "sdk2py":
            self.command_sender.send_command(
                cmd_q,
                cmd_dq,
                cmd_tau,
                dof_pos_latest,
                kp_override=kp_override,
                kd_override=kd_override,
            )
        elif self.backend == "binding":
            # Determine the number of joints to send
            # For 43 DOF robots, cmd_q will be 29 DOF (body only)
            # breakpoint()
            num_joints_to_send = len(cmd_q)
            
            # Apply waist_pitch sign flip before sending (motor convention differs from config convention)
            cmd_q = np.array(cmd_q, dtype=np.float32).copy()
            cmd_dq = np.array(cmd_dq, dtype=np.float32).copy()
            cmd_tau = np.array(cmd_tau, dtype=np.float32).copy()
            
            # For 43 DOF robots, body SDK only has 29 motors
            if self.robot_config.num_joints == 43:
                num_body_motors = 29
                # Body motors use identity mapping (motor i = joint i for i=0..28)
                cmd_q_target = np.zeros(num_body_motors)
                cmd_dq_target = np.zeros(num_body_motors)
                cmd_tau_target = np.zeros(num_body_motors)
                cmd_kp_override = np.zeros(num_body_motors) if kp_override is not None else None
                cmd_kd_override = np.zeros(num_body_motors) if kd_override is not None else None
                
                for j_id in range(min(num_joints_to_send, num_body_motors)):
                    cmd_q_target[j_id] = float(cmd_q[j_id])
                    cmd_dq_target[j_id] = float(cmd_dq[j_id])
                    cmd_tau_target[j_id] = float(cmd_tau[j_id])
                    if cmd_kp_override is not None:
                        cmd_kp_override[j_id] = float(kp_override[j_id])
                    if cmd_kd_override is not None:
                        cmd_kd_override[j_id] = float(kd_override[j_id])
            else:
                # Standard robot without hands
                cmd_q_target = np.zeros(self.robot_config.num_motors)
                cmd_dq_target = np.zeros(self.robot_config.num_motors)
                cmd_tau_target = np.zeros(self.robot_config.num_motors)
                cmd_kp_override = np.zeros(self.robot_config.num_motors) if kp_override is not None else None
                cmd_kd_override = np.zeros(self.robot_config.num_motors) if kd_override is not None else None
                motor_order = self._unitree_motor_order or self.robot_config.joint2motor
                
                for j_id in range(num_joints_to_send):
                    m_id = motor_order[j_id]
                    cmd_q_target[m_id] = float(cmd_q[j_id])
                    cmd_dq_target[m_id] = float(cmd_dq[j_id])
                    cmd_tau_target[m_id] = float(cmd_tau[j_id])
                    if cmd_kp_override is not None:
                        cmd_kp_override[m_id] = float(kp_override[j_id])
                    if cmd_kd_override is not None:
                        cmd_kd_override[m_id] = float(kd_override[j_id])
            
            self._send_binding_command(
                cmd_q_target,
                cmd_dq_target,
                cmd_tau_target,
                kp_override=cmd_kp_override,
                kd_override=cmd_kd_override,
            )
        else:
            raise RuntimeError("InterfaceWrapper not initialized correctly.")

    def _send_binding_command(self, cmd_q, cmd_dq, cmd_tau, kp_override=None, kd_override=None):
        """Send command using the C++/pybind11 binding."""
        cmd = self.unitree_interface.create_zero_command()
        cmd.q_target = list(cmd_q)
        cmd.dq_target = list(cmd_dq)
        cmd.tau_ff = list(cmd_tau)
        motor_kp = np.array(kp_override if kp_override is not None else self.robot_config.motor_kp)
        motor_kd = np.array(kd_override if kd_override is not None else self.robot_config.motor_kd)
        cmd.kp = list(motor_kp * self._kp_level)
        cmd.kd = list(motor_kd * self._kd_level)
        self.unitree_interface.write_low_command(cmd)

    # ============================================================================
    # Wireless Controller / Joystick Interface
    # ============================================================================

    def _setup_wireless_controller(self):
        """Setup wireless controller for joystick input."""
        if self.sdk_type == "unitree":
            # Wireless controller is already initialized in the binding
            pass
        elif self.sdk_type == "booster":
            from holosoma_inference.sdk.command_sender.booster.joystick_message import (
                BoosterJoystickMessage,
            )
            from holosoma_inference.sdk.command_sender.booster.remote_control_service import (
                BoosterRemoteControlService,
            )

            # Booster robots use evdev-based joystick input
            try:
                self.booster_remote_control = BoosterRemoteControlService()
                self.booster_joystick_msg = BoosterJoystickMessage(self.booster_remote_control)
                print(colored("Booster Remote Control Service Initialized", "green"))
            except ImportError as e:
                print(colored(f"Warning: Failed to initialize booster remote control: {e}", "yellow"))
                self.booster_remote_control = None
                self.booster_joystick_msg = None
        else:
            raise NotImplementedError(f"Joystick is not supported for {self.sdk_type} SDK.")
        self._wc_msg = None
        self._key_states = {}
        self._last_key_states = {}
        self._wc_key_map = self._default_wc_key_map()
        print(colored("Wireless Controller Initialized", "green"))

    def _default_wc_key_map(self):
        """Default wireless controller key mapping."""
        return {
            1: "R1",
            2: "L1",
            3: "L1+R1",
            4: "start",
            8: "select",
            10: "L1+select",
            16: "R2",
            32: "L2",
            64: "F1",
            128: "F2",
            256: "A",
            264: "select+A",
            512: "B",
            520: "select+B",
            544: "L2+B",  # Manual emergency stop
            768: "A+B",
            1024: "X",
            1032: "select+X",
            1280: "A+X",
            1536: "B+X",
            2048: "Y",
            2304: "A+Y",
            2560: "B+Y",
            2056: "select+Y",
            3072: "X+Y",
            4096: "up",
            4097: "R1+up",
            4352: "A+up",
            4608: "B+up",
            4104: "select+up",
            5120: "X+up",
            6144: "Y+up",
            8192: "right",
            8193: "R1+right",
            8448: "A+right",
            9216: "X+right",
            10240: "Y+right",
            8200: "select+right",
            16384: "down",
            16392: "select+down",
            16385: "R1+down",
            16640: "A+down",
            16896: "B+down",
            17408: "X+down",
            18432: "Y+down",
            32768: "left",
            32769: "R1+left",
            32776: "select+left",
            33024: "A+left",
            33792: "X+left",
            34816: "Y+left",
        }

    def wireless_controller_handler(self, msg):
        """Handle the wireless controller message."""
        self._wc_msg = msg

    def get_joystick_msg(self):
        """
        Get the latest joystick/wireless controller message in a unified format.
        Returns an object with .lx, .ly, .rx, .keys, etc. for both backends.
        """
        if self.sdk_type == "unitree":
            return self.unitree_interface.read_wireless_controller()
        if self.sdk_type == "booster":
            return self.booster_joystick_msg if hasattr(self, "booster_joystick_msg") else None
        return None

    def get_joystick_key(self, wc_msg=None):
        """
        Get the current key (cur_key) from the joystick message using the key map.
        """
        if wc_msg is None:
            wc_msg = self.get_joystick_msg()
        if wc_msg is None:
            return None
        return self._wc_key_map.get(getattr(wc_msg, "keys", 0), None)

    def process_joystick_input(self, lin_vel_command, ang_vel_command, stand_command, upper_body_motion_active):
        """
        Process joystick input and update commands in a unified way.
        Args:
            lin_vel_command: np.ndarray, shape (1, 2)
            ang_vel_command: np.ndarray, shape (1, 1)
            stand_command: np.ndarray, shape (1, 1)
            upper_body_motion_active: bool
        Returns:
            (lin_vel_command, ang_vel_command, key_states): updated values
        """
        wc_msg = self.get_joystick_msg()
        if wc_msg is None:
            return lin_vel_command, ang_vel_command, self._key_states
        # Process stick input
        if getattr(wc_msg, "keys", 0) == 0 and not upper_body_motion_active:
            lx = getattr(wc_msg, "lx", 0.0)
            ly = getattr(wc_msg, "ly", 0.0)
            rx = getattr(wc_msg, "rx", 0.0)
            lin_vel_command[0, 1] = -(lx if abs(lx) > 0.1 else 0.0) * stand_command[0, 0]
            lin_vel_command[0, 0] = (ly if abs(ly) > 0.1 else 0.0) * stand_command[0, 0]
            ang_vel_command[0, 0] = -(rx if abs(rx) > 0.1 else 0.0) * stand_command[0, 0]
        # Process button input
        cur_key = self.get_joystick_key(wc_msg)
        self._last_key_states = self._key_states.copy()
        if cur_key:
            self._key_states[cur_key] = True
        else:
            self._key_states = dict.fromkeys(self._wc_key_map.values(), False)

        return lin_vel_command, ang_vel_command, self._key_states

    # ============================================================================
    # Gain Management
    # ============================================================================

    @property
    def kp_level(self):
        """Get or set the proportional gain level."""
        if self.backend == "sdk2py":
            return self.command_sender.kp_level
        if self.backend == "binding":
            return self._kp_level
        return None

    @kp_level.setter
    def kp_level(self, value):
        if self.backend == "sdk2py":
            self.command_sender.kp_level = value
        elif self.backend == "binding":
            self._kp_level = value

    @property
    def kd_level(self):
        """Get or set the derivative gain level."""
        if self.backend == "sdk2py":
            return getattr(self.command_sender, "kd_level", 1.0)
        if self.backend == "binding":
            return self._kd_level
        return None

    @kd_level.setter
    def kd_level(self, value):
        if self.backend == "sdk2py":
            self.command_sender.kd_level = value
        elif self.backend == "binding":
            self._kd_level = value

    # ============================================================================
    # Hand Interface
    # ============================================================================

    def _setup_hand_interfaces(self):
        """Setup hand interfaces for Dex3-1 hands (only for unitree binding backend)."""
        if self.backend != "binding":
            print(colored("Warning: Hand interfaces are only supported with unitree binding backend", "yellow"))
            return

        if self.sdk_type != "unitree":
            print(colored("Warning: Hand interfaces are only supported for unitree robots", "yellow"))
            return

        try:
            import unitree_interface
            import time

            # CRITICAL: Wait for body interface DDS threads to stabilize before creating hands
            time.sleep(1.0)  # 1000ms delay - same as simulation bridge

            # Create left and right hand interfaces
            # IMPORTANT: Use re_init=False for BOTH hands since body interface already initialized DDS
            # Re-initializing DDS while threads are running causes segfaults
            self.left_hand_interface = unitree_interface.HandInterface.create_left_hand(
                self.interface_str, re_init=False
            )
            self.right_hand_interface = unitree_interface.HandInterface.create_right_hand(
                self.interface_str, re_init=False
            )
            
            # CRITICAL: Immediately send zero-gain commands to prevent unexpected motion
            passive_left_cmd = self.left_hand_interface.create_zero_command()
            passive_right_cmd = self.right_hand_interface.create_zero_command()
            
            # Set all gains to zero for passive mode
            passive_left_cmd.kp = [0.0] * 7
            passive_left_cmd.kd = [0.0] * 7
            passive_right_cmd.kp = [0.0] * 7
            passive_right_cmd.kd = [0.0] * 7
            
            # Send initial passive commands
            self.left_hand_interface.write_hand_command(passive_left_cmd)
            self.right_hand_interface.write_hand_command(passive_right_cmd)
            
            print(colored("Hand Interfaces Initialized (Left and Right) - Passive Mode Active", "green"))
        except ImportError as e:
            print(colored(f"Warning: Failed to initialize hand interfaces: {e}", "yellow"))
            self.left_hand_interface = None
            self.right_hand_interface = None
        except Exception as e:
            print(colored(f"Error: Failed to setup hand interfaces: {e}", "red"))
            self.left_hand_interface = None
            self.right_hand_interface = None

    def send_hand_command(self, left_hand_cmd=None, right_hand_cmd=None):
        """
        Send command to hand(s).

        Args:
            left_hand_cmd: HandMotorCommand for left hand (or None to skip)
            right_hand_cmd: HandMotorCommand for right hand (or None to skip)
        """
        if not self.use_hands:
            return

        if left_hand_cmd is not None and self.left_hand_interface is not None:
            self.left_hand_interface.write_hand_command(left_hand_cmd)

        if right_hand_cmd is not None and self.right_hand_interface is not None:
            self.right_hand_interface.write_hand_command(right_hand_cmd)

    # ============================================================================
    # 43 DOF Full State and Command (Body + Hands)
    # ============================================================================

    def get_full_state_43dof(self):
        """
        Get full 43 DOF state (body + hands) in correct joint order.
        
        Returns:
            np.ndarray: Shape (1, 3+4+43+3+3+43) = (1, 99)
                [base_pos(3), quat(4), joint_pos(43), base_lin_vel(3), base_ang_vel(3), joint_vel(43)]
                
        Joint order (43 DOF):
            0-11: legs (12)
            12-14: waist (3)
            15-21: left arm (7)
            22-28: left hand (7)
            29-35: right arm (7)
            36-42: right hand (7)
        """
        if not self.use_hands:
            raise RuntimeError("Hand interfaces not initialized. Set use_hands=True.")
        
        # Get body state (29 DOF) - shape (1, 3+4+29+3+3+29) = (1, 71)
        body_state = self.get_low_state()
        
        # Get hand states
        left_state = self.left_hand_interface.read_hand_state() if self.left_hand_interface else None
        right_state = self.right_hand_interface.read_hand_state() if self.right_hand_interface else None
        
        if left_state is None or right_state is None:
            raise RuntimeError("Failed to read hand states")
        
        # Extract body components
        base_pos = body_state[:, 0:3]
        quat = body_state[:, 3:7]
        body_joint_pos_29 = body_state[:, 7:36]  # 29 body joints
        base_lin_vel = body_state[:, 36:39]
        base_ang_vel = body_state[:, 39:42]
        body_joint_vel_29 = body_state[:, 42:71]  # 29 body joint velocities
        
        # Extract hand positions and velocities
        left_hand_pos = np.array(left_state.motor.q).reshape(1, 7)
        left_hand_vel = np.array(left_state.motor.dq).reshape(1, 7)
        right_hand_pos = np.array(right_state.motor.q).reshape(1, 7)
        right_hand_vel = np.array(right_state.motor.dq).reshape(1, 7)
        
        # Reconstruct 43 DOF joint positions in correct order
        # 0-11: legs, 12-14: waist, 15-21: left arm, 22-28: left hand, 29-35: right arm, 36-42: right hand
        joint_pos_43 = np.zeros((1, 43))
        joint_vel_43 = np.zeros((1, 43))
        
        # Map 29 body joints to 43 DOF positions
        joint_pos_43[:, 0:15] = body_joint_pos_29[:, 0:15]      # legs + waist
        joint_pos_43[:, 15:22] = body_joint_pos_29[:, 15:22]    # left arm
        joint_pos_43[:, 22:29] = left_hand_pos                   # left hand
        joint_pos_43[:, 29:36] = body_joint_pos_29[:, 22:29]    # right arm
        joint_pos_43[:, 36:43] = right_hand_pos                  # right hand
        
        joint_vel_43[:, 0:15] = body_joint_vel_29[:, 0:15]      # legs + waist
        joint_vel_43[:, 15:22] = body_joint_vel_29[:, 15:22]    # left arm
        joint_vel_43[:, 22:29] = left_hand_vel                   # left hand
        joint_vel_43[:, 29:36] = body_joint_vel_29[:, 22:29]    # right arm
        joint_vel_43[:, 36:43] = right_hand_vel                  # right hand
        
        # Concatenate into full state
        return np.concatenate([
            base_pos,
            quat,
            joint_pos_43,
            base_lin_vel,
            base_ang_vel,
            joint_vel_43,
        ], axis=1)

    def send_full_command_43dof(
        self,
        cmd_q_43,
        cmd_dq_43,
        cmd_tau_43,
        dof_pos_latest_43=None,
        kp_override_43=None,
        kd_override_43=None,
    ):
        """
        Send full 43 DOF command (body + hands) with correct joint mapping.
        
        Args:
            cmd_q_43: Target positions for all 43 joints
            cmd_dq_43: Target velocities for all 43 joints
            cmd_tau_43: Feedforward torques for all 43 joints
            dof_pos_latest_43: Latest positions for all 43 joints (optional)
            kp_override_43: KP gains for all 43 joints (optional)
            kd_override_43: KD gains for all 43 joints (optional)
            
        Joint order (43 DOF):
            0-11: legs (12)
            12-14: waist (3)
            15-21: left arm (7)
            22-28: left hand (7)
            29-35: right arm (7)
            36-42: right hand (7)
        """
        if not self.use_hands:
            raise RuntimeError("Hand interfaces not initialized. Set use_hands=True.")
        
        # Ensure inputs are numpy arrays
        cmd_q_43 = np.asarray(cmd_q_43).flatten()
        cmd_dq_43 = np.asarray(cmd_dq_43).flatten()
        cmd_tau_43 = np.asarray(cmd_tau_43).flatten()
        
        if len(cmd_q_43) != 43:
            raise ValueError(f"Expected 43 joint positions, got {len(cmd_q_43)}")
        
        # Extract body commands (29 DOF)
        # Map 43 DOF -> 29 DOF: 0-14 (legs+waist), 15-21 (left arm), 29-35 (right arm)
        cmd_q_body = np.zeros(29)
        cmd_dq_body = np.zeros(29)
        cmd_tau_body = np.zeros(29)
        
        cmd_q_body[0:15] = cmd_q_43[0:15]      # legs + waist
        cmd_q_body[15:22] = cmd_q_43[15:22]    # left arm
        cmd_q_body[22:29] = cmd_q_43[29:36]    # right arm
        
        cmd_dq_body[0:15] = cmd_dq_43[0:15]
        cmd_dq_body[15:22] = cmd_dq_43[15:22]
        cmd_dq_body[22:29] = cmd_dq_43[29:36]
        
        cmd_tau_body[0:15] = cmd_tau_43[0:15]
        cmd_tau_body[15:22] = cmd_tau_43[15:22]
        cmd_tau_body[22:29] = cmd_tau_43[29:36]
        
        # Handle optional parameters
        dof_pos_latest_body = None
        if dof_pos_latest_43 is not None:
            dof_pos_latest_43 = np.asarray(dof_pos_latest_43).flatten()
            dof_pos_latest_body = np.zeros(29)
            dof_pos_latest_body[0:15] = dof_pos_latest_43[0:15]
            dof_pos_latest_body[15:22] = dof_pos_latest_43[15:22]
            dof_pos_latest_body[22:29] = dof_pos_latest_43[29:36]
        
        # Always map 43-DOF gains into body motor order so the body SDK
        # does not receive hand gains for right-arm motors.
        if kp_override_43 is None and self.robot_config.motor_kp is not None:
            kp_override_43 = np.asarray(self.robot_config.motor_kp, dtype=np.float32).flatten()
        if kd_override_43 is None and self.robot_config.motor_kd is not None:
            kd_override_43 = np.asarray(self.robot_config.motor_kd, dtype=np.float32).flatten()

        kp_override_body = None
        if kp_override_43 is not None:
            if len(kp_override_43) != 43:
                raise ValueError(f"Expected 43 KP values, got {len(kp_override_43)}")
            kp_override_body = np.zeros(29, dtype=np.float32)
            kp_override_body[0:15] = kp_override_43[0:15]
            kp_override_body[15:22] = kp_override_43[15:22]
            kp_override_body[22:29] = kp_override_43[29:36]
        
        kd_override_body = None
        if kd_override_43 is not None:
            if len(kd_override_43) != 43:
                raise ValueError(f"Expected 43 KD values, got {len(kd_override_43)}")
            kd_override_body = np.zeros(29, dtype=np.float32)
            kd_override_body[0:15] = kd_override_43[0:15]
            kd_override_body[15:22] = kd_override_43[15:22]
            kd_override_body[22:29] = kd_override_43[29:36]
        
        # Send body command (29 DOF)
        self.send_low_command(
            cmd_q_body,
            cmd_dq_body,
            cmd_tau_body,
            dof_pos_latest_body,
            kp_override=kp_override_body,
            kd_override=kd_override_body,
        )
        
        # Extract hand commands (7 DOF each)
        left_hand_cmd = self.left_hand_interface.create_zero_command() if self.left_hand_interface else None
        right_hand_cmd = self.right_hand_interface.create_zero_command() if self.right_hand_interface else None
        
        if left_hand_cmd is not None:
            left_hand_cmd.q_target = cmd_q_43[22:29].tolist()      # indices 22-28
            left_hand_cmd.dq_target = cmd_dq_43[22:29].tolist()
            left_hand_cmd.tau_ff = cmd_tau_43[22:29].tolist()
            
            # Always set kp/kd for hands (kp_override_43 is populated from robot_config if originally None)
            if kp_override_43 is not None:
                left_hand_cmd.kp = kp_override_43[22:29].tolist()
            else:
                # Fallback: use default hand gains if robot_config didn't have motor_kp
                left_hand_cmd.kp = [2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
            if kd_override_43 is not None:
                left_hand_cmd.kd = kd_override_43[22:29].tolist()
            else:
                # Fallback: use default hand gains if robot_config didn't have motor_kd
                left_hand_cmd.kd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        
        if right_hand_cmd is not None:
            right_hand_cmd.q_target = cmd_q_43[36:43].tolist()     # indices 36-42
            right_hand_cmd.dq_target = cmd_dq_43[36:43].tolist()
            right_hand_cmd.tau_ff = cmd_tau_43[36:43].tolist()
            
            # Always set kp/kd for hands (kp_override_43 is populated from robot_config if originally None)
            if kp_override_43 is not None:
                right_hand_cmd.kp = kp_override_43[36:43].tolist()
            else:
                # Fallback: use default hand gains if robot_config didn't have motor_kp
                right_hand_cmd.kp = [2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
            if kd_override_43 is not None:
                right_hand_cmd.kd = kd_override_43[36:43].tolist()
            else:
                # Fallback: use default hand gains if robot_config didn't have motor_kd
                right_hand_cmd.kd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        
        # Send hand commands
        self.send_hand_command(left_hand_cmd, right_hand_cmd)

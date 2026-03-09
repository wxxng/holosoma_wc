import time

import numpy as np
from loguru import logger
from unitree_interface import (
    LowState,
    MessageType,
    MotorCommand,
    RobotType,
    UnitreeInterface,
    WirelessController,
)

from holosoma.bridge.base.basic_sdk2py_bridge import BasicSdk2Bridge


class UnitreeSdk2Bridge(BasicSdk2Bridge):
    """Unitree SDK bridge implementation using unitree_interface C++ bindings.

    Supports standard robots (g1_29dof, h1, h1-2, go2_12dof) and
    robots with hands (g1_43dof) in a unified implementation.
    """

    SUPPORTED_ROBOT_TYPES = {"g1_29dof", "h1", "h1-2", "go2_12dof", "g1_43dof"}

    def _init_sdk_components(self):
        """Initialize Unitree SDK-specific components."""

        robot_type = self.robot.asset.robot_type

        # Validate robot type first
        if robot_type not in self.SUPPORTED_ROBOT_TYPES and not robot_type.startswith("g1_43dof_"):
            raise ValueError(
                f"Invalid robot type '{robot_type}'. Unitree SDK supports: {self.SUPPORTED_ROBOT_TYPES}"
            )

        # Check if this robot has hands (43 DOF)
        self.has_hands = robot_type == "g1_43dof" or robot_type.startswith("g1_43dof_")

        # Get network interface from config
        interface_name = self.bridge_config.interface 

        if self.has_hands:
            self._init_hands_components(interface_name)
        else:
            self._init_standard_components(robot_type, interface_name)

    def _init_standard_components(self, robot_type, interface_name):
        """Initialize standard (non-hands) SDK components."""
        # Map robot type to SDK enum
        robot_type_map = {
            "g1_29dof": RobotType.G1,
            "h1": RobotType.H1,
            "h1-2": RobotType.H1_2,
            "go2_12dof": RobotType.GO2,
        }

        # Map to message type (HG for humanoid robots with 35 motors, GO2 for others)
        message_type_map = {
            "g1_29dof": MessageType.HG,
            "h1": MessageType.GO2,
            "h1-2": MessageType.HG,
            "go2_12dof": MessageType.GO2,
        }

        sdk_robot_type = robot_type_map[robot_type]
        sdk_message_type = message_type_map[robot_type]

        # Create interface (handles DDS initialization internally)
        self.interface = UnitreeInterface(interface_name, sdk_robot_type, sdk_message_type)

        # Initialize data structures
        self.low_state = LowState(self.num_motor)
        self.low_cmd = MotorCommand(self.num_motor)
        self.wireless_controller = WirelessController()

    def _init_hands_components(self, interface_name):
        """Initialize 43 DOF (29 body + 14 hands) SDK components."""
        from unitree_interface import HandInterface

        # === Body control (29 DOF) using C++ bindings ===
        self.interface = UnitreeInterface(interface_name, RobotType.G1, MessageType.HG)
        self.low_state = LowState(29)  # Body only
        self.low_cmd = MotorCommand(29)
        self.wireless_controller = WirelessController()

        # Initialize low_cmd with zero values (passive by default)
        for i in range(29):
            self.low_cmd.q_target[i] = 0.0
            self.low_cmd.dq_target[i] = 0.0
            self.low_cmd.tau_ff[i] = 0.0
            self.low_cmd.kp[i] = 0.0
            self.low_cmd.kd[i] = 0.0

        # CRITICAL: Add delay after body interface initialization to avoid segfault
        # DDS and thread initialization need time to stabilize before creating more channels
        logger.info("Waiting for body interface to stabilize before initializing hands...")
        time.sleep(1.0)  # 1000ms delay to ensure DDS/threads are ready

        # === Hand control (14 DOF) using C++ bindings ===
        self.num_body_motor = 29
        self.num_hand_motor = 14  # 7 per hand

        # MuJoCo joint order: legs(0-11), waist(12-14), left_arm(15-21), left_hand(22-28), right_arm(29-35), right_hand(36-42)
        # SDK expects: legs(0-11), waist(12-14), left_arm(15-21), right_arm(22-28), hands(29-42)
        self.mujoco_left_hand_indices = list(range(22, 29))   # [22-28]
        self.mujoco_right_hand_indices = list(range(36, 43))  # [36-42]

        # Initialize hand interfaces using C++ bindings
        # Use re_init=False to avoid reinitializing DDS that might interfere with body interface
        self.left_hand_interface = HandInterface.create_left_hand(interface_name, re_init=False)
        self.right_hand_interface = HandInterface.create_right_hand(interface_name, re_init=False)

        # Initialize hand command structures with zero values
        self.left_hand_cmd = self.left_hand_interface.create_zero_command()
        self.right_hand_cmd = self.right_hand_interface.create_zero_command()

        self.left_hand_cmd.kp = [0.0] * 7
        self.left_hand_cmd.kd = [0.0] * 7
        self.right_hand_cmd.kp = [0.0] * 7
        self.right_hand_cmd.kd = [0.0] * 7

        logger.info("Initialized C++ SDK bridge: 29 body motors + 14 hand motors (all via C++ binding)")

    def low_cmd_handler(self, msg=None):
        """Handle Unitree low-level command messages."""
        # Poll for incoming commands from DDS
        cmd = self.interface.read_incoming_command()
        if self.has_hands:
            if cmd:
                self.low_cmd = cmd
        else:
            self.low_cmd = cmd

    def publish_low_state(self):
        """Publish Unitree low-level state."""
        if self.has_hands:
            self._publish_low_state_with_hands()
        else:
            self._publish_low_state_standard()

    def _publish_low_state_standard(self):
        """Publish state for standard robots (no hands)."""
        # Get simulator data
        positions, velocities, accelerations = self._get_dof_states()
        actuator_forces = self._get_actuator_forces()
        quaternion, gyro, acceleration = self._get_base_imu_data()

        # Populate motor state
        self.low_state.motor.q = positions.tolist()
        self.low_state.motor.dq = velocities.tolist()
        self.low_state.motor.ddq = accelerations.tolist()
        self.low_state.motor.tau_est = actuator_forces.tolist()

        # Populate IMU state
        quat_array = quaternion.detach().cpu().numpy()
        self.low_state.imu.quat = [
            float(quat_array[0]),  # w
            float(quat_array[1]),  # x
            float(quat_array[2]),  # y
            float(quat_array[3]),  # z
        ]
        self.low_state.imu.omega = gyro.detach().cpu().numpy().tolist()
        self.low_state.imu.accel = acceleration.detach().cpu().numpy().tolist()

        # Set timestamp (milliseconds)
        self.low_state.tick = int(self.sim_time * 1e3)

        # Publish (CRC calculated automatically in C++)
        self.interface.publish_low_state(self.low_state)

    def _publish_low_state_with_hands(self):
        """Publish state for 43 DOF robots (29 body + 14 hands)."""
        from unitree_interface import HandState

        # Read body commands from C++ interface (if any new commands available)
        self.low_cmd_handler()

        # Get simulator data (43 DOF total in MuJoCo order)
        positions, velocities, accelerations = self._get_dof_states()

        # Use computed torques instead of actuator_forces (which is empty without explicit actuators)
        torques = self.torques if self.torques is not None else np.zeros(self.num_motor)

        quaternion, gyro, acceleration = self._get_base_imu_data()

        # === Publish body state (29 DOF) via C++ ===
        # Map MuJoCo indices [0-21, 29-35] to SDK body indices [0-28]
        body_positions = np.concatenate([positions[0:22], positions[29:36]])
        body_velocities = np.concatenate([velocities[0:22], velocities[29:36]])
        body_accelerations = np.concatenate([accelerations[0:22], accelerations[29:36]])
        body_torques = np.concatenate([torques[0:22], torques[29:36]])

        self.low_state.motor.q = body_positions.tolist()
        self.low_state.motor.dq = body_velocities.tolist()
        self.low_state.motor.ddq = body_accelerations.tolist()
        self.low_state.motor.tau_est = body_torques.tolist()

        # Populate IMU state
        quat_array = quaternion.detach().cpu().numpy()
        self.low_state.imu.quat = [
            float(quat_array[0]),  # w
            float(quat_array[1]),  # x
            float(quat_array[2]),  # y
            float(quat_array[3]),  # z
        ]
        self.low_state.imu.omega = gyro.detach().cpu().numpy().tolist()
        self.low_state.imu.accel = acceleration.detach().cpu().numpy().tolist()
        self.low_state.tick = int(self.sim_time * 1e3)

        # Publish body state
        self.interface.publish_low_state(self.low_state)

        # === Publish hand states via C++ binding ===
        # Read any incoming hand commands from policy
        self.left_hand_cmd = self.left_hand_interface.read_incoming_command()
        self.right_hand_cmd = self.right_hand_interface.read_incoming_command()

        # Left hand state (MuJoCo indices 22-28)
        left_hand_state = HandState()
        left_hand_state.motor.q = [float(positions[22 + i]) for i in range(7)]
        left_hand_state.motor.dq = [float(velocities[22 + i]) for i in range(7)]
        left_hand_state.motor.tau_est = [float(torques[22 + i]) for i in range(7)]
        left_hand_state.motor.temperature = [[0, 0] for _ in range(7)]  # Not simulated
        left_hand_state.motor.voltage = [0.0] * 7  # Not simulated

        # Right hand state (MuJoCo indices 36-42)
        right_hand_state = HandState()
        right_hand_state.motor.q = [float(positions[36 + i]) for i in range(7)]
        right_hand_state.motor.dq = [float(velocities[36 + i]) for i in range(7)]
        right_hand_state.motor.tau_est = [float(torques[36 + i]) for i in range(7)]
        right_hand_state.motor.temperature = [[0, 0] for _ in range(7)]  # Not simulated
        right_hand_state.motor.voltage = [0.0] * 7  # Not simulated

        # Publish hand states via C++ binding
        self.left_hand_interface.publish_hand_state(left_hand_state)
        self.right_hand_interface.publish_hand_state(right_hand_state)

    def publish_wireless_controller(self):
        """Publish wireless controller data using unitree_interface."""
        # Call base class to populate wireless_controller from joystick
        super().publish_wireless_controller()

        # Publish using C++ interface
        if self.joystick is not None:
            self.interface.publish_wireless_controller(self.wireless_controller)

    def compute_torques(self):
        """Compute torques for all supported robots."""
        if self.has_hands:
            return self._compute_torques_with_hands()
        else:
            return self._compute_torques_standard()

    def _compute_torques_standard(self):
        """Compute torques using Unitree's unified command structure."""
        if not (hasattr(self, "low_cmd") and self.low_cmd):
            return self.torques

        try:
            # Extract from Unitree's unified structure
            return self._compute_pd_torques(
                tau_ff=self.low_cmd.tau_ff,
                kp=self.low_cmd.kp,
                kd=self.low_cmd.kd,
                q_target=self.low_cmd.q_target,
                dq_target=self.low_cmd.dq_target,
            )
        except Exception as e:
            logger.error(f"Error computing torques: {e}")
            raise

    def _compute_torques_with_hands(self):
        """Compute torques for 43 DOF (29 body + 14 hands) with proper index mapping."""
        if not (hasattr(self, "low_cmd") and self.low_cmd and self.left_hand_cmd and self.right_hand_cmd):
            return self.torques

        try:
            # Build full 43 DOF command arrays with proper index mapping
            tau_ff_full = np.zeros(43)
            kp_full = np.zeros(43)
            kd_full = np.zeros(43)
            q_target_full = np.zeros(43)
            dq_target_full = np.zeros(43)

            # === Body commands (29 DOF from SDK) ===
            # SDK [0-21] -> MuJoCo [0-21] (legs, waist, left arm)
            tau_ff_full[0:22] = self.low_cmd.tau_ff[0:22]
            kp_full[0:22] = self.low_cmd.kp[0:22]
            kd_full[0:22] = self.low_cmd.kd[0:22]
            q_target_full[0:22] = self.low_cmd.q_target[0:22]
            dq_target_full[0:22] = self.low_cmd.dq_target[0:22]

            # SDK [22-28] -> MuJoCo [29-35] (right arm)
            tau_ff_full[29:36] = self.low_cmd.tau_ff[22:29]
            kp_full[29:36] = self.low_cmd.kp[22:29]
            kd_full[29:36] = self.low_cmd.kd[22:29]
            q_target_full[29:36] = self.low_cmd.q_target[22:29]
            dq_target_full[29:36] = self.low_cmd.dq_target[22:29]

            # === Hand commands ===
            # Left hand: SDK [0-6] -> MuJoCo [22-28]
            tau_ff_full[22:29] = self.left_hand_cmd.tau_ff[0:7]
            kp_full[22:29] = self.left_hand_cmd.kp[0:7]
            kd_full[22:29] = self.left_hand_cmd.kd[0:7]
            q_target_full[22:29] = self.left_hand_cmd.q_target[0:7]
            dq_target_full[22:29] = self.left_hand_cmd.dq_target[0:7]

            # Right hand: SDK [0-6] -> MuJoCo [36-42]
            tau_ff_full[36:43] = self.right_hand_cmd.tau_ff[0:7]
            kp_full[36:43] = self.right_hand_cmd.kp[0:7]
            kd_full[36:43] = self.right_hand_cmd.kd[0:7]
            q_target_full[36:43] = self.right_hand_cmd.q_target[0:7]
            dq_target_full[36:43] = self.right_hand_cmd.dq_target[0:7]

            # Use base class helper for PD control computation
            return self._compute_pd_torques(
                tau_ff=tau_ff_full,
                kp=kp_full,
                kd=kd_full,
                q_target=q_target_full,
                dq_target=dq_target_full,
            )

        except Exception as e:
            logger.error(f"Error computing torques: {e}")
            raise

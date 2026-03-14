import json
import time
import socket

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

OBJECT_STATE_UDP_PORT = 10002
OBJECT_STATE_UDP_HOST = "127.0.0.1"
ROBOT_STATE_UDP_PORT = 10003
ROBOT_STATE_UDP_HOST = "127.0.0.1"
TABLE_COMMAND_UDP_PORT = 10004
TABLE_COMMAND_UDP_HOST = "127.0.0.1"
SCENE_INFO_UDP_PORT = 10005
SCENE_INFO_UDP_HOST = "127.0.0.1"
TRAJ_VIZ_UDP_PORT = 10006
TRAJ_VIZ_UDP_HOST = "127.0.0.1"


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

        # Object/robot state UDP publisher (sim2sim on loopback)
        self._object_state_sock = None
        self._object_state_addr = None
        self._object_state_warned = False
        self._robot_state_sock = None
        self._robot_state_addr = None
        self._robot_state_warned = False
        self._scene_info_sock = None
        self._scene_info_addr = None
        self._scene_info_warned = False
        self._traj_viz_sock = None
        self._traj_viz_short_pos = None
        self._traj_viz_long_pos = None
        self._table_command_sock = None
        self._table_missing_warned = False
        self._table_actor_available = None
        self._table_body_id = None
        self._table_original_pos = None   # saved before first move, for reset
        self._table_original_quat = None
        self._scene_reset_counter = 0
        if interface_name == "lo":
            self._object_state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._object_state_addr = (OBJECT_STATE_UDP_HOST, OBJECT_STATE_UDP_PORT)
            logger.info(
                f"Object state UDP publisher enabled on {OBJECT_STATE_UDP_HOST}:{OBJECT_STATE_UDP_PORT}"
            )
            self._robot_state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._robot_state_addr = (ROBOT_STATE_UDP_HOST, ROBOT_STATE_UDP_PORT)
            logger.info(
                f"Robot state UDP publisher enabled on {ROBOT_STATE_UDP_HOST}:{ROBOT_STATE_UDP_PORT}"
            )
            self._scene_info_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._scene_info_addr = (SCENE_INFO_UDP_HOST, SCENE_INFO_UDP_PORT)
            logger.info(
                f"Scene info UDP publisher enabled on {SCENE_INFO_UDP_HOST}:{SCENE_INFO_UDP_PORT}"
            )
            try:
                self._traj_viz_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._traj_viz_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self._traj_viz_sock.bind((TRAJ_VIZ_UDP_HOST, TRAJ_VIZ_UDP_PORT))
                self._traj_viz_sock.setblocking(False)
                logger.info(
                    f"Trajectory viz UDP receiver enabled on {TRAJ_VIZ_UDP_HOST}:{TRAJ_VIZ_UDP_PORT}"
                )
            except OSError as exc:
                self._traj_viz_sock = None
                logger.warning(f"Failed to init trajectory viz UDP receiver: {exc}")
            try:
                self._table_command_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._table_command_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self._table_command_sock.bind((TABLE_COMMAND_UDP_HOST, TABLE_COMMAND_UDP_PORT))
                self._table_command_sock.setblocking(False)
                logger.info(
                    f"Table command UDP receiver enabled on {TABLE_COMMAND_UDP_HOST}:{TABLE_COMMAND_UDP_PORT}"
                )
            except OSError as exc:
                self._table_command_sock = None
                logger.warning(f"Failed to init table command UDP receiver: {exc}")

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
        self._poll_table_commands()
        self._poll_traj_viz_packets()
        self._render_traj_viz()
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
        self._publish_object_state_udp()
        self._publish_robot_state_udp()
        self._publish_scene_info_udp()

    def _publish_low_state_with_hands(self):
        """Publish state for 43 DOF robots (29 body + 14 hands)."""
        from unitree_interface import HandState

        self._poll_table_commands()
        self._poll_traj_viz_packets()
        self._render_traj_viz()

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
        self._publish_object_state_udp()
        self._publish_robot_state_udp()
        self._publish_scene_info_udp()

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

    def _publish_object_state_udp(self):
        if self._object_state_sock is None or self._object_state_addr is None:
            return
        try:
            states = self.simulator.get_actor_states(["object"], env_ids=None)
        except Exception:
            if not self._object_state_warned:
                logger.warning(
                    "Object state not available in simulator. Ensure MJCF has a body named 'object' "
                    "and it's registered in the simulator object registry."
                )
                self._object_state_warned = True
            return
        if states is None or states.numel() == 0:
            return
        state = states[0].detach().cpu().numpy().astype(np.float32, copy=False)
        self._object_state_sock.sendto(state.tobytes(), self._object_state_addr)

    def _publish_robot_state_udp(self):
        if self._robot_state_sock is None or self._robot_state_addr is None:
            return
        try:
            states = self.simulator.get_actor_states(["robot"], env_ids=None)
        except Exception:
            if not self._robot_state_warned:
                logger.warning(
                    "Robot state not available in simulator. Ensure robot is registered in the object registry."
                )
                self._robot_state_warned = True
            return
        if states is None or states.numel() == 0:
            return
        state = states[0].detach().cpu().numpy().astype(np.float32, copy=False)
        self._robot_state_sock.sendto(state.tobytes(), self._robot_state_addr)

    def _get_scene_object_name(self) -> str | None:
        robot_type = str(self.robot.asset.robot_type)
        if robot_type.startswith("g1_43dof_"):
            object_name = robot_type[len("g1_43dof_"):].strip()
            return object_name or None
        return None

    def _publish_scene_info_udp(self):
        if self._scene_info_sock is None or self._scene_info_addr is None:
            return
        object_name = self._get_scene_object_name()
        if not object_name:
            if not self._scene_info_warned:
                logger.warning("Scene object name is unavailable; skipping scene info UDP publish.")
                self._scene_info_warned = True
            return
        packet = json.dumps(
            {
                "object_name": object_name,
                "reset_counter": int(self._scene_reset_counter),
            }
        ).encode("utf-8")
        try:
            self._scene_info_sock.sendto(packet, self._scene_info_addr)
        except OSError as exc:
            if not self._scene_info_warned:
                logger.warning(f"Failed to publish scene info UDP packet: {exc}")
                self._scene_info_warned = True

    @staticmethod
    def _wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
        q = np.asarray(quat_wxyz, dtype=np.float32).reshape(-1)
        if q.size != 4:
            raise ValueError(f"Expected 4 quaternion values, got shape {q.shape}")
        return np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)

    def _poll_table_commands(self) -> None:
        if self._table_command_sock is None:
            return
        while True:
            try:
                data, _addr = self._table_command_sock.recvfrom(8 * 4)
            except BlockingIOError:
                break
            if not data:
                continue
            packet = np.frombuffer(data, dtype=np.float32)
            if packet.size < 1:
                continue
            cmd_id = int(round(float(packet[0])))
            if cmd_id == 1 and packet.size >= 8:
                pos = packet[1:4].astype(np.float32, copy=True)
                quat_wxyz = packet[4:8].astype(np.float32, copy=True)
                self._apply_table_pose(pos, quat_wxyz)
            elif cmd_id == 2:
                self._remove_table()

    def _poll_traj_viz_packets(self) -> None:
        if self._traj_viz_sock is None:
            return
        while True:
            try:
                data, _addr = self._traj_viz_sock.recvfrom(15 * 3 * 4)
            except BlockingIOError:
                break
            if not data:
                continue
            packet = np.frombuffer(data, dtype=np.float32)
            if packet.size != 15 * 3:
                continue
            positions = packet.reshape(15, 3)
            self._traj_viz_short_pos = positions[:10].copy()
            self._traj_viz_long_pos = positions[10:].copy()

    def _render_traj_viz(self) -> None:
        if self.simulator.viewer is None:
            return
        if self._traj_viz_short_pos is None and self._traj_viz_long_pos is None:
            return

        try:
            import mujoco
        except Exception:
            return

        scn = self.simulator.viewer.user_scn
        scn.ngeom = 0
        self.add_traj_viz_to_scene(scn, clear_existing=False)

    def add_traj_viz_to_scene(self, scn, *, clear_existing: bool = False) -> None:
        """Append trajectory debug spheres to an arbitrary MuJoCo scene."""
        if self._traj_viz_short_pos is None and self._traj_viz_long_pos is None:
            return

        try:
            import mujoco
        except Exception:
            return

        if clear_existing:
            scn.ngeom = 0

        identity_mat = np.eye(3).flatten()
        short_rgba = np.array([0.2, 0.9, 0.2, 0.8], dtype=np.float32)
        long_rgba = np.array([0.9, 0.2, 0.2, 0.8], dtype=np.float32)
        size = np.array([0.015, 0.0, 0.0], dtype=np.float64)

        short_pos = None if self._traj_viz_short_pos is None else self._traj_viz_short_pos.copy()
        long_pos = None if self._traj_viz_long_pos is None else self._traj_viz_long_pos.copy()

        for points, rgba in ((short_pos, short_rgba), (long_pos, long_rgba)):
            if points is None:
                continue
            for pos in points:
                if scn.ngeom >= scn.maxgeom:
                    return
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    size,
                    np.asarray(pos, dtype=np.float64),
                    identity_mat,
                    rgba,
                )
                scn.ngeom += 1

    def _get_table_state(self):
        if self._table_actor_available is False:
            return None
        try:
            states = self.simulator.get_actor_states(["table"], env_ids=None)
        except Exception:
            self._table_actor_available = False
            if not self._table_missing_warned:
                logger.warning(
                    "Table is not registered as actor state; using static-body fallback update for 'table'."
                )
                self._table_missing_warned = True
            return None
        if states is None or states.numel() == 0:
            self._table_actor_available = False
            return None
        self._table_actor_available = True
        return states

    def _apply_table_pose_static_body(self, pos_xyz: np.ndarray, quat_wxyz: np.ndarray) -> bool:
        """Fallback path for static MuJoCo table body without freejoint actor state."""
        root_model = getattr(self.simulator, "root_model", None)
        root_data = getattr(self.simulator, "root_data", None)
        if root_model is None or root_data is None:
            return False

        try:
            import mujoco
        except Exception:
            return False

        if self._table_body_id is None:
            body_name = "table"
            body_id = mujoco.mj_name2id(root_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0 and hasattr(self.simulator, "_get_prefixed_name"):
                prefixed = self.simulator._get_prefixed_name(body_name)
                body_id = mujoco.mj_name2id(root_model, mujoco.mjtObj.mjOBJ_BODY, prefixed)
            self._table_body_id = int(body_id)

        if self._table_body_id < 0:
            return False

        # Save the original position/orientation before the first modification so we
        # can restore them when the simulator is reset (R key).
        if self._table_original_pos is None:
            self._table_original_pos = root_model.body_pos[self._table_body_id].copy()
            self._table_original_quat = root_model.body_quat[self._table_body_id].copy()

        root_model.body_pos[self._table_body_id] = np.asarray(pos_xyz, dtype=np.float32)
        root_model.body_quat[self._table_body_id] = np.asarray(quat_wxyz, dtype=np.float32)
        # Reflect model-body pose edits in runtime body transforms.
        mujoco.mj_kinematics(root_model, root_data)
        return True

    def restore_table(self) -> None:
        """Restore the table body to its original XML position after a simulator reset.

        mj_resetData() only resets mjData (qpos/qvel/etc.), not mjModel.body_pos,
        so any _apply_table_pose_static_body / _remove_table edits persist across
        resets unless we explicitly undo them here.
        """
        if self._table_original_pos is None:
            return  # table was never moved; nothing to restore

        root_model = getattr(self.simulator, "root_model", None)
        root_data = getattr(self.simulator, "root_data", None)
        if root_model is None or root_data is None:
            return

        try:
            import mujoco as _mujoco
        except Exception:
            return

        if self._table_body_id is None or self._table_body_id < 0:
            return

        root_model.body_pos[self._table_body_id] = self._table_original_pos.copy()
        root_model.body_quat[self._table_body_id] = self._table_original_quat.copy()
        _mujoco.mj_kinematics(root_model, root_data)
        self._scene_reset_counter += 1
        logger.info("Table body restored to original XML position after reset.")

    def _apply_table_pose(self, pos_xyz: np.ndarray, quat_wxyz: np.ndarray) -> None:
        quat = np.asarray(quat_wxyz, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(quat))
        if norm <= 0.0:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            quat = quat / norm

        table_states = self._get_table_state()
        if table_states is None:
            self._apply_table_pose_static_body(pos_xyz, quat)
            return

        quat_xyzw = self._wxyz_to_xyzw(quat)

        new_states = table_states.clone()
        new_states[0, 0:3] = new_states.new_tensor(pos_xyz, dtype=new_states.dtype)
        new_states[0, 3:7] = new_states.new_tensor(quat_xyzw, dtype=new_states.dtype)
        new_states[0, 7:13] = 0.0

        self.simulator.set_actor_states(["table"], env_ids=None, states=new_states)

    def _remove_table(self) -> None:
        table_states = self._get_table_state()
        if table_states is None:
            # Static table fallback: move table body below ground.
            self._apply_table_pose_static_body(
                np.array([0.0, 0.0, -5.0], dtype=np.float32),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )
            return
        new_states = table_states.clone()
        new_states[0, 2] = -5.0
        new_states[0, 7:13] = 0.0
        self.simulator.set_actor_states(["table"], env_ids=None, states=new_states)

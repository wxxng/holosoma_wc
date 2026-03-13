import json
import os
import socket
import numpy as np
from loguru import logger
from termcolor import colored

from holosoma_inference.config.config_types import RobotConfig
from holosoma_inference.sdk.command_sender import create_command_sender
from holosoma_inference.sdk.state_processor import create_state_processor
from holosoma_inference.utils.math.quat import quat_inverse, quat_mul, wxyz_to_xyzw, xyzw_to_wxyz

OBJECT_STATE_UDP_PORT = 10002
OBJECT_STATE_UDP_HOST = "127.0.0.1"
ROBOT_STATE_UDP_PORT = 10003
ROBOT_STATE_UDP_HOST = "127.0.0.1"
TABLE_COMMAND_UDP_PORT = 10004
TABLE_COMMAND_UDP_HOST = "127.0.0.1"

# World topics UDP (JSON) receiver (for external state sources).
# Can be overridden via environment variables.
WORLD_TOPICS_UDP_BIND_IP = os.environ.get("HOLOSOMA_WORLD_TOPIC_UDP_BIND_IP", "0.0.0.0")
WORLD_TOPICS_UDP_PORT = int(os.environ.get("HOLOSOMA_WORLD_TOPIC_UDP_PORT", "5005"))
# If enabled, use /world/robot_pose quaternion for base orientation (instead of IMU quaternion).
# Angular velocity still comes from IMU.
WORLD_TOPICS_USE_ROBOT_POSE_QUAT = os.environ.get("HOLOSOMA_WORLD_USE_ROBOT_POSE_QUAT", "0") == "1"

# Coordinate/frame conventions:
# - Incoming UDP world topics are in RDF axes (x=Right, y=Down, z=Forward) with quats in xyzw.
# - Internal code expects FLU world axes (x=Forward, y=Left, z=Up).
_Q_RDF_TO_FLU_WXYZ = np.array([[0.5, -0.5, 0.5, -0.5]], dtype=np.float32)  # wxyz
_Q_RDF_TO_FLU_INV_WXYZ = quat_inverse(_Q_RDF_TO_FLU_WXYZ)

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
        self.logger = logger
        self.use_joystick = use_joystick
        self.use_hands = use_hands
        self.robot_config = robot_config
        self.domain_id = domain_id
        self.interface_str = interface_str
        self.sdk_type = robot_config.sdk_type
        self.backend = None
        self._unitree_motor_order = None

        # Initialize gain levels for binding backend
        self._kp_level = 1.0
        self._kd_level = 1.0

        # Initialize sdk components
        self._init_sdk_components()

        # Object/robot world-state receivers
        self._object_state_sock = None
        self._object_state_cache = None
        self._robot_state_sock = None
        self._robot_state_cache = None
        self._table_command_sock = None
        self._table_command_addr = None
        self._table_command_warned = False
        # World topics receiver (external pose/velocity via UDP JSON)
        self._world_topics_sock = None
        self._world_robot_pose_cache = None  # [x,y,z,qx,qy,qz,qw] in FLU
        self._world_object_pose_cache = None  # [x,y,z,qx,qy,qz,qw] in FLU
        self._world_robot_twist_cache = None  # [vx,vy,vz,wx,wy,wz] in FLU
        self._world_robot_quat_warned = False

        if self.interface_str == "lo":
            self._init_object_state_receiver()
            self._init_robot_state_receiver()
            self._init_table_command_sender()
        elif self.interface_str == "enp132s0":
            self._init_world_topics_receiver()

    def _init_object_state_receiver(self):
        """Initialize UDP receiver for simulator object state (loopback only)."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((OBJECT_STATE_UDP_HOST, OBJECT_STATE_UDP_PORT))
            sock.setblocking(False)
            self._object_state_sock = sock
            self.logger.info(
                f"Object state UDP receiver enabled on {OBJECT_STATE_UDP_HOST}:{OBJECT_STATE_UDP_PORT}"
            )
        except OSError as exc:
            self.logger.warning(f"Failed to init object state UDP receiver: {exc}")
            self._object_state_sock = None

    def _init_robot_state_receiver(self):
        """Initialize UDP receiver for simulator robot state (loopback only)."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((ROBOT_STATE_UDP_HOST, ROBOT_STATE_UDP_PORT))
            sock.setblocking(False)
            self._robot_state_sock = sock
            self.logger.info(
                f"Robot state UDP receiver enabled on {ROBOT_STATE_UDP_HOST}:{ROBOT_STATE_UDP_PORT}"
            )
        except OSError as exc:
            self.logger.warning(f"Failed to init robot state UDP receiver: {exc}")
            self._robot_state_sock = None

    def _init_table_command_sender(self):
        """Initialize UDP sender for simulator table commands (loopback only)."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._table_command_sock = sock
            self._table_command_addr = (TABLE_COMMAND_UDP_HOST, TABLE_COMMAND_UDP_PORT)
            self.logger.info(
                f"Table command UDP sender enabled to {TABLE_COMMAND_UDP_HOST}:{TABLE_COMMAND_UDP_PORT}"
            )
        except OSError as exc:
            self.logger.warning(f"Failed to init table command UDP sender: {exc}")
            self._table_command_sock = None
            self._table_command_addr = None

    def _init_world_topics_receiver(self):
        """Initialize UDP receiver for external world topics (JSON packets)."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((WORLD_TOPICS_UDP_BIND_IP, WORLD_TOPICS_UDP_PORT))
            sock.setblocking(False)
            self._world_topics_sock = sock
            self.logger.info(
                f"World topics UDP receiver enabled on {WORLD_TOPICS_UDP_BIND_IP}:{WORLD_TOPICS_UDP_PORT}"
            )
        except OSError as exc:
            self.logger.warning(f"Failed to init world topics UDP receiver: {exc}")
            self._world_topics_sock = None

    @staticmethod
    def _rdf_to_flu_vec3(vec: np.ndarray) -> np.ndarray:
        """Convert a 3D vector from RDF (Right-Down-Forward) to FLU (Forward-Left-Up)."""
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        if v.size != 3:
            raise ValueError(f"Expected 3D vector, got shape {v.shape}")
        # [x_fwd, y_left, z_up] = [z_fwd, -x_right, -y_down]
        return np.array([v[2], -v[0], -v[1]], dtype=np.float32)

    @staticmethod
    def _rdf_to_flu_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
        """Convert quaternion from RDF basis to FLU basis. Input/output are xyzw."""
        q = np.asarray(quat_xyzw, dtype=np.float32).reshape(-1)
        if q.size != 4:
            raise ValueError(f"Expected quaternion xyzw, got shape {q.shape}")
        q_wxyz = xyzw_to_wxyz(q.reshape(1, 4))
        q_flu_wxyz = quat_mul(quat_mul(_Q_RDF_TO_FLU_WXYZ, q_wxyz), _Q_RDF_TO_FLU_INV_WXYZ)
        q_flu_xyzw = wxyz_to_xyzw(q_flu_wxyz)
        norm = np.linalg.norm(q_flu_xyzw, axis=1, keepdims=True)
        norm = np.where(norm == 0.0, 1.0, norm)
        q_flu_xyzw = q_flu_xyzw / norm
        return q_flu_xyzw.reshape(-1)

    def _poll_world_topics(self) -> None:
        """Drain pending world-topic packets and update caches (best-effort, non-blocking)."""
        if self._world_topics_sock is None:
            return
        while True:
            try:
                data, _addr = self._world_topics_sock.recvfrom(8192)
            except BlockingIOError:
                break
            if not data:
                continue
            try:
                packet = json.loads(data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            topic = str(packet.get("topic", ""))
            msg_type = str(packet.get("type", ""))

            if msg_type == "PoseStamped":
                pos_raw = packet.get("pos", None)
                if pos_raw is None:
                    continue
                try:
                    pos_flu = self._rdf_to_flu_vec3(np.asarray(pos_raw, dtype=np.float32))
                except ValueError:
                    continue

                if "robot_pose" in topic:
                    # Default: root orientation comes from IMU; only use UDP quat if explicitly enabled.
                    quat_flu_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                    if WORLD_TOPICS_USE_ROBOT_POSE_QUAT:
                        quat_raw = packet.get("quat_xyzw", packet.get("quat", None))
                        if quat_raw is None:
                            if not self._world_robot_quat_warned:
                                self.logger.warning("Robot pose quaternion missing in UDP packet; using identity.")
                                self._world_robot_quat_warned = True
                        else:
                            try:
                                quat_flu_xyzw = self._rdf_to_flu_quat_xyzw(np.asarray(quat_raw, dtype=np.float32))
                            except ValueError:
                                continue
                    pose = np.concatenate([pos_flu, quat_flu_xyzw], axis=0).astype(np.float32, copy=False)
                    self._world_robot_pose_cache = pose
                elif "object_pose" in topic:
                    quat_raw = packet.get("quat_xyzw", packet.get("quat", None))
                    if quat_raw is None:
                        continue
                    try:
                        quat_flu_xyzw = self._rdf_to_flu_quat_xyzw(np.asarray(quat_raw, dtype=np.float32))
                    except ValueError:
                        continue
                    pose = np.concatenate([pos_flu, quat_flu_xyzw], axis=0).astype(np.float32, copy=False)
                    self._world_object_pose_cache = pose

            elif msg_type == "TwistStamped":
                # Only use base linear velocity from UDP; angular velocity comes from robot IMU.
                lin_raw = packet.get("lin", None)
                if lin_raw is None:
                    continue
                try:
                    lin_flu = self._rdf_to_flu_vec3(np.asarray(lin_raw, dtype=np.float32))
                except ValueError:
                    continue

                twist = np.concatenate([lin_flu, np.zeros(3, dtype=np.float32)], axis=0).astype(np.float32, copy=False)
                if "robot_lin_vel" in topic or topic.endswith("/lin_vel"):
                    self._world_robot_twist_cache = twist

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

    def get_object_state(self):
        """Get latest object state [x,y,z,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz] if available."""
        if self._world_topics_sock is not None:
            self._poll_world_topics()
            if self._world_object_pose_cache is None:
                return None
            # Object velocity is not provided by the world topic bridge; fill with zeros.
            pos_quat = self._world_object_pose_cache
            return np.concatenate(
                [
                    pos_quat[:3],
                    pos_quat[3:7],
                    np.zeros(3, dtype=np.float32),  # lin vel
                    np.zeros(3, dtype=np.float32),  # ang vel
                ],
                axis=0,
            ).astype(np.float32, copy=False)
        if self._object_state_sock is None:
            return None
        while True:
            try:
                data, _addr = self._object_state_sock.recvfrom(13 * 4)
            except BlockingIOError:
                break
            if data:
                self._object_state_cache = np.frombuffer(data, dtype=np.float32).copy()
        return self._object_state_cache

    def get_robot_state(self):
        """Get latest robot state [x,y,z,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz] if available."""
        if self._world_topics_sock is not None:
            self._poll_world_topics()
            if self._world_robot_pose_cache is None:
                return None
            pos_quat = self._world_robot_pose_cache
            twist = self._world_robot_twist_cache
            if twist is None:
                twist = np.zeros(6, dtype=np.float32)
            return np.concatenate([pos_quat[:3], pos_quat[3:7], twist], axis=0).astype(np.float32, copy=False)
        if self._robot_state_sock is None:
            return None
        while True:
            try:
                data, _addr = self._robot_state_sock.recvfrom(13 * 4)
            except BlockingIOError:
                break
            if data:
                self._robot_state_cache = np.frombuffer(data, dtype=np.float32).copy()
        return self._robot_state_cache

    def get_object_pos_w(self):
        """Get object world position if available."""
        state = self.get_object_state()
        if state is None or state.size < 3:
            return None
        return state[:3].reshape(1, 3)

    def set_table_pose(self, pos_xyz, quat_wxyz=None):
        """Send table pose command to simulator bridge over UDP.

        Packet format (float32[8]):
          [cmd_id, px, py, pz, qw, qx, qy, qz]
        cmd_id=1 -> set table pose
        """
        if self._table_command_sock is None or self._table_command_addr is None:
            if not self._table_command_warned:
                self.logger.warning("Table command sender is not initialized; cannot set table pose.")
                self._table_command_warned = True
            return False

        pos = np.asarray(pos_xyz, dtype=np.float32).reshape(-1)
        if pos.size != 3:
            self.logger.warning(f"Invalid table position shape {pos.shape}; expected 3 values.")
            return False

        if quat_wxyz is None:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            quat = np.asarray(quat_wxyz, dtype=np.float32).reshape(-1)
            if quat.size != 4:
                self.logger.warning(f"Invalid table quaternion shape {quat.shape}; expected 4 values (wxyz).")
                return False
            norm = float(np.linalg.norm(quat))
            if norm <= 0.0:
                quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            else:
                quat = quat / norm

        packet = np.concatenate(
            [np.array([1.0], dtype=np.float32), pos.astype(np.float32, copy=False), quat.astype(np.float32, copy=False)],
            axis=0,
        )

        try:
            self._table_command_sock.sendto(packet.tobytes(), self._table_command_addr)
            return True
        except OSError as exc:
            self.logger.warning(f"Failed to send table pose command: {exc}")
            return False

    def request_remove_table(self):
        """Request simulator to remove/hide table.

        Packet format (float32[8]):
          [cmd_id, 0, 0, 0, 1, 0, 0, 0]
        cmd_id=2 -> remove table
        """
        if self._table_command_sock is None or self._table_command_addr is None:
            if not self._table_command_warned:
                self.logger.warning("Table command sender is not initialized; cannot request table removal.")
                self._table_command_warned = True
            return False

        packet = np.array([2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        try:
            self._table_command_sock.sendto(packet.tobytes(), self._table_command_addr)
            return True
        except OSError as exc:
            self.logger.warning(f"Failed to send remove-table command: {exc}")
            return False

    def _convert_binding_state_to_array(self):

        state = self.unitree_interface.read_low_state()
        # Compose array: [base_pos(3), quat(4), joint_pos(N), base_lin_vel(3), base_ang_vel(3), joint_vel(N), ...]
        base_pos = np.zeros(3)
        quat = np.array(state.imu.quat)
        motor_pos = np.array(state.motor.q)
        base_lin_vel = np.zeros(3)
        base_ang_vel = np.array(state.imu.omega)
        motor_vel = np.array(state.motor.dq)

        # Best-effort: populate base_pos and base_lin_vel from any available world-state source (sim2sim or UDP topics).
        robot_state = self.get_robot_state()
        if robot_state is not None and robot_state.size >= 3:
            base_pos = robot_state[:3].astype(np.float32, copy=False)
            if robot_state.size >= 10:
                base_lin_vel = robot_state[7:10].astype(np.float32, copy=False)
            if WORLD_TOPICS_USE_ROBOT_POSE_QUAT and robot_state.size >= 7:
                # robot_state quaternion is [qx,qy,qz,qw] in FLU.
                quat_xyzw = robot_state[3:7].astype(np.float32, copy=False).reshape(1, 4)
                quat = xyzw_to_wxyz(quat_xyzw).reshape(-1)

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

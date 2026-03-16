"""
ROS2 JointState bridge for G1 43-DOF robot.

Receives joint state data via UDP from read_dof_pos_real.py (Python 3.10 env)
and publishes to ROS2 /g1/joint_states topic using system Python 3.12.

Run with system Python (NOT conda env):
  source /opt/ros/jazzy/setup.bash
  /usr/bin/python3 ros2_joint_state_bridge.py

  or:
  python3 ros2_joint_state_bridge.py  (if system python3 is 3.12)
"""

import socket
import struct

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

BRIDGE_HOST = "127.0.0.1"
BRIDGE_PORT = 9877

JOINT_NAMES = [
    # Legs (0-11)
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # Waist (12-14)
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # Left arm (15-21)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    # Left hand (22-28)
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    # Right arm (29-35)
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    # Right hand (36-42)
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
]

# Must match read_dof_pos_real.py: timestamp(double) + pos(43f) + vel(43f)
_PACKET_FMT = "d" + "43f" + "43f"
_PACKET_SIZE = struct.calcsize(_PACKET_FMT)


class JointStateBridge(Node):
    def __init__(self, sock: socket.socket):
        super().__init__("g1_joint_state_bridge")
        self.sock = sock
        self.pub = self.create_publisher(JointState, "/g1/joint_states", 10)
        self.timer = self.create_timer(0.001, self.recv_and_publish)  # poll at 1kHz
        self.get_logger().info(
            f"Listening on UDP {BRIDGE_HOST}:{BRIDGE_PORT}, publishing /g1/joint_states"
        )

    def recv_and_publish(self):
        try:
            data, _ = self.sock.recvfrom(_PACKET_SIZE)
        except BlockingIOError:
            return  # no data yet

        if len(data) != _PACKET_SIZE:
            return

        unpacked = struct.unpack(_PACKET_FMT, data)
        # unpacked[0]: sender timestamp (unused, we use ROS time)
        pos = list(unpacked[1:44])
        vel = list(unpacked[44:87])

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = pos
        msg.velocity = vel
        msg.effort = []
        self.pub.publish(msg)


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((BRIDGE_HOST, BRIDGE_PORT))
    sock.setblocking(False)

    rclpy.init()
    node = JointStateBridge(sock)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        sock.close()


if __name__ == "__main__":
    main()

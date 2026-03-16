"""
Real-time joint position reader for G1 43-DOF robot.

Architecture (Python version mismatch workaround):
  holoinference_wc env (Python 3.10) reads joint states from robot SDK,
  then sends via UDP to a separate ROS2 bridge process (Python 3.12).

Usage:
  # Terminal only (no ROS2):
  python read_dof_pos_real.py

  # With ROS2 topic publishing (two terminals):
  Terminal 1: python read_dof_pos_real.py --ros
  Terminal 2: /usr/bin/python3 ros2_joint_state_bridge.py
              (or: python3 ros2_joint_state_bridge.py  # if system python is 3.12)

  # Check topic:
  ros2 topic echo /g1/joint_states
  ros2 topic hz /g1/joint_states
"""

import argparse
import socket
import struct
import time

import numpy as np

from holosoma_inference.config.config_values import robot
from holosoma_inference.sdk.interface_wrapper import InterfaceWrapper

# UDP bridge settings (must match ros2_joint_state_bridge.py)
BRIDGE_HOST = "127.0.0.1"
BRIDGE_PORT = 9877

# G1 43-DOF joint names (MuJoCo order, matches g1_43dof dof_names)
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

# UDP packet: timestamp(double) + pos(43 float32) + vel(43 float32) = 8 + 172 + 172 = 352 bytes
_PACKET_FMT = "d" + "43f" + "43f"


def run_ros_sender(interface: InterfaceWrapper, rate_hz: float):
    """Read joint state and send via UDP to ROS2 bridge process."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    period = 1.0 / rate_hz
    print(f"Sending joint states to {BRIDGE_HOST}:{BRIDGE_PORT} at {rate_hz} Hz")
    print("Make sure ros2_joint_state_bridge.py is running.\n")
    try:
        while True:
            t0 = time.monotonic()
            state = interface.get_full_state_43dof()
            dof_pos = state[0, 7:50].astype(np.float32)   # 43 joint positions
            dof_vel = state[0, 56:99].astype(np.float32)  # 43 joint velocities

            packet = struct.pack(_PACKET_FMT, time.time(), *dof_pos, *dof_vel)
            sock.sendto(packet, (BRIDGE_HOST, BRIDGE_PORT))

            elapsed = time.monotonic() - t0
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sock.close()


def run_terminal(interface: InterfaceWrapper, rate_hz: float):
    """Print joint positions to terminal in a loop."""
    period = 1.0 / rate_hz
    print(f"Reading joint positions at {rate_hz} Hz. Press Ctrl+C to stop.\n")
    try:
        while True:
            t0 = time.monotonic()
            state = interface.get_full_state_43dof()
            dof_pos = state[0, 7:50]  # 43 joint positions

            print("\033[H\033[J", end="")  # clear terminal
            print("=== G1 43-DOF Joint Positions ===\n")
            for i, (name, pos) in enumerate(zip(JOINT_NAMES, dof_pos)):
                print(f"  {i:2d}. {name:<35s}: {pos:8.4f} rad  ({np.degrees(pos):8.2f} deg)")

            elapsed = time.monotonic() - t0
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\nStopped.")


def main():
    parser = argparse.ArgumentParser(description="Read G1 43-DOF joint positions in real time.")
    parser.add_argument("--interface", default="enp132s0",
                        help="Network interface (default: enp132s0). Use 'lo' for sim.")
    parser.add_argument("--rate", type=float, default=50.0,
                        help="Read/publish rate in Hz (default: 50)")
    parser.add_argument("--ros", action="store_true",
                        help="Send joint states via UDP to ros2_joint_state_bridge.py")
    args = parser.parse_args()

    robot_config = robot.g1_43dof
    interface = InterfaceWrapper(
        robot_config=robot_config,
        interface_str=args.interface,
        use_joystick=False,
        use_hands=True,
    )

    if args.ros:
        run_ros_sender(interface, args.rate)
    else:
        run_terminal(interface, args.rate)


if __name__ == "__main__":
    main()

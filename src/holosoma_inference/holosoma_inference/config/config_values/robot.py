"""Default robot configurations for holosoma_inference.

This module provides pre-configured robot hardware and control parameters
for different robot types.
"""

from __future__ import annotations

from holosoma_inference.config.config_types.robot import RobotConfig

# =============================================================================
# G1 Robot Config
# =============================================================================

# fmt: off

g1_29dof = RobotConfig(
    # Identity
    robot_type="g1_29dof",
    robot="g1",

    # SDK Configuration
    sdk_type="unitree",
    motor_type="serial",
    message_type="HG",
    use_sensor=False,

    # Dimensions
    num_motors=29,
    num_joints=29,
    num_upper_body_joints=14,

    # Default Positions
    default_dof_angles=(
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # left leg
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # right leg
        0.0, 0.0, 0.0,  # waist
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,  # left arm
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,  # right arm
    ),
    default_motor_angles=(
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # left leg
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # right leg
        0.0, 0.0, 0.0,  # waist
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,  # left arm
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,  # right arm
    ),

    # Limits
    joint_pos_min=(
        -2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,  # left leg
        -2.5307, -2.9671, -2.7576, -0.087267, -0.87267, -0.2618,  # right leg
        -2.618, -0.52, -0.52,  # waist
        -3.0892, -1.5882, -2.618, -1.0472, -1.972222054, -1.61443, -1.61443,  # left arm
        -3.0892, -2.2515, -2.618, -1.0472, -1.972222054, -1.61443, -1.61443,  # right arm
    ),
    joint_pos_max=(
        2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618,  # left leg
        2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618,  # right leg
        2.618, 0.52, 0.52,  # waist
        2.6704, 2.2515, 2.618, 2.0944, 1.972222054, 1.61443, 1.61443,  # left arm
        2.6704, 1.5882, 2.618, 2.0944, 1.972222054, 1.61443, 1.61443,  # right arm
    ),
    joint_vel_limit=(
        32.0, 20.0, 32.0, 20.0, 37.0, 37.0,  # left leg
        32.0, 20.0, 32.0, 20.0, 37.0, 37.0,  # right leg
        32.0, 37.0, 37.0,  # waist
        37.0, 37.0, 37.0, 37.0, 37.0, 22.0, 22.0,  # left arm
        37.0, 37.0, 37.0, 37.0, 37.0, 22.0, 22.0,  # right arm
    ),
    motor_effort_limit=(
        88.0, 139.0, 88.0, 139.0, 50.0, 50.0,  # left leg
        88.0, 139.0, 88.0, 139.0, 50.0, 50.0,  # right leg
        88.0, 50.0, 50.0,  # waist
        25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,  # left arm
        25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,  # right arm
    ),

    # Mappings
    motor2joint=tuple(range(29)),  # Identity mapping
    joint2motor=tuple(range(29)),  # Identity mapping
    dof_names=(
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
        "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ),
    dof_names_upper_body=(
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
        "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ),
    dof_names_lower_body=(
        "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    ),

    # Link Names
    torso_link_name="torso_link",
    left_hand_link_name="left_rubber_hand",
    right_hand_link_name="right_rubber_hand",

    # Hand Configuration
    has_hands=False,

    # Unitree-Specific Constants
    unitree_legged_const={
        "HIGHLEVEL": 238,
        "LOWLEVEL": 255,
        "TRIGERLEVEL": 240,
        "PosStopF": 2146000000.0,
        "VelStopF": 16000.0,
        "MODE_MACHINE": 5,
        "MODE_PR": 0,
    },
    weak_motor_joint_index={
        "left_hip_yaw_joint": 0, "left_hip_roll_joint": 1, "left_hip_pitch_joint": 2,
        "left_knee_joint": 3, "left_ankle_pitch_joint": 4, "left_ankle_roll_joint": 5,
        "right_hip_yaw_joint": 6, "right_hip_roll_joint": 7, "right_hip_pitch_joint": 8,
        "right_knee_joint": 9, "right_ankle_pitch_joint": 10, "right_ankle_roll_joint": 11,
        "waist_yaw_joint": 12, "waist_roll_joint": 13, "waist_pitch_joint": 14,
        "left_shoulder_pitch_joint": 15, "left_shoulder_roll_joint": 16,
        "left_shoulder_yaw_joint": 17, "left_elbow_joint": 18,
        "left_wrist_roll_joint": 19, "left_wrist_pitch_joint": 20, "left_wrist_yaw_joint": 21,
        "right_shoulder_pitch_joint": 22, "right_shoulder_roll_joint": 23,
        "right_shoulder_yaw_joint": 24, "right_elbow_joint": 25,
        "right_wrist_roll_joint": 26, "right_wrist_pitch_joint": 27, "right_wrist_yaw_joint": 28,
    },
    motion={"body_name_ref": ["torso_link"]},
)


# =============================================================================
# T1 Robot Config
# =============================================================================

t1_29dof = RobotConfig(
    # Identity
    robot_type="t1_29dof",
    robot="t1",

    # SDK Configuration
    sdk_type="booster",  # T1 uses booster SDK
    motor_type="serial",
    message_type="HG",  # Using default
    use_sensor=False,

    # Dimensions
    num_motors=29,
    num_joints=29,
    num_upper_body_joints=16,  # T1 has 16 upper body joints (includes head)

    # Default Positions
    default_dof_angles=(
        0.0, 0.0,  # head (yaw, pitch)
        0.2, -1.35, 0.0, -0.5, 0.0, 0.0, 0.0,  # left arm
        0.2, 1.35, 0.0, 0.5, 0.0, 0.0, 0.0,  # right arm
        0.0,  # waist
        -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,  # left leg
        -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,  # right leg
    ),
    default_motor_angles=(
        0.0, 0.0,  # head
        0.2, -1.35, 0.0, -0.5, 0.0, 0.0, 0.0,  # left arm
        0.2, 1.35, 0.0, 0.5, 0.0, 0.0, 0.0,  # right arm
        0.0,  # waist
        -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,  # left leg
        -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,  # right leg
    ),

    # Limits
    joint_pos_min=(
        -1.57, -0.35,  # head
        -3.31, -1.74, -2.27, -2.27, -2.27, -2.27, -2.27,  # left arm
        -3.31, -1.57, -2.27, -2.27, -2.27, -2.27, -2.27,  # right arm
        -1.57,  # waist
        -1.8, -0.3, -1.0, 0.0, -0.87, -0.44,  # left leg
        -1.8, -1.57, -1.0, 0.0, -0.87, -0.44,  # right leg
    ),
    joint_pos_max=(
        1.57, 1.22,  # head
        1.22, 1.57, 2.27, 2.27, 2.27, 2.27, 2.27,  # left arm
        1.22, 1.74, 2.27, 2.27, 2.27, 2.27, 2.27,  # right arm
        1.57,  # waist
        1.57, 1.57, 1.0, 2.34, 0.35, 0.44,  # left leg
        1.57, 0.3, 1.0, 2.34, 0.35, 0.44,  # right leg
    ),
    joint_vel_limit=(
        12.56, 12.56,  # head
        18.84, 18.84, 18.84, 18.84, 18.84, 18.84, 18.84,  # left arm
        18.84, 18.84, 18.84, 18.84, 18.84, 18.84, 18.84,  # right arm
        10.88,  # waist
        12.5, 10.9, 10.9, 11.7, 18.8, 12.4,  # left leg
        12.5, 10.9, 10.9, 11.7, 18.8, 12.4,  # right leg
    ),
    motor_effort_limit=(
        7.0, 7.0,  # head
        18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0,  # left arm
        18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0,  # right arm
        30.0,  # waist
        45.0, 30.0, 30.0, 60.0, 12.0, 12.0,  # left leg
        45.0, 30.0, 30.0, 60.0, 12.0, 12.0,  # right leg
    ),

    # Mappings
    motor2joint=tuple(range(29)),  # Identity mapping
    joint2motor=tuple(range(29)),  # Identity mapping
    dof_names=(
        "AAHead_yaw", "Head_pitch",
        "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
        "Left_Wrist_Pitch", "Left_Wrist_Yaw", "Left_Hand_Roll",
        "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
        "Right_Wrist_Pitch", "Right_Wrist_Yaw", "Right_Hand_Roll",
        "Waist",
        "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
        "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
        "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
        "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
    ),
    dof_names_upper_body=(
        "AAHead_yaw", "Head_pitch",
        "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
        "Left_Wrist_Pitch", "Left_Wrist_Yaw", "Left_Hand_Roll",
        "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
        "Right_Wrist_Pitch", "Right_Wrist_Yaw", "Right_Hand_Roll",
    ),
    dof_names_lower_body=(
        "Waist",
        "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
        "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
        "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
        "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
    ),

    # Link Names
    torso_link_name="Trunk",
    left_hand_link_name=None,
    right_hand_link_name=None,

    # Set unitree-specific values to `None`
    unitree_legged_const=None,
    weak_motor_joint_index=None,
    motion=None,
)


# =============================================================================
# G1 Robot Config with Hands (43 DOF)
# =============================================================================

g1_43dof = RobotConfig(
    # Identity
    robot_type="g1_43dof",
    robot="g1",

    # SDK Configuration
    sdk_type="unitree",
    motor_type="serial",
    message_type="HG",
    use_sensor=False,

    # Dimensions
    num_motors=43,  # 29 body + 14 hand motors
    num_joints=43,  # 29 body + 14 hand joints
    num_upper_body_joints=28,  # 14 arms + 14 hands

    # Hand Configuration
    has_hands=True,
    num_hand_dofs=7,  # Per hand

    # Default Positions (43 DOF: legs, waist, left_arm, left_hand, right_arm, right_hand)
    # MuJoCo joint order
    default_dof_angles=(
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # left leg (0-5)
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # right leg (6-11)
        0.0, 0.0, 0.0,  # waist (12-14)
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,  # left arm (15-21)
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # left hand (22-28)
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,  # right arm (29-35)
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # right hand (36-42)
    ),
    # Body SDK motor order (29 motors): [0-21]legs+waist+left_arm, [22-28]right_arm
    # Hands are controlled via separate SDK interfaces with their own 0-6 indexing
    default_motor_angles=(
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # left leg (0-5)
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # right leg (6-11)
        0.0, 0.0, 0.0,  # waist (12-14)
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,  # left arm (15-21)
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,  # right arm (22-28)
    ),

    # Limits (43 DOF: legs, waist, left_arm, left_hand, right_arm, right_hand)
    joint_pos_min=(
        -2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,  # left leg (0-5)
        -2.5307, -2.9671, -2.7576, -0.087267, -0.87267, -0.2618,  # right leg (6-11)
        -2.618, -0.52, -0.52,  # waist (12-14)
        -3.0892, -1.5882, -2.618, -1.0472, -1.97222, -1.61443, -1.61443,  # left arm (15-21)
        -1.0472, -0.724312, 0.0, -1.5708, -1.74533, -1.5708, -1.74533,  # left hand (22-28)
        -3.0892, -2.2515, -2.618, -1.0472, -1.97222, -1.61443, -1.61443,  # right arm (29-35)
        -1.0472, -1.0472, -1.74533, 0.0, 0.0, 0.0, 0.0,  # right hand (36-42)
    ),
    joint_pos_max=(
        2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618,  # left leg (0-5)
        2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618,  # right leg (6-11)
        2.618, 0.52, 0.52,  # waist (12-14)
        2.6704, 2.2515, 2.618, 2.0944, 1.97222, 1.61443, 1.61443,  # left arm (15-21)
        1.0472, 1.0472, 1.74533, 0.0, 0.0, 0.0, 0.0,  # left hand (22-28)
        2.6704, 1.5882, 2.618, 2.0944, 1.97222, 1.61443, 1.61443,  # right arm (29-35)
        1.0472, 0.724312, 0.0, 1.5708, 1.74533, 1.5708, 1.74533,  # right hand (36-42)
    ),
    joint_vel_limit=(
        32.0, 20.0, 32.0, 20.0, 37.0, 37.0,  # left leg (0-5)
        32.0, 20.0, 32.0, 20.0, 37.0, 37.0,  # right leg (6-11)
        32.0, 37.0, 37.0,  # waist (12-14)
        37.0, 37.0, 37.0, 37.0, 37.0, 22.0, 22.0,  # left arm (15-21)
        10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,  # left hand (22-28)
        37.0, 37.0, 37.0, 37.0, 37.0, 22.0, 22.0,  # right arm (29-35)
        10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,  # right hand (36-42)
    ),
    motor_effort_limit=(
        88.0, 139.0, 88.0, 139.0, 50.0, 50.0,  # left leg (0-5)
        88.0, 139.0, 88.0, 139.0, 50.0, 50.0,  # right leg (6-11)
        88.0, 50.0, 50.0,  # waist (12-14)
        25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,  # left arm (15-21)
        2.45, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4,  # left hand (22-28)
        25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,  # right arm (29-35)
        2.45, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4,  # right hand (36-42)
    ),

    # Mappings for 43 DOF
    # Note: For robots with hands, num_joints=43 but hardware has:
    #   - 29 body motors via main SDK (indices 0-28)
    #   - 14 hand motors via separate hand SDK interfaces (7 per hand)
    # 
    # These mappings are 43-element arrays that map between:
    #   - Joint space (43 DOF): MuJoCo order with hands interleaved
    #   - Motor space (43 motors total, but split across interfaces)
    #
    # Joint order: [0-11]legs, [12-14]waist, [15-21]L_arm, [22-28]L_hand, [29-35]R_arm, [36-42]R_hand
    # Motor order: [0-21]legs+waist+L_arm, [22-28]R_arm (body SDK), [29-35]L_hand, [36-42]R_hand (hand SDK)
    motor2joint=(
        # Motors 0-21: legs + waist + left_arm -> Joints 0-21 (identity)
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        # Motors 22-28: right_arm (body SDK) -> Joints 29-35
        29, 30, 31, 32, 33, 34, 35,
        # Motors 29-35: left_hand (hand SDK) -> Joints 22-28
        22, 23, 24, 25, 26, 27, 28,
        # Motors 36-42: right_hand (hand SDK) -> Joints 36-42
        36, 37, 38, 39, 40, 41, 42,
    ),
    joint2motor=(
        # Joints 0-21: legs + waist + left_arm -> Motors 0-21 (identity)
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        # Joints 22-28: left_hand -> Motors 29-35 (hand SDK)
        29, 30, 31, 32, 33, 34, 35,
        # Joints 29-35: right_arm -> Motors 22-28 (body SDK)
        22, 23, 24, 25, 26, 27, 28,
        # Joints 36-42: right_hand -> Motors 36-42 (hand SDK)
        36, 37, 38, 39, 40, 41, 42,
    ),
    dof_names=(
        # Legs (0-11)
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        # Waist (12-14)
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        # Left arm (15-21)
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
        "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        # Left hand (22-28) - MuJoCo places hand after arm in body tree
        "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
        "left_hand_middle_0_joint", "left_hand_middle_1_joint",
        "left_hand_index_0_joint", "left_hand_index_1_joint",
        # Right arm (29-35) - MuJoCo places right arm after left hand in body tree
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        # Right hand (36-42)
        "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
        "right_hand_middle_0_joint", "right_hand_middle_1_joint",
        "right_hand_index_0_joint", "right_hand_index_1_joint",
    ),
    dof_names_upper_body=(
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
        "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ),
    dof_names_lower_body=(
        "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    ),

    # Link Names
    torso_link_name="torso_link",
    left_hand_link_name="left_hand",
    right_hand_link_name="right_hand",

    # Set unitree-specific values to `None`
    unitree_legged_const=None,
    weak_motor_joint_index=None,
    motion={"body_name_ref": ["torso_link"]},
)

# =============================================================================
# Default Configurations Dictionary
# =============================================================================

DEFAULTS = {
    "g1-29dof": g1_29dof,
    "g1-43dof": g1_43dof,
    "t1-29dof": t1_29dof,
}
"""Dictionary of all available robot configurations.

Keys use hyphen-case naming convention for CLI compatibility.
"""

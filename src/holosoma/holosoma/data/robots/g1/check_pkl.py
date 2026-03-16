import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=3, suppress=True)

file_path1 = "logs/sim2real/locomotion/locomotion_standing_20260316_121248_941500.pkl"
file_path2 = "logs/sim2real/locomotion/locomotion_standing_20260316_122552_744020.pkl"

# ── Joint name lists ──────────────────────────────────────────────────────────

_VEL_CMD_NAMES = ["vel_cmd/lin_x", "vel_cmd/lin_y", "vel_cmd/ang_z"]

_BODY_JOINT_NAMES = [
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]
_HAND_JOINT_NAMES = [
    "lh_thumb_0", "lh_thumb_1", "lh_thumb_2",
    "lh_middle_0", "lh_middle_1", "lh_index_0", "lh_index_1",
    "rh_thumb_0", "rh_thumb_1", "rh_thumb_2",
    "rh_middle_0", "rh_middle_1", "rh_index_0", "rh_index_1",
]
_ACTION_ORDER_NAMES = [
    "a/left_hip_pitch", "a/right_hip_pitch", "a/waist_yaw",
    "a/left_hip_roll", "a/right_hip_roll", "a/waist_roll",
    "a/left_hip_yaw", "a/right_hip_yaw", "a/waist_pitch",
    "a/left_knee", "a/right_knee",
    "a/left_shoulder_pitch", "a/right_shoulder_pitch",
    "a/left_ankle_pitch", "a/right_ankle_pitch",
    "a/left_shoulder_roll", "a/right_shoulder_roll",
    "a/left_ankle_roll", "a/right_ankle_roll",
    "a/left_shoulder_yaw", "a/right_shoulder_yaw",
    "a/left_elbow", "a/right_elbow",
    "a/left_wrist_roll", "a/right_wrist_roll",
    "a/left_wrist_pitch", "a/right_wrist_pitch",
    "a/left_wrist_yaw", "a/right_wrist_yaw",
    "a/lh_index_0", "a/lh_middle_0", "a/lh_thumb_0",
    "a/rh_index_0", "a/rh_middle_0", "a/rh_thumb_0",
    "a/lh_index_1", "a/lh_middle_1", "a/lh_thumb_1",
    "a/rh_index_1", "a/rh_middle_1", "a/rh_thumb_1",
    "a/lh_thumb_2", "a/rh_thumb_2",
]
_ANG_VEL_NAMES = ["base_ang_vel/x", "base_ang_vel/y", "base_ang_vel/z"]
_PROJ_GRAV_NAMES = ["proj_grav/x", "proj_grav/y", "proj_grav/z"]

# ── Observation layouts ───────────────────────────────────────────────────────
# locomotion_prior_43dof default:
#   vel_cmd(3) | body_pos(29) | body_vel(29) | ang_vel(3) | proj_grav(3) | hand_pos(14) | hand_vel(14) | actions(43)
OBS_NAMES_DEFAULT = (
    _VEL_CMD_NAMES
    + [f"body_pos/{n}" for n in _BODY_JOINT_NAMES]
    + [f"body_vel/{n}" for n in _BODY_JOINT_NAMES]
    + _ANG_VEL_NAMES
    + _PROJ_GRAV_NAMES
    + [f"hand_pos/{n}" for n in _HAND_JOINT_NAMES]
    + [f"hand_vel/{n}" for n in _HAND_JOINT_NAMES]
    + _ACTION_ORDER_NAMES
)  # 138 dims

# locomotion_prior_43dof --task.change_loco_order:
#   actions(43) | hand_pos(14) | hand_vel(14) | body_pos(29) | body_vel(29) | ang_vel(3) | proj_grav(3) | vel_cmd(3)
OBS_NAMES_CHANGE_ORDER = (
    _ACTION_ORDER_NAMES
    + [f"hand_pos/{n}" for n in _HAND_JOINT_NAMES]
    + [f"hand_vel/{n}" for n in _HAND_JOINT_NAMES]
    + [f"body_pos/{n}" for n in _BODY_JOINT_NAMES]
    + [f"body_vel/{n}" for n in _BODY_JOINT_NAMES]
    + _ANG_VEL_NAMES
    + _PROJ_GRAV_NAMES
    + _VEL_CMD_NAMES
)  # 138 dims

OBS_LAYOUTS = {
    "default": OBS_NAMES_DEFAULT,
    "change_loco_order": OBS_NAMES_CHANGE_ORDER,
}

# ── DOF names (scaled_policy_action order = DOF_NAMES config order) ───────────
DOF_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "left_hand_thumb_0", "left_hand_thumb_1", "left_hand_thumb_2",
    "left_hand_middle_0", "left_hand_middle_1",
    "left_hand_index_0", "left_hand_index_1",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
    "right_hand_thumb_0", "right_hand_thumb_1", "right_hand_thumb_2",
    "right_hand_middle_0", "right_hand_middle_1",
    "right_hand_index_0", "right_hand_index_1",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_and_summarize(file_path, label):
    print(f"\n{'='*60}")
    print(f"[{label}] {file_path}")
    print(f"  Exists: {os.path.exists(file_path)}")
    if not os.path.exists(file_path):
        print("  File not found, skipping.")
        return None
    print(f"  Size: {os.path.getsize(file_path)} bytes")
    with open(file_path, "rb") as f:
        head = f.read(16)
        print(f"  HEAD: {head}  HEX: {head.hex()}")
    data = joblib.load(file_path)
    print(f"  Loaded type: {type(data)}")
    if isinstance(data, dict):
        print("  Per-key summary:")
        for key, value in data.items():
            summary = {"type": type(value).__name__}
            try:
                if hasattr(value, "shape"):
                    summary["shape"] = value.shape
                elif isinstance(value, (list, tuple)):
                    summary["len"] = len(value)
                elif isinstance(value, dict):
                    summary["keys"] = list(value.keys())[:10]
            except Exception as exc:
                summary["error"] = f"{type(exc).__name__}: {exc}"
            print(f"    - {key}: {summary}")
    return data


# ── Plot functions ────────────────────────────────────────────────────────────

def compare_joint_action(data1, data2, joint_indices):
    actions1 = data1["scaled_policy_action"]  # (T1, 43)
    actions2 = data2["scaled_policy_action"]  # (T2, 43)

    n_joints = len(joint_indices)
    fig, axes = plt.subplots(n_joints, 1, figsize=(12, 3 * n_joints), squeeze=False)
    fig.suptitle("scaled_policy_action comparison\nsim2sim (blue) vs sim2real (orange)", fontsize=13)

    for row, j in enumerate(joint_indices):
        ax = axes[row][0]
        name = DOF_NAMES[j] if j < len(DOF_NAMES) else f"joint_{j}"
        ax.plot(np.arange(len(actions1)), actions1[:, j], label="sim2sim", color="steelblue", linewidth=1.2)
        ax.plot(np.arange(len(actions2)), actions2[:, j], label="sim2real", color="darkorange", linewidth=1.2)
        ax.set_title(f"[{j}] {name}", fontsize=10)
        ax.set_xlabel("timestep")
        ax.set_ylabel("action (rad)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def compare_observation(data1, data2, obs_indices, obs_names):
    obs1 = data1["observation"]  # (T1, D)
    obs2 = data2["observation"]  # (T2, D)

    n = len(obs_indices)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), squeeze=False)
    fig.suptitle("observation comparison\nsim2sim (blue) vs sim2real (orange)", fontsize=13)

    for row, i in enumerate(obs_indices):
        ax = axes[row][0]
        name = obs_names[i] if i < len(obs_names) else f"obs_{i}"
        ax.plot(np.arange(len(obs1)), obs1[:, i], label="sim2sim", color="steelblue", linewidth=1.2)
        ax.plot(np.arange(len(obs2)), obs2[:, i], label="sim2real", color="darkorange", linewidth=1.2)
        ax.set_title(f"[{i}] {name}", fontsize=10)
        ax.set_xlabel("timestep")
        ax.set_ylabel("value")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def observation_error(data1, data2, timestep, obs_names):
    obs1 = data1["observation"]
    obs2 = data2["observation"]
    max_t = min(len(obs1), len(obs2)) - 1
    if timestep < 0 or timestep > max_t:
        print(f"Timestep {timestep} out of range. Valid range: 0 ~ {max_t}")
        return
    diff = np.abs(obs1[timestep] - obs2[timestep])
    top3_idx = np.argsort(diff)[-3:][::-1]
    print(f"\nTimestep {timestep} — top-3 abs diff:")
    for rank, i in enumerate(top3_idx):
        name = obs_names[i] if i < len(obs_names) else f"obs_{i}"
        print(f"  #{rank+1}  [{i:3d}] {name:45s}  sim2sim={obs1[timestep,i]:.4f}  sim2real={obs2[timestep,i]:.4f}  diff={diff[i]:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def add_obs_layout_arg(p):
    p.add_argument(
        "--obs-layout",
        choices=list(OBS_LAYOUTS.keys()),
        default="change_loco_order",
        help="Observation layout: 'default' (g1_43dof_loco_prior without flag) or "
             "'change_loco_order' (with --task.change_loco_order, default for this script)",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    p_action = subparsers.add_parser("compare_joint_action", help="Compare scaled_policy_action")
    p_action.add_argument(
        "--joints", "-j",
        type=int, nargs="+", default=list(range(12)),
        help="Joint indices in DOF_NAMES order (default: 0-11, leg joints)",
    )

    p_obs = subparsers.add_parser("compare_observation", help="Compare observation dims")
    p_obs.add_argument(
        "--dims", "-d",
        type=int, nargs="+", default=list(range(3)),
        help="Observation dim indices (default: 0-2)",
    )
    add_obs_layout_arg(p_obs)

    p_err = subparsers.add_parser("observation_error", help="Show top-3 abs diff dims at a timestep")
    p_err.add_argument("--timestep", "-t", type=int, required=True, help="Timestep index")
    add_obs_layout_arg(p_err)

    args = parser.parse_args()

    data1 = load_and_summarize(file_path1, "data1 (sim2sim)")
    data2 = load_and_summarize(file_path2, "data2 (sim2real)")

    if args.command == "compare_joint_action":
        if data1 is None or data2 is None:
            print("One or both files could not be loaded.")
        else:
            compare_joint_action(data1, data2, args.joints)

    elif args.command == "compare_observation":
        if data1 is None or data2 is None:
            print("One or both files could not be loaded.")
        else:
            obs_names = OBS_LAYOUTS[args.obs_layout]
            compare_observation(data1, data2, args.dims, obs_names)

    elif args.command == "observation_error":
        if data1 is None or data2 is None:
            print("One or both files could not be loaded.")
        else:
            obs_names = OBS_LAYOUTS[args.obs_layout]
            observation_error(data1, data2, args.timestep, obs_names)

    else:
        breakpoint()

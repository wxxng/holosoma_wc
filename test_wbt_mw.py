#!/usr/bin/env python3
"""
MuJoCo whole-body motion tracking test using master_policy.onnx.

Loads a motion clip PKL and runs the master_policy (29-DOF body) to track it.
The motion command sequence is built dynamically from the current + 9 future
frames of the motion clip, so the policy tracks the full trajectory (not just
stabilizes at the first frame).

Usage:
    python test_wbt_mw.py --pkl <motion_clip.pkl> [--clip-key <key>]
    python test_wbt_mw.py --pkl <motion_clip.pkl> --record
"""

import argparse
from pathlib import Path

import joblib
import mujoco
import numpy as np
import onnxruntime

from g1_robot_common import (
    DOF_NAMES, DEFAULT_DOF_ANGLES,
    ACTION_ORDER_43DOF, ACTION_SCALE, ACTION_OFFSET, ACTION_CLIP_MIN, ACTION_CLIP_MAX,
    HAND_JOINT_NAMES,
    name_indices, mj_hinge_addrs, mj_actuator_ids, apply_pd_control,
    quat_rotate_inverse, SimpleVideoRecorder, run_mujoco_loop,
)

np.set_printoptions(precision=4, suppress=True)

# ── master_policy joint orderings ──────────────────────────────────────────

# Stabilization (master_policy) observation input order (29 body joints, waist-first)
STABILIZATION_OBS_ORDER_29DOF = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Stabilization (master_policy) action output order (29 body joints)
STABILIZATION_ACTION_ORDER_29DOF = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]

# Motion command sequence joint order (body-only, 29 DOF)
G1_MOTION_JOINT_NAMES_29 = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# 43-DOF DEX3 motion clip joint order
G1_DEX3_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
]


# ═══════════════════════════════════════════════════════════════════════════
# Motion clip loading
# ═══════════════════════════════════════════════════════════════════════════

def load_motion_clip(pkl_path: str, clip_key: str | None = None) -> dict:
    """Load a motion clip PKL and extract data needed for master_policy tracking.

    The PKL must be a dict mapping clip keys to clip dicts containing at least
    ``dof_pos`` (T x N_DOF) and ``fps``.  N_DOF is either 29 (G1_MOTION_JOINT_NAMES_29)
    or 43 (G1_DEX3_JOINT_NAMES).

    Returns dict with:
        clip_key      : str - the clip key used
        motion_fps    : float
        motion_length : int - number of frames
        motion_dof_pos_29 : np.ndarray [T, 29] - body joints in G1_MOTION_JOINT_NAMES_29 order
        motion_dof_vel_29 : np.ndarray [T, 29] - body joint velocities (finite difference)
        init_q        : np.ndarray [43] - first-frame joint targets in DOF_NAMES config order
        hand_target_q : np.ndarray [T, 14] or None - hand targets in HAND_JOINT_NAMES order
                        (None if motion is 29-DOF body-only)
    """
    motion_data = joblib.load(pkl_path)
    if not isinstance(motion_data, dict):
        raise TypeError(f"Motion PKL must be a dict of clips, got {type(motion_data)}")

    if clip_key is None:
        clip_key = next(iter(motion_data.keys()))
        print(f"No clip_key specified, using first clip: '{clip_key}'")
    elif clip_key not in motion_data:
        available = list(motion_data.keys())
        raise KeyError(
            f"clip_key '{clip_key}' not found. "
            f"Available: {available[:10]}{' ...' if len(available) > 10 else ''}"
        )

    clip = motion_data[clip_key]
    if not isinstance(clip, dict):
        raise TypeError(f"Clip '{clip_key}' must be a dict, got {type(clip)}")

    motion_fps = float(clip["fps"])
    motion_dof_pos = np.asarray(clip["dof_pos"], dtype=np.float32)  # [T, N_DOF]
    motion_length = motion_dof_pos.shape[0]
    n_dof = motion_dof_pos.shape[1]

    print(f"Motion clip '{clip_key}': {motion_length} frames @ {motion_fps}Hz, {n_dof} DOF")

    if n_dof == 43:
        motion_joint_names = G1_DEX3_JOINT_NAMES
    elif n_dof == 29:
        motion_joint_names = G1_MOTION_JOINT_NAMES_29
    else:
        raise ValueError(f"Unexpected motion DOF count: {n_dof} (expected 29 or 43)")

    # ── Body joints in G1_MOTION_JOINT_NAMES_29 order ─────────────────────
    body_in_motion = name_indices(G1_MOTION_JOINT_NAMES_29, motion_joint_names)
    motion_dof_pos_29 = motion_dof_pos[:, body_in_motion]  # [T, 29]

    # Finite-difference velocities (body, 29-DOF)
    dof_pos_next = np.roll(motion_dof_pos_29, -1, axis=0)
    dof_pos_next[-1] = motion_dof_pos_29[-1]
    motion_dof_vel_29 = (dof_pos_next - motion_dof_pos_29) * motion_fps
    motion_dof_vel_29[-1] = 0.0

    # ── Initial joint configuration in DOF_NAMES (config) order ───────────
    # Find each motion joint's position within DOF_NAMES (config order)
    init_q = DEFAULT_DOF_ANGLES.copy()
    config_idx = name_indices(motion_joint_names, DOF_NAMES)  # (N_DOF,) indices into DOF_NAMES
    init_q[config_idx] = motion_dof_pos[0]

    # ── Hand targets (only for 43-DOF clips) ──────────────────────────────
    hand_target_q = None
    if n_dof == 43:
        hand_in_motion = name_indices(HAND_JOINT_NAMES, motion_joint_names)
        hand_target_q = motion_dof_pos[:, hand_in_motion]  # [T, 14] in HAND_JOINT_NAMES order

    return {
        "clip_key": clip_key,
        "motion_fps": motion_fps,
        "motion_length": motion_length,
        "motion_dof_pos_29": motion_dof_pos_29,
        "motion_dof_vel_29": motion_dof_vel_29,
        "init_q": init_q,
        "hand_target_q": hand_target_q,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Observation and action building
# ═══════════════════════════════════════════════════════════════════════════

def build_motion_cmd_seq(
    motion_dof_pos_29: np.ndarray,
    motion_dof_vel_29: np.ndarray,
    timestep: int,
    motion_length: int,
) -> np.ndarray:
    """Build the 10-frame motion command sequence starting at ``timestep``.

    Returns np.ndarray [1, 580]: 10 frames × 58 dims (29 pos + 29 vel),
    in G1_MOTION_JOINT_NAMES_29 order.  Frames are clamped at the end of
    the clip.
    """
    indices = np.clip(np.arange(timestep, timestep + 10), 0, motion_length - 1)
    pos_seq = motion_dof_pos_29[indices]  # [10, 29]
    vel_seq = motion_dof_vel_29[indices]  # [10, 29]
    cmd_seq = np.concatenate([pos_seq, vel_seq], axis=1)  # [10, 58]
    return cmd_seq.reshape(1, -1).astype(np.float32)  # [1, 580]


def build_master_obs(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    base_quat_wxyz: np.ndarray,
    base_ang_vel: np.ndarray,
    last_action_29: np.ndarray,
    motion_cmd_seq: np.ndarray,
    obs_idx: np.ndarray,
) -> np.ndarray:
    """Build the 673-dim master_policy observation vector.

    Obs layout:
        dof_pos_rel  (29): joint_pos[obs_idx] - DEFAULT_DOF_ANGLES[obs_idx]
        dof_vel      (29): joint_vel[obs_idx]
        base_ang_vel  (3): angular velocity in body frame
        proj_grav     (3): projected gravity in body frame
        last_action  (29): previous raw action (STABILIZATION_ACTION_ORDER)
        motion_cmd_seq(580): 10 × (pos + vel) in G1_MOTION_JOINT_NAMES_29 order

    Total: 29+29+3+3+29+580 = 673
    """
    dof_pos_rel = joint_pos[obs_idx] - DEFAULT_DOF_ANGLES[obs_idx]
    dof_vel = joint_vel[obs_idx]
    proj_grav = quat_rotate_inverse(base_quat_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float32))

    obs = np.concatenate([
        dof_pos_rel,
        dof_vel,
        base_ang_vel,
        proj_grav,
        last_action_29,
        motion_cmd_seq.flatten(),
    ])
    return obs.reshape(1, -1).astype(np.float32)  # [1, 673]


# ═══════════════════════════════════════════════════════════════════════════
# Main simulation
# ═══════════════════════════════════════════════════════════════════════════

def run(args):
    root = Path(__file__).parent
    xml_path = args.xml or str(root / "src/holosoma/holosoma/data/robots/g1/g1_43dof.xml")
    policy_path = args.policy or str(
        root / "src/holosoma_inference/holosoma_inference/models/wbt/base/master_policy.onnx"
    )
    pkl_path = args.pkl

    # ── Load motion clip ───────────────────────────────────────────────────
    motion = load_motion_clip(pkl_path, args.clip_key)
    motion_fps = motion["motion_fps"]
    motion_length = motion["motion_length"]
    motion_dof_pos_29 = motion["motion_dof_pos_29"]
    motion_dof_vel_29 = motion["motion_dof_vel_29"]
    hand_target_q = motion["hand_target_q"]  # [T, 14] or None

    # ── Load ONNX policy ───────────────────────────────────────────────────
    print(f"Loading policy: {policy_path}")
    session = onnxruntime.InferenceSession(policy_path)
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    print(f"  Input:  {inp.name} {inp.shape}")
    print(f"  Output: {out.name} {out.shape}")

    # ── Load MuJoCo model ──────────────────────────────────────────────────
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = 1.0 / args.sim_hz

    if args.no_self_collision:
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        for i in range(model.ngeom):
            if i != floor_id and model.geom_contype[i] > 0:
                model.geom_conaffinity[i] = 0
        print("Self-collision disabled (robot geom conaffinity set to 0; explicit foot-floor pairs preserved).")

    data = mujoco.MjData(model)
    sim_dt = float(model.opt.timestep)
    steps_per_policy = args.sim_hz // args.policy_hz
    policy_dt = sim_dt * steps_per_policy
    print(f"Model: {model.nq} qpos, {model.nv} qvel, {model.nu} actuators, dt={sim_dt}s")
    print(f"Physics: {1/sim_dt:.0f}Hz | Policy: {1/policy_dt:.0f}Hz ({steps_per_policy} substeps)")

    dof_qpos_addrs, dof_qvel_addrs = mj_hinge_addrs(model, DOF_NAMES)
    actuator_ids = mj_actuator_ids(model, DOF_NAMES)

    # ── Precompute index maps ──────────────────────────────────────────────
    obs_idx = name_indices(STABILIZATION_OBS_ORDER_29DOF, DOF_NAMES)       # (29,) into config
    act_idx = name_indices(STABILIZATION_ACTION_ORDER_29DOF, DOF_NAMES)    # (29,) into config
    stab_in_43 = name_indices(STABILIZATION_ACTION_ORDER_29DOF, ACTION_ORDER_43DOF)  # (29,)
    act_scale_29 = ACTION_SCALE[stab_in_43]
    act_offset_29 = ACTION_OFFSET[stab_in_43]
    act_clip_min_29 = ACTION_CLIP_MIN[stab_in_43]
    act_clip_max_29 = ACTION_CLIP_MAX[stab_in_43]
    hand_idx = name_indices(HAND_JOINT_NAMES, DOF_NAMES)  # (14,) into config

    # ── Initialize robot pose ──────────────────────────────────────────────
    data.qpos[dof_qpos_addrs] = motion["init_q"]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    print(f"Initialized from motion first frame.")

    # ── Video recorder ────────────────────────────────────────────────────
    video_recorder = None
    if args.record or args.offscreen:
        video_recorder = SimpleVideoRecorder(
            model,
            name="test_wbt_mw",
            camera_pos=tuple(args.camera_pos),
            camera_target=tuple(args.camera_target),
            width=args.video_width,
            height=args.video_height,
            save_dir=args.video_dir,
            output_format=args.video_format,
        )
        print(f"Video recording enabled: {args.video_width}x{args.video_height}")

    # ── Simulation state ───────────────────────────────────────────────────
    last_action_29 = np.zeros(29, dtype=np.float32)
    target_q = motion["init_q"].copy()
    policy_step = 0
    motion_timestep = 0
    motion_time_acc = 0.0

    print(f"\nStarting WBT motion tracking.")
    print(f"Motion: {motion_length} frames @ {motion_fps}Hz = {motion_length/motion_fps:.2f}s")

    def _on_policy_step():
        nonlocal last_action_29, target_q, motion_time_acc, motion_timestep, policy_step
        joint_pos = data.qpos[dof_qpos_addrs].astype(np.float32)
        joint_vel = data.qvel[dof_qvel_addrs].astype(np.float32)
        base_quat_wxyz = data.qpos[3:7].astype(np.float32)
        base_ang_vel = data.qvel[3:6].astype(np.float32)  # already local/body frame

        motion_cmd_seq = build_motion_cmd_seq(
            motion_dof_pos_29, motion_dof_vel_29, motion_timestep, motion_length,
        )
        obs = build_master_obs(
            joint_pos, joint_vel, base_quat_wxyz, base_ang_vel,
            last_action_29, motion_cmd_seq, obs_idx,
        )
        raw_action_29 = session.run(None, {inp.name: obs})[0][0]
        # breakpoint()
        last_action_29 = raw_action_29.copy()

        scaled_29 = np.clip(
            raw_action_29 * act_scale_29 + act_offset_29,
            act_clip_min_29, act_clip_max_29,
        )
        target_q = joint_pos.copy()
        target_q[act_idx] = scaled_29

        if hand_target_q is not None:
            t_clamp = min(motion_timestep, motion_length - 1)
            target_q[hand_idx] = hand_target_q[t_clamp]

        motion_time_acc += policy_dt
        motion_timestep = min(int(motion_time_acc * motion_fps), motion_length - 1)
        policy_step += 1

        if policy_step % 50 == 0:
            base_z = float(data.qpos[2])
            print(
                f"  [t={motion_time_acc:6.2f}s] policy_step={policy_step} "
                f"motion_t={motion_timestep}/{motion_length}  base_z={base_z:.3f}m",
                flush=True,
            )

    def _apply_ctrl(d):
        apply_pd_control(d, target_q, dof_qpos_addrs, dof_qvel_addrs, actuator_ids)

    def _on_reset():
        nonlocal last_action_29, target_q, motion_timestep, motion_time_acc, policy_step
        last_action_29 = np.zeros(29, dtype=np.float32)
        target_q = motion["init_q"].copy()
        motion_timestep = 0
        motion_time_acc = 0.0
        policy_step = 0
        data.qpos[dof_qpos_addrs] = motion["init_q"]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        print(f"\n  [Reset] Restarted from motion frame 0", flush=True)

    def _should_stop():
        return motion_timestep >= motion_length - 1

    try:
        run_mujoco_loop(
            model, data,
            sim_dt=sim_dt,
            steps_per_policy=steps_per_policy,
            on_policy_step=_on_policy_step,
            apply_ctrl=_apply_ctrl,
            on_reset=_on_reset,
            should_stop=_should_stop if args.offscreen else None,
            video_recorder=video_recorder,
            offscreen=args.offscreen,
        )
    finally:
        if video_recorder is not None:
            video_recorder.save(fps=1.0 / policy_dt)
            video_recorder.cleanup()

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="WBT motion tracking with master_policy.onnx in MuJoCo"
    )
    parser.add_argument(
        "--pkl",
        required=True,
        help="Path to motion clip PKL (dict of {clip_key: {dof_pos, fps, ...}})",
    )
    parser.add_argument(
        "--clip-key", default=None,
        help="Clip key within the PKL (default: first key)",
    )
    parser.add_argument("--xml", default=None, help="MuJoCo XML path (default: g1_43dof.xml)")
    parser.add_argument(
        "--policy", default=None,
        help="Path to master_policy.onnx (default: models/wbt/base/master_policy.onnx)",
    )
    parser.add_argument("--sim-hz", type=int, default=200, help="Simulation frequency in Hz (default: 200)")
    parser.add_argument("--policy-hz", type=int, default=50, help="Policy frequency in Hz (default: 50)")
    parser.add_argument("--no-self-collision", action="store_true", help="Disable robot self-collision (preserves foot-floor contact pairs).")
    parser.add_argument("--record", action="store_true", help="Record video while showing viewer.")
    parser.add_argument("--offscreen", action="store_true", help="Run headless (no viewer window), record video, stop when motion ends.")
    parser.add_argument("--video-dir", type=str, default="logs/videos")
    parser.add_argument("--video-width", type=int, default=1920)
    parser.add_argument("--video-height", type=int, default=1080)
    parser.add_argument("--video-format", type=str, default="h264", choices=["h264", "mp4"])
    parser.add_argument(
        "--camera-pos", type=float, nargs=3, default=[-2.0, 0.0, 1.5],
        help="Camera position [x y z]",
    )
    parser.add_argument(
        "--camera-target", type=float, nargs=3, default=[1.0, 0.0, 1.0],
        help="Camera target [x y z]",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
MuJoCo locomotion state replay + optional policy inference test.

Mode 1 (default): Replay pkl trajectory via PD control.
Mode 2 (--infer):  Replay pkl trajectory, then switch to ONNX policy inference.

Usage:
    python test_loco.py                          # replay only
    python test_loco.py --infer                   # replay then infer
    python test_loco.py --loop                    # loop replay
    python test_loco.py --pkl path/to/other.pkl   # custom pkl
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import joblib
import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime
import torch

from g1_robot_common import (
    DOF_NAMES, DEFAULT_DOF_ANGLES,
    ACTION_ORDER_43DOF, ACTION_SCALE, ACTION_OFFSET, ACTION_CLIP_MIN, ACTION_CLIP_MAX,
    BODY_JOINT_NAMES, HAND_JOINT_NAMES,
    name_indices, mj_hinge_addrs, mj_actuator_ids, apply_pd_control,
    quat_rotate_inverse, SimpleVideoRecorder, run_mujoco_loop,
)

np.set_printoptions(precision=4, suppress=True)

# ── Precomputed index maps ─────────────────────────────────────────────────
BODY_INDICES = name_indices(BODY_JOINT_NAMES, DOF_NAMES)           # (29,)
HAND_INDICES = name_indices(HAND_JOINT_NAMES, DOF_NAMES)           # (14,)
ROBOT_STATE_TO_BODY_IDS = name_indices(BODY_JOINT_NAMES, ACTION_ORDER_43DOF)
ROBOT_STATE_TO_HAND_IDS = name_indices(HAND_JOINT_NAMES, ACTION_ORDER_43DOF)


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def to_numpy(x):
    """Convert tensor or array to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().squeeze()
    return np.asarray(x).squeeze()


def load_trajectory(pkl_path):
    """Load pkl and extract per-step robot_state + obs_dict command."""
    data = joblib.load(pkl_path)
    print(f"Loaded {len(data)} steps from {pkl_path}")

    traj = []
    obs_dict = {key: [] for key in data[0]["obs_dict"].keys()}
    actions = []
    observations = []

    # obs_dict : "task" / "proprio_hand" / "proprio_body" / "command"
    # obs :
    #  - "command" : 3
    #  - "proprio_body" : 29 + 29 + 3 + 3  = 64
    #  - "proprio_hand" : 14 + 14 = 28
    #  - "task" : 43

    for step in data:
        rs = step["robot_state"]
        entry = {
            "root_pose": to_numpy(rs["root_pose_wxyz"]),
            "root_lin_vel": to_numpy(rs["root_lin_vel"]),
            "root_ang_vel": to_numpy(rs["root_ang_vel"]),
            "joint_pos": to_numpy(rs["joint_pos"]),
            "joint_vel": to_numpy(rs["joint_vel"]),
            "joint_order": rs["joint_order"],
        }
        # Extract raw action (action order, before scale/offset)
        if "actions" in step:
            entry["action"] = to_numpy(step["actions"])
        # Extract processed action (action order, after scale/offset/clip) if available
        if "processed_actions" in step:
            entry["processed_action"] = to_numpy(step["processed_actions"])
        # Extract velocity command if available
        if "obs_dict" in step and "command" in step["obs_dict"]:
            cmd = step["obs_dict"]["command"]
            entry["command"] = to_numpy(cmd) if not isinstance(cmd, np.ndarray) else cmd.copy()
        traj.append(entry)
        for key, val in step["obs_dict"].items():
            obs_dict[key].append(val)

        assert np.max(abs(step["observations"][:3] - step["obs_dict"]["command"])) < 1e-4
        assert np.max(abs(step["observations"][3:67] - step["obs_dict"]["proprio_body"])) < 1e-4
        assert np.max(abs(step["observations"][67:95] - step["obs_dict"]["proprio_hand"])) < 1e-4
        assert np.max(abs(step["observations"][95:] - step["obs_dict"]["task"])) < 1e-4

        observations.append(step["observations"])
        actions.append(step["actions"])


    obs_dict = {key: np.array(val) for key, val in obs_dict.items()}
    observations = np.array(observations)
    actions = np.array(actions)
    return traj, obs_dict, actions, observations


def build_reorder_map(pkl_joint_order, mj_dof_names):
    """Build index mapping from pkl joint order to MuJoCo config order.
    Returns array where mj_ordered[reorder[i]] = pkl_data[i].
    """
    name_to_mj_idx = {name: idx for idx, name in enumerate(mj_dof_names)}
    reorder = np.zeros(len(pkl_joint_order), dtype=np.int64)
    for pkl_idx, name in enumerate(pkl_joint_order):
        if name not in name_to_mj_idx:
            raise KeyError(f"PKL joint '{name}' not found in MuJoCo DOF_NAMES")
        reorder[pkl_idx] = name_to_mj_idx[name]
    return reorder


def reorder_to_mj(pkl_data, reorder_map):
    """Reorder data from pkl/action joint order to MuJoCo config order."""
    mj_data = np.zeros(len(reorder_map), dtype=np.float32)
    mj_data[reorder_map] = pkl_data
    return mj_data


# ═══════════════════════════════════════════════════════════════════════════
# Policy inference
# ═══════════════════════════════════════════════════════════════════════════

def build_action_to_config_map():
    """Build mapping from ACTION_ORDER_43DOF to DOF_NAMES (config order).
    action_config[config_idx] = action_order[action_idx]
    """
    return build_reorder_map(ACTION_ORDER_43DOF, DOF_NAMES)


def process_action(raw_action, action_to_config_map):
    """Process raw policy output: scale + offset + clip, then reorder to config order.

    Args:
        raw_action: (43,) in action order
        action_to_config_map: mapping from action order to config order
    Returns:
        target_q: (43,) in config order (absolute joint targets)
    """
    # Scale + offset (in action order)
    action = raw_action * ACTION_SCALE + ACTION_OFFSET
    # Clip
    action = np.clip(action, ACTION_CLIP_MIN, ACTION_CLIP_MAX)
    # Reorder to config order
    return reorder_to_mj(action, action_to_config_map)


def build_obs(mj_data, dof_qpos_addrs, dof_qvel_addrs, vel_command, last_raw_action):
    """Build observation vector for the hotdex locomotion policy.

    Obs layout (138 dims):
        vel_command     (3)  : [x_linvel, y_linvel, yaw_angvel]
        proprio_body   (64)  : body_pos_rel(29) + body_vel(29) + ang_vel(3) + proj_grav(3)
        proprio_hand   (28)  : hand_pos_rel(14) + hand_vel(14)
        actions        (43)  : last raw policy output (action order, before scale)
    """
    # Read state in config order
    joint_pos = mj_data.qpos[dof_qpos_addrs].astype(np.float32)  # (43,)
    joint_vel = mj_data.qvel[dof_qvel_addrs].astype(np.float32)  # (43,)
    base_quat_wxyz = mj_data.qpos[3:7].astype(np.float32)        # (4,)

    # base linear velocity is world-frame, convert to body frame.
    base_lin_vel_world = mj_data.qvel[0:3].astype(np.float32)    # (3,)
    base_lin_vel = quat_rotate_inverse(base_quat_wxyz, base_lin_vel_world).astype(np.float32)

    # base angular velocity from qvel is already local/body frame.
    base_ang_vel = mj_data.qvel[3:6].astype(np.float32)          # (3,)

    # Relative joint positions
    joint_pos_rel = joint_pos - DEFAULT_DOF_ANGLES

    # proprio_body (64): body_pos(29) + body_vel(29) + ang_vel(3) + proj_grav(3)
    body_pos = joint_pos_rel[BODY_INDICES]  # (29,)
    body_vel = joint_vel[BODY_INDICES]       # (29,)
    proj_grav = quat_rotate_inverse(base_quat_wxyz, np.array([0.0, 0.0, -1.0]))  # (3,)
    proprio_body = np.concatenate([body_pos, body_vel, base_ang_vel, proj_grav])   # (64,)

    # proprio_hand (28): hand_pos(14) + hand_vel(14)
    hand_pos = joint_pos_rel[HAND_INDICES]  # (14,)
    hand_vel = joint_vel[HAND_INDICES]       # (14,)
    proprio_hand = np.concatenate([hand_pos, hand_vel])  # (28,)

    # Concatenate: vel_command(3) + proprio_body(64) + proprio_hand(28) + actions(43) = 138
    # obs = np.concatenate([
    #     vel_command.astype(np.float32),
    #     proprio_body,
    #     proprio_hand,
    #     last_raw_action.astype(np.float32),
    # ])
    obs = np.concatenate([
        last_raw_action.astype(np.float32),
        proprio_hand,
        proprio_body,
        vel_command.astype(np.float32),
    ])
    return obs.reshape(1, -1).astype(np.float32)  # (1, 138)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def _set_state_from_traj(data, step, dof_qpos_addrs, dof_qvel_addrs, reorder_map):
    """Set MuJoCo state directly from a trajectory step (pkl joint order → config order)."""
    data.qpos[0:7] = step["root_pose"]
    data.qvel[0:3] = step["root_lin_vel"]
    data.qvel[3:6] = step["root_ang_vel"]
    data.qpos[dof_qpos_addrs] = reorder_to_mj(step["joint_pos"], reorder_map)
    data.qvel[dof_qvel_addrs] = reorder_to_mj(step["joint_vel"], reorder_map)

def run_measure_one_step_error(args):
    """For every step N in the trajectory: reset to data[N], apply action[N] via PD control,
    compare the resulting joint_pos with data[N+1]['joint_pos'], and log the error."""
    root = Path(__file__).parent
    xml_path = args.xml or str(root / "src/holosoma/holosoma/data/robots/g1/g1_43dof.xml")
    pkl_path = args.pkl

    traj, obs_dict, actions, observations = load_trajectory(pkl_path)
    n_steps = len(traj)
    reorder_map = build_reorder_map(traj[0]["joint_order"], DOF_NAMES)
    action_to_config_map = build_action_to_config_map()
    joint_order = list(traj[0]["joint_order"])

    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = 1.0 / args.sim_hz
    mj_data = mujoco.MjData(model)
    sim_dt = float(model.opt.timestep)
    steps_per_policy = args.sim_hz // args.policy_hz

    dof_qpos_addrs, dof_qvel_addrs = mj_hinge_addrs(model, DOF_NAMES)
    actuator_ids = mj_actuator_ids(model, DOF_NAMES)

    video_recorder = None
    if args.render or args.record:
        video_recorder = SimpleVideoRecorder(
            model,
            name="test_loco_mw_one_step_error",
            camera_pos=tuple(args.camera_pos),
            camera_target=tuple(args.camera_target),
            width=args.video_width,
            height=args.video_height,
            save_dir=args.video_dir,
            output_format=args.video_format,
        )
        print(f"Video capture enabled: {args.video_width}x{args.video_height}")

    # ── Sanity check: traj[0]['joint_pos'] vs observations[0] joint positions ──
    jp0_pkl = traj[0]["joint_pos"].astype(np.float32)  # (43,) in pkl joint order
    jp0_config = reorder_to_mj(jp0_pkl, reorder_map)   # (43,) in config order
    obs0_body_pos_rel = observations[0][3:32].astype(np.float32)   # (29,)
    obs0_hand_pos_rel = observations[0][67:81].astype(np.float32)  # (14,)
    traj_body_pos_rel = (jp0_config[BODY_INDICES] - DEFAULT_DOF_ANGLES[BODY_INDICES]).astype(np.float32)  # (29,)
    traj_hand_pos_rel = (jp0_config[HAND_INDICES] - DEFAULT_DOF_ANGLES[HAND_INDICES]).astype(np.float32)  # (14,)

    body_sanity_diff = traj_body_pos_rel - obs0_body_pos_rel
    hand_sanity_diff = traj_hand_pos_rel - obs0_hand_pos_rel
    print(f"\n[Sanity] traj[0]['joint_pos'] vs observations[0] joint pos (traj_rel - obs_rel):")
    body_bad = np.where(np.abs(body_sanity_diff) > 1e-2)[0]
    hand_bad = np.where(np.abs(hand_sanity_diff) > 1e-2)[0]
    if len(body_bad) == 0 and len(hand_bad) == 0:
        print(f"  OK — max body diff={np.abs(body_sanity_diff).max():.2e}, max hand diff={np.abs(hand_sanity_diff).max():.2e}")
    else:
        if len(body_bad) > 0:
            print(f"  body_joint_pos mismatches (> 1e-2):")
            for i in body_bad:
                print(f"    {BODY_JOINT_NAMES[i]:<40s}  traj={traj_body_pos_rel[i]:+.6f}  obs={obs0_body_pos_rel[i]:+.6f}  diff={body_sanity_diff[i]:+.6f}")
        if len(hand_bad) > 0:
            print(f"  hand_joint_pos mismatches (> 1e-2):")
            for i in hand_bad:
                print(f"    {HAND_JOINT_NAMES[i]:<40s}  traj={traj_hand_pos_rel[i]:+.6f}  obs={obs0_hand_pos_rel[i]:+.6f}  diff={hand_sanity_diff[i]:+.6f}")
    print()

    print(f"Measuring one-step error over {n_steps - 1} steps  (dt={sim_dt}, substeps={steps_per_policy})")
    print(f"{'Step':>6s}  {'max|diff|':>10s}  {'mean|diff|':>10s}  {'rms':>10s}  {'obs_max|diff|':>14s}  {'obs_mean|diff|':>14s}")
    print("-" * 75)

    all_diffs = []      # list of (43,) arrays in pkl joint order
    all_obs_diffs = []  # list of (138,) arrays
    all_proc_diffs = []  # list of (43,) arrays in action order (computed vs recorded processed_action)

    viewer_ctx = mujoco.viewer.launch_passive(model, mj_data) if args.render else None
    try:
        for step_idx in range(n_steps - 1):
            step = traj[step_idx]
            if "action" not in step:
                print(f"Step {step_idx}: no action recorded, skipping")
                continue

            # Reset MuJoCo state from recorded data[N]
            _set_state_from_traj(mj_data, step, dof_qpos_addrs, dof_qvel_addrs, reorder_map)
            mujoco.mj_forward(model, mj_data)
            if video_recorder is not None:
                video_recorder.capture_frame(mj_data)
            if viewer_ctx is not None:
                viewer_ctx.sync()

            # Build observation from reconstructed MuJoCo state and compare with recorded obs
            vel_command = step.get("command", np.zeros(3, dtype=np.float32)).astype(np.float32)
            last_raw_action = traj[step_idx - 1]["action"].astype(np.float32) if step_idx > 0 else np.zeros(43, dtype=np.float32)
            built_obs = build_obs(mj_data, dof_qpos_addrs, dof_qvel_addrs, vel_command, last_raw_action).squeeze()  # (138,)
            recorded_obs = observations[step_idx].astype(np.float32)  # (138,)
            obs_diff = built_obs - recorded_obs
            obs_abs_diff = np.abs(obs_diff)
            all_obs_diffs.append(obs_diff)
            # Debug: print joints with obs diff > 0.001
            body_pos_diff = obs_diff[3:32]   # body_pos_rel (29)
            hand_pos_diff = obs_diff[67:81]  # hand_pos_rel (14)
            body_large = np.where(np.abs(body_pos_diff) > 1e-2)[0]
            hand_large = np.where(np.abs(hand_pos_diff) > 1e-2)[0]
            if len(body_large) > 0:
                print(f"  [step {step_idx}] body_joint_pos diff > 1e-2:")
                for i in body_large:
                    print(f"    {BODY_JOINT_NAMES[i]:<40s}  {body_pos_diff[i]:+.6f}")
            if len(hand_large) > 0:
                print(f"  [step {step_idx}] hand_joint_pos diff > 1e-2:")
                for i in hand_large:
                    print(f"    {HAND_JOINT_NAMES[i]:<40s}  {hand_pos_diff[i]:+.6f}")
            # breakpoint()
            # Apply recorded action via PD control for one policy step
            raw_action = step["action"]
            # ── Verify process_action matches pkl processed_actions ───────────
            if "processed_action" in step:
                computed = np.clip(
                    raw_action * ACTION_SCALE + ACTION_OFFSET,
                    ACTION_CLIP_MIN, ACTION_CLIP_MAX,
                ).astype(np.float32)
                recorded_proc = step["processed_action"].astype(np.float32)
                proc_diff = computed - recorded_proc
                proc_abs = np.abs(proc_diff)
                all_proc_diffs.append(proc_diff)
                if proc_abs.max() > 1e-4:
                    print(f"  [step {step_idx}] processed_action mismatch (max={proc_abs.max():.6f}):")
                    for i in np.where(proc_abs > 1e-4)[0]:
                        print(
                            f"    {ACTION_ORDER_43DOF[i]:<40s}"
                            f"  computed={computed[i]:+.6f}"
                            f"  recorded={recorded_proc[i]:+.6f}"
                            f"  diff={proc_diff[i]:+.6f}"
                        )
            target_q = process_action(raw_action, action_to_config_map)
            for _ in range(steps_per_policy):
                apply_pd_control(mj_data, target_q, dof_qpos_addrs, dof_qvel_addrs, actuator_ids)
                mujoco.mj_step(model, mj_data)

            # MuJoCo result → convert config order → pkl joint order
            mj_jp_config = mj_data.qpos[dof_qpos_addrs].astype(np.float32)
            mj_jp_pkl = mj_jp_config[reorder_map]  # (43,) in pkl joint order

            # Recorded next-step joint_pos (already in pkl joint order)
            data_jp_pkl = traj[step_idx + 1]["joint_pos"].astype(np.float32)

            diff = mj_jp_pkl - data_jp_pkl
            abs_diff = np.abs(diff)
            all_diffs.append(diff)

            print(f"{step_idx:6d}  {abs_diff.max():10.6f}  {abs_diff.mean():10.6f}  {np.sqrt((diff**2).mean()):10.6f}  {obs_abs_diff.max():14.6f}  {obs_abs_diff.mean():14.6f}")

    finally:
        if viewer_ctx is not None:
            viewer_ctx.close()
        if video_recorder is not None:
            video_recorder.save(fps=float(args.policy_hz))
            video_recorder.cleanup()

    if not all_diffs:
        print("No steps with actions found.")
        return

    all_diffs = np.array(all_diffs)      # (N-1, 43) in pkl joint order
    all_obs_diffs = np.array(all_obs_diffs)  # (N-1, 138)

    # ── processed_action diff summary ───────────────────────────────────────
    if all_proc_diffs:
        all_proc_diffs = np.array(all_proc_diffs)  # (N-1, 43) in action order
        print(f"\n{'='*70}")
        print(f"process_action verification (computed - recorded) over {len(all_proc_diffs)} steps:")
        print(f"{'Joint (action order)':<42s}  {'max|diff|':>10s}  {'mean|diff|':>10s}")
        print("-" * 70)
        for j, jname in enumerate(ACTION_ORDER_43DOF):
            maxd = np.abs(all_proc_diffs[:, j]).max()
            meand = np.abs(all_proc_diffs[:, j]).mean()
            marker = " <<<" if maxd > 1e-4 else ""
            print(f"  {jname:<40s}  {maxd:10.6f}  {meand:10.6f}{marker}")
        print(f"\nOverall max|diff| : {np.abs(all_proc_diffs).max():.6f}")
        print(f"Overall mean|diff|: {np.abs(all_proc_diffs).mean():.6f}")
    else:
        print("\n[process_action verify] 'processed_actions' not found in pkl — skipped.")

    # ── Observation diff summary ────────────────────────────────────────────
    OBS_SLICES = {
        "command       (3)  ": slice(0, 3),
        "body_pos_rel  (29) ": slice(3, 32),
        "body_vel      (29) ": slice(32, 61),
        "base_ang_vel  (3)  ": slice(61, 64),
        "proj_gravity  (3)  ": slice(64, 67),
        "hand_pos_rel  (14) ": slice(67, 81),
        "hand_vel      (14) ": slice(81, 95),
        "actions       (43) ": slice(95, 138),
    }
    print(f"\n{'='*70}")
    print(f"Observation reconstruction error (built_obs - recorded_obs) over {len(all_obs_diffs)} steps:")
    print(f"{'Segment':<24s}  {'max|diff|':>10s}  {'mean|diff|':>10s}  {'rms':>10s}")
    print("-" * 60)
    for seg_name, sl in OBS_SLICES.items():
        seg = all_obs_diffs[:, sl]
        maxd = np.abs(seg).max()
        meand = np.abs(seg).mean()
        rms = np.sqrt((seg**2).mean())
        marker = " <<<" if maxd > 1e-3 else ""
        print(f"  {seg_name:<22s}  {maxd:10.6f}  {meand:10.6f}  {rms:10.6f}{marker}")
    print(f"\nObs overall max|diff| : {np.abs(all_obs_diffs).max():.6f}")
    print(f"Obs overall mean|diff|: {np.abs(all_obs_diffs).mean():.6f}")
    print(f"Obs overall RMS       : {np.sqrt((all_obs_diffs**2).mean()):.6f}")
    # ── Joint position error summary ────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Per-joint summary over {len(all_diffs)} steps  (mj_result - recorded_next):")
    print(f"{'Joint':<42s}  {'max|diff|':>10s}  {'mean|diff|':>10s}")
    print("-" * 70)
    for j, jname in enumerate(joint_order):
        maxd = np.abs(all_diffs[:, j]).max()
        meand = np.abs(all_diffs[:, j]).mean()
        marker = " <<<" if maxd > 0.01 else ""
        print(f"  {jname:<40s}  {maxd:10.6f}  {meand:10.6f}{marker}")

    print(f"\nOverall max|diff| : {np.abs(all_diffs).max():.6f}")
    print(f"Overall mean|diff|: {np.abs(all_diffs).mean():.6f}")
    print(f"Overall RMS       : {np.sqrt((all_diffs**2).mean()):.6f}")

    # Save error log
    log_dir = root / "test_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"one_step_error_{timestamp}.pkl"
    joblib.dump({"diffs": all_diffs, "joint_order": joint_order}, log_path)
    print(f"\nSaved error log: {log_path}")


def _run_state_replay(args, data, model, traj, n_steps, dof_qpos_addrs, dof_qvel_addrs,
                      reorder_map, video_recorder, policy_dt):
    """Run state replay mode (set qpos/qvel directly from trajectory, no physics)."""
    traj_idx = 0
    done = False
    reset_requested = False

    def key_callback(keycode):
        nonlocal done, reset_requested
        if keycode == ord("Q") or keycode == 256:
            done = True
        elif keycode == ord("R"):
            reset_requested = True

    try:
        if args.offscreen:
            print("Running headless (offscreen) state replay.\n")
            while traj_idx < n_steps:
                step = traj[traj_idx]
                _set_state_from_traj(data, step, dof_qpos_addrs, dof_qvel_addrs, reorder_map)
                mujoco.mj_forward(model, data)
                t = traj_idx / 50.0
                print(f"\r  [State Replay] Step {traj_idx}/{n_steps}  t={t:.3f}s", end="", flush=True)
                traj_idx += 1
                if video_recorder is not None:
                    video_recorder.capture_frame(data)
            print()
        else:
            print("Press Q or ESC to quit, R to reset.\n")
            wall_next_step = time.perf_counter()
            with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
                while viewer.is_running() and not done:
                    if reset_requested:
                        traj_idx = 0
                        reset_requested = False
                        _set_state_from_traj(data, traj[0], dof_qpos_addrs, dof_qvel_addrs, reorder_map)
                        mujoco.mj_forward(model, data)
                        wall_next_step = time.perf_counter()
                        print(f"\n  [Reset] Restarted from step 0", flush=True)
                        continue
                    if traj_idx < n_steps:
                        step = traj[traj_idx]
                        _set_state_from_traj(data, step, dof_qpos_addrs, dof_qvel_addrs, reorder_map)
                        mujoco.mj_forward(model, data)
                        t = traj_idx / 50.0
                        print(f"\r  [State Replay] Step {traj_idx}/{n_steps}  t={t:.3f}s", end="", flush=True)
                        traj_idx += 1
                    elif args.loop:
                        traj_idx = 0
                        continue
                    # else: hold last state
                    if video_recorder is not None:
                        video_recorder.capture_frame(data)
                    viewer.sync()
                    wall_next_step += policy_dt
                    wall_now = time.perf_counter()
                    sleep_s = wall_next_step - wall_now
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    else:
                        wall_next_step = wall_now
    except KeyboardInterrupt:
        print("\n  [Interrupted] Ctrl+C detected.")


def run(args):
    root = Path(__file__).parent
    xml_path = args.xml or str(root / "src/holosoma/holosoma/data/robots/g1/g1_43dof.xml")
    pkl_path = args.pkl

    # Load trajectory
    traj, obs_dict, actions, observations = load_trajectory(pkl_path)
    n_steps = len(traj)
    reorder_map = build_reorder_map(traj[0]["joint_order"], DOF_NAMES)

    # Action processing map (needed for action replay and inference)
    action_to_config_map = build_action_to_config_map()

    replay_actions = None
    if not args.state_replay and not args.infer:
        replay_actions = [s["action"] for s in traj if "action" in s]
        if len(replay_actions) != n_steps:
            raise ValueError(f"Expected {n_steps} actions, got {len(replay_actions)}. "
                             "PKL must contain 'actions' per step.")
        print(f"Loaded {len(replay_actions)} raw actions for replay")

    vel_command = np.zeros(3, dtype=np.float32)
    vel_command[0] = 1.0
    # if "command" in traj[0]:
    #     vel_command = traj[0]["command"].astype(np.float32)
    #     print(f"Velocity command from pkl: {vel_command}")

    print(f"Trajectory: {n_steps} steps at 50Hz = {n_steps / 50.0:.3f}s")

    onnx_session = None
    if args.infer:
        onnx_path = args.onnx or str(
            root / "src/holosoma_inference/holosoma_inference/models/loco/g1_43dof/walk_prior_dr_0306.onnx"
        )
        print(f"Loading ONNX policy: {onnx_path}")
        onnx_session = onnxruntime.InferenceSession(onnx_path)
        inp = onnx_session.get_inputs()[0]
        out = onnx_session.get_outputs()[0]
        print(f"  Input:  {inp.name} {inp.shape}")
        print(f"  Output: {out.name} {out.shape}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = 1.0 / args.sim_hz
    data = mujoco.MjData(model)
    sim_dt = float(model.opt.timestep)
    print(f"Model: {model.nq} qpos, {model.nv} qvel, {model.nu} actuators, dt={sim_dt}")

    steps_per_policy = args.sim_hz // args.policy_hz
    policy_dt = sim_dt * steps_per_policy
    policy_hz = 1.0 / policy_dt
    print(f"Physics: {1.0/sim_dt:.0f}Hz | Policy: {policy_hz:.0f}Hz (every {steps_per_policy} steps)")

    dof_qpos_addrs, dof_qvel_addrs = mj_hinge_addrs(model, DOF_NAMES)
    actuator_ids = mj_actuator_ids(model, DOF_NAMES)

    if args.infer and args.random_init:
        init_idx = int(np.random.randint(0, n_steps))
        print(f"Random init: step {init_idx}/{n_steps}  t={init_idx/50.0:.3f}s")
    else:
        init_idx = 0
    step0 = traj[init_idx]
    init_joint_pos_config = reorder_to_mj(step0["joint_pos"], reorder_map)

    if args.state_replay:
        _set_state_from_traj(data, step0, dof_qpos_addrs, dof_qvel_addrs, reorder_map)
    else:
        data.qpos[0:7] = step0["root_pose"]
        data.qpos[dof_qpos_addrs] = init_joint_pos_config
        data.qvel[0:3] = step0["root_lin_vel"]
        data.qvel[3:6] = step0["root_ang_vel"]
        data.qvel[dof_qvel_addrs] = 0.0
    mujoco.mj_forward(model, data)

    video_recorder = None
    prev_states = [{"joint_vel": data.qvel[dof_qvel_addrs]}]

    if args.record or args.offscreen:
        video_recorder = SimpleVideoRecorder(
            model,
            name="test_loco_mw",
            camera_pos=tuple(args.camera_pos),
            camera_target=tuple(args.camera_target),
            width=args.video_width,
            height=args.video_height,
            save_dir=args.video_dir,
            output_format=args.video_format,
        )
        print(f"Video recording enabled: {args.video_width}x{args.video_height}, {args.video_format} format")
        print(f"  Camera pos: {tuple(args.camera_pos)}, target: {tuple(args.camera_target)}")

    traj_idx = 0
    log_data = [] if args.log else None

    if args.state_replay:
        mode_str = "state replay"
    elif args.infer:
        mode_str = f"infer (random init step={init_idx})" if args.random_init else "infer"
    elif args.loop:
        mode_str = "loop"
    else:
        mode_str = "replay only"
    print(f"\nStarting... mode={mode_str}")

    try:
        if args.state_replay:
            _run_state_replay(
                args, data, model, traj, n_steps, dof_qpos_addrs, dof_qvel_addrs,
                reorder_map, video_recorder, policy_dt,
            )
        else:
            if args.infer:
                target_q = init_joint_pos_config.copy()
            else:
                target_q = process_action(replay_actions[0], action_to_config_map)
            infer_step = 0
            last_raw_action = np.zeros(43, dtype=np.float32)

            def _on_policy_step():
                nonlocal traj_idx, target_q, last_raw_action, infer_step
                if not args.infer:
                    if traj_idx < n_steps:
                        raw_action = replay_actions[traj_idx]
                        last_raw_action = raw_action.copy()
                        target_q = process_action(raw_action, action_to_config_map)
                        t = traj_idx / 50.0
                        print(f"\r  [Replay] Step {traj_idx}/{n_steps}  t={t:.3f}s", end="", flush=True)
                        if log_data is not None:
                            log_data.append({
                                "step": traj_idx,
                                "action": raw_action.copy(),
                                "joint_pos": data.qpos[dof_qpos_addrs].astype(np.float32)[action_to_config_map].copy(),
                            })
                        traj_idx += 1
                    elif args.loop:
                        traj_idx = 0
                        data.qpos[0:7] = traj[0]["root_pose"]
                        data.qvel[0:3] = traj[0]["root_lin_vel"]
                        data.qvel[3:6] = traj[0]["root_ang_vel"]

                if args.infer:
                    obs = build_obs(data, dof_qpos_addrs, dof_qvel_addrs, vel_command, last_raw_action)

                    # if infer_step == 0:
                    #     assert abs(obs - observations[0]).max() < 1e-2

                    if infer_step == 1:
                        print(obs - observations[1])
                        print("Proprio body joint pos")
                        print((obs - observations[1])[0, 3:3+29])
                        print("Proprio body joint vel")
                        print((obs - observations[1])[0, 3+29:3+29+29])
                        print("Proprio body base ang vel")
                        print((obs - observations[1])[0, 3+29+29:3+29+29+3])
                        print("Proprio body base projected_gravity")
                        print((obs - observations[1])[0, 3+29+29+3:3+29+29+3+3])
                        print("Proprio hand joint pos")
                        print((obs - observations[1])[0, 3+29+29+3+3:3+29+29+3+3+14])
                        print("Proprio hand joint vel")
                        print((obs - observations[1])[0, 3+29+29+3+3+14:3+29+29+3+3+14+14])
                        print((prev_states[-1]['joint_pos']-prev_states[-2]['joint_pos'])/sim_dt - prev_states[-1]['joint_vel'])
                        # breakpoint()
                    raw_action = onnx_session.run(None, {"obs": obs})[0].squeeze()
                    last_raw_action = raw_action.copy()
                    target_q = process_action(raw_action, action_to_config_map)
                    if args.ignore_hand_action:
                        target_q[HAND_INDICES] = DEFAULT_DOF_ANGLES[HAND_INDICES]
                    infer_step += 1
                    t = infer_step / 50.0
                    if infer_step % 50 == 0:
                        print(f"\r  [Infer] Step {infer_step}  t={t:.3f}s", end="", flush=True)

            def _apply_ctrl(d):
                apply_pd_control(
                    d, target_q, dof_qpos_addrs, dof_qvel_addrs, actuator_ids,
                    verbose_vel_limit=True,
                )
                prev_states.append({
                    "joint_vel": d.qvel[dof_qvel_addrs].copy(),
                    "joint_pos": d.qpos[dof_qpos_addrs].copy(),
                })

            def _on_reset():
                nonlocal traj_idx, infer_step, last_raw_action, target_q
                nonlocal init_idx, step0, init_joint_pos_config
                traj_idx = 0
                infer_step = 0
                last_raw_action = np.zeros(43, dtype=np.float32)
                if args.infer:
                    if args.random_init:
                        init_idx = int(np.random.randint(0, n_steps))
                        step0 = traj[init_idx]
                        init_joint_pos_config = reorder_to_mj(step0["joint_pos"], reorder_map)
                        print(f"\n  [Reset] Random init from step {init_idx}", flush=True)
                    else:
                        print(f"\n  [Reset] Restarted from step 0", flush=True)
                    target_q = init_joint_pos_config.copy()
                else:
                    target_q = process_action(replay_actions[0], action_to_config_map)
                    print(f"\n  [Reset] Restarted from step 0", flush=True)
                data.qpos[0:7] = step0["root_pose"]
                data.qpos[dof_qpos_addrs] = init_joint_pos_config
                data.qvel[0:3] = step0["root_lin_vel"]
                data.qvel[3:6] = step0["root_ang_vel"]
                data.qvel[dof_qvel_addrs] = 0.0
                mujoco.mj_forward(model, data)

            def _should_stop():
                return not args.infer and not args.loop and traj_idx >= n_steps

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
        if log_data:
            log_dir = root / "test_log"
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = log_dir / f"test_loco_{timestamp}.pkl"
            joblib.dump(log_data, log_path)
            print(f"\n  [Log] Saved {len(log_data)} steps to {log_path}")
        if video_recorder is not None:
            video_fps = 1.0 / (sim_dt * steps_per_policy)
            video_recorder.save(fps=video_fps)
            video_recorder.cleanup()

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Locomotion replay + inference in MuJoCo")
    parser.add_argument(
        "--pkl",
        default="src/holosoma/holosoma/data/motions/g1_43dof/prior/walk_o_a_full_state.pkl",
        help="Path to pkl file with robot_state trajectory",
    )
    parser.add_argument("--xml", default=None, help="MuJoCo XML path (default: g1_43dof.xml)")
    parser.add_argument("--onnx", default=None, help="ONNX policy path (default: walk_prior.onnx)")
    parser.add_argument("--measure_one_step_error", action="store_true",
                        help="For each step N: reset to data[N], apply action[N], compare result with data[N+1].")
    parser.add_argument("--render", action="store_true",
                        help="With --measure_one_step_error: open a viewer to visualize each step.")
    parser.add_argument("--state-replay", action="store_true",
                        help="Direct state replay: set qpos/qvel from pkl each step (no PD control).")
    parser.add_argument("--loop", action="store_true", help="Loop replay (no inference)")
    parser.add_argument("--infer", action="store_true", help="Run ONNX policy inference (no replay)")
    parser.add_argument("--random-init", action="store_true",
                        help="With --infer: initialize from a random trajectory step instead of step 0.")
    parser.add_argument("--log", action="store_true", help="Log actions and joint_pos per step, save as pkl")
    parser.add_argument("--offscreen", action="store_true",
                        help="Run headless (no viewer). For replay/infer: stops when trajectory ends.")
    parser.add_argument("--record", action="store_true", help="Enable offscreen video recording (saved on exit).")
    parser.add_argument("--video-dir", type=str, default="logs/videos", help="Directory for saved videos.")
    parser.add_argument("--video-width", type=int, default=1920, help="Video frame width in pixels.")
    parser.add_argument("--video-height", type=int, default=1080, help="Video frame height in pixels.")
    parser.add_argument("--video-format", type=str, default="h264", choices=["h264", "mp4"], help="Video codec.")
    parser.add_argument("--camera-pos", type=float, nargs=3, default=[-2.0, 0.0, 1.5],
                        help="Camera position [x y z] (default: -2.0 0.0 1.5).")
    parser.add_argument("--camera-target", type=float, nargs=3, default=[1.0, 0.0, 1.0],
                        help="Camera target [x y z] (default: 1.0 0.0 1.0).")
    parser.add_argument("--sim-hz", type=int, default=200, help="Simulation frequency in Hz (default: 200)")
    parser.add_argument("--policy-hz", type=int, default=50, help="Policy frequency in Hz (default: 50)")
    parser.add_argument("--ignore-hand-action", action="store_true",
                        help="Override hand joint targets with default positions instead of policy output.")
    args = parser.parse_args()

    if args.measure_one_step_error:
        run_measure_one_step_error(args)
    else:
        run(args)


if __name__ == "__main__":
    main()

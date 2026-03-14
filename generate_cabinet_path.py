import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import joblib
import numpy as np
import os

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]

H = 0.54 # 0.86 / 1.15

# ---------------------------
# Quaternion utils
# ---------------------------

def quat_normalize(q):
    q = np.asarray(q, dtype=np.float32)
    n = np.linalg.norm(q) + 1e-8
    return q / n


def quat_mul_wxyz(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)


def quat_from_axis_angle(axis_xyz, angle_rad):
    axis = np.asarray(axis_xyz, dtype=np.float32)
    n = np.linalg.norm(axis) + 1e-8
    axis = axis / n
    s = math.sin(angle_rad / 2.0)
    return quat_normalize(np.array(
        [math.cos(angle_rad / 2.0), axis[0]*s, axis[1]*s, axis[2]*s],
        dtype=np.float32
    ))


def quat_slerp_wxyz(q0, q1, t):
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        out = q0 + (q1 - q0) * float(t)
        return quat_normalize(out)

    theta0 = math.acos(dot)
    sin0 = math.sin(theta0)
    a = math.sin((1.0 - float(t)) * theta0) / sin0
    b = math.sin(float(t) * theta0) / sin0
    return quat_normalize(a * q0 + b * q1)


def make_obj_rot_intervals(
    T: int,
    dt: float,
    q0_xyzw,
    rotation_interval_sec: float,
    dtheta_max_rad: float,
    rng_seed: int = 0,
    axis_xyz=(0.0, 0.0, 1.0),
):
    assert T > 0
    assert dt > 0
    assert rotation_interval_sec > 0
    assert dtheta_max_rad >= 0

    rng = np.random.default_rng(rng_seed)

    q0_xyzw = np.asarray(q0_xyzw, dtype=np.float32)
    assert q0_xyzw.shape == (4,)

    q_cur = quat_normalize(np.array([q0_xyzw[3], q0_xyzw[0], q0_xyzw[1], q0_xyzw[2]], dtype=np.float32))

    interval_frames = max(1, int(round(rotation_interval_sec / dt)))
    grs_wxyz = np.zeros((T, 4), dtype=np.float32)

    t0 = 0
    while t0 < T:
        t1 = min(T, t0 + interval_frames)

        if dtheta_max_rad <= 1e-12:
            q_next = q_cur
        else:
            delta = float(rng.uniform(-dtheta_max_rad, dtheta_max_rad))
            q_delta = quat_from_axis_angle(axis_xyz, delta)
            q_next = quat_mul_wxyz(q_delta, q_cur)

        span = (t1 - t0)
        for k in range(span):
            t = 1.0 if span == 1 else (k / (span - 1))
            grs_wxyz[t0 + k] = quat_slerp_wxyz(q_cur, q_next, t)

        q_cur = q_next
        t0 = t1

    grs_xyzw = grs_wxyz[:, [1, 2, 3, 0]].copy()
    return grs_xyzw


# ---------------------------
# Basic geometry / resampling
# ---------------------------

def lerp2(a: Vec2, b: Vec2, t: float) -> Vec2:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def resample_linear_2d(a: Vec2, b: Vec2, speed_mps: float, dt: float) -> List[Vec2]:
    assert speed_mps > 0 and dt > 0
    dist = math.hypot(b[0] - a[0], b[1] - a[1])
    if dist <= 1e-9:
        return [a]
    step = speed_mps * dt
    n = max(1, int(math.ceil(dist / step)))
    out = []
    for k in range(n + 1):
        t = k / n
        out.append(lerp2(a, b, t))
    return out


def _turn_angle_rad(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    u = p1 - p0
    v = p2 - p1
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    u /= nu
    v /= nv
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(math.acos(c))


def _polyline_arclength(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    d = P[1:] - P[:-1]
    seg_len = np.linalg.norm(d, axis=1).astype(np.float32)
    s_nodes = np.zeros((P.shape[0],), dtype=np.float32)
    s_nodes[1:] = np.cumsum(seg_len)
    total = float(s_nodes[-1])
    return seg_len, s_nodes, total


def _eval_polyline_at_s(P: np.ndarray, seg_len: np.ndarray, s_nodes: np.ndarray, s: float) -> Vec2:
    if s <= 0.0:
        return (float(P[0, 0]), float(P[0, 1]))
    total = float(s_nodes[-1])
    if s >= total:
        return (float(P[-1, 0]), float(P[-1, 1]))

    i = int(np.searchsorted(s_nodes, s, side="right") - 1)
    i = max(0, min(i, P.shape[0] - 2))

    L = float(seg_len[i])
    if L < 1e-9:
        return (float(P[i, 0]), float(P[i, 1]))

    u = (s - float(s_nodes[i])) / L
    q = (1.0 - u) * P[i] + u * P[i + 1]
    return (float(q[0]), float(q[1]))


def _trap_profile(L: float, v_max: float, a_max: float) -> Tuple[float, float, float, float]:
    t_to_vmax = v_max / a_max
    d_acc = 0.5 * a_max * t_to_vmax * t_to_vmax

    if 2.0 * d_acc >= L:
        v_peak = math.sqrt(max(0.0, L * a_max))
        t_acc = v_peak / a_max
        t_cruise = 0.0
        t_total = 2.0 * t_acc
        return t_acc, t_cruise, t_total, v_peak

    v_peak = v_max
    d_cruise = L - 2.0 * d_acc
    t_cruise = d_cruise / v_peak
    t_acc = t_to_vmax
    t_total = 2.0 * t_acc + t_cruise
    return t_acc, t_cruise, t_total, v_peak


def _trap_s_at_t(t: float, L: float, a_max: float, t_acc: float, t_cruise: float, v_peak: float) -> float:
    if t <= 0.0:
        return 0.0

    t1 = t_acc
    t2 = t_acc + t_cruise
    t3 = 2.0 * t_acc + t_cruise

    if t >= t3:
        return L

    if t <= t1:
        return 0.5 * a_max * t * t

    if t <= t2:
        d_acc = 0.5 * a_max * t1 * t1
        return d_acc + v_peak * (t - t1)

    tau = t3 - t
    return L - 0.5 * a_max * tau * tau


def resample_polyline_trapezoid_with_stops(
    pts: List[Vec2],
    dt: float,
    v_max: float,
    a_max: float,
    stop_angle_deg: float = 90.0,
) -> List[Vec2]:
    if len(pts) < 2:
        return pts[:]
    if dt <= 0.0:
        raise ValueError("dt must be > 0")
    if v_max <= 0.0:
        raise ValueError("v_max must be > 0")
    if a_max <= 0.0:
        raise ValueError("a_max must be > 0")

    P = np.asarray(pts, dtype=np.float32)
    N = P.shape[0]

    seg_len, s_nodes, total = _polyline_arclength(P)
    if total <= 1e-9:
        return [(float(P[0, 0]), float(P[0, 1]))]

    stop_angle = math.radians(float(stop_angle_deg))
    stop_idx = np.zeros((N,), dtype=bool)
    stop_idx[0] = True
    stop_idx[-1] = True

    if N >= 3:
        for i in range(1, N - 1):
            theta = _turn_angle_rad(P[i - 1], P[i], P[i + 1])
            if theta >= stop_angle:
                stop_idx[i] = True

    stop_indices = sorted(set([i for i in range(N) if stop_idx[i]]))

    out: List[Vec2] = []

    def push(p: Vec2) -> None:
        if not out:
            out.append(p)
            return
        if math.hypot(out[-1][0] - p[0], out[-1][1] - p[1]) > 1e-9:
            out.append(p)

    push((float(P[0, 0]), float(P[0, 1])))

    for a_i, b_i in zip(stop_indices[:-1], stop_indices[1:]):
        s_a = float(s_nodes[a_i])
        s_b = float(s_nodes[b_i])
        L = s_b - s_a
        if L <= 1e-9:
            push((float(P[b_i, 0]), float(P[b_i, 1])))
            continue

        t_acc, t_cruise, t_total, v_peak = _trap_profile(L=L, v_max=float(v_max), a_max=float(a_max))

        num_steps = int(math.floor(t_total / dt))
        for k in range(1, num_steps + 1):
            t = k * dt
            s_local = _trap_s_at_t(
                t=t,
                L=L,
                a_max=float(a_max),
                t_acc=float(t_acc),
                t_cruise=float(t_cruise),
                v_peak=float(v_peak),
            )
            s = s_a + s_local
            push(_eval_polyline_at_s(P, seg_len, s_nodes, s))

        push((float(P[b_i, 0]), float(P[b_i, 1])))

    return out


# ---------------------------
# Path build: hold + go to src + go to tgt (straight), with smoothing profile
# ---------------------------

def _fmt_xz_tag(x: float, z: float) -> str:
    xi = int(round(x * 100.0))
    zi = int(round(z * 100.0))
    return f"{xi:03d}_{zi:03d}"


def make_obj_pos_hold_src_tgt(
    obj_pos0: Vec3,
    dt: float,
    hold_sec: float,
    move_speed: float,
    src_pos: Vec3,
    tgt_pos: Vec3,
    cruise_speed: float,
    accel_max: float,
    stop_angle_deg: float = 90.0,
    tgt_hold_sec: float = 0.0,  # <-- add
) -> List[Vec3]:
    assert dt > 0
    assert hold_sec >= 0
    assert move_speed > 0
    assert cruise_speed > 0
    assert accel_max > 0
    assert tgt_hold_sec >= 0  # <-- add

    x0, y0, z0 = obj_pos0
    sx, sy, sz = src_pos
    tx, ty, tz = tgt_pos

    y0 = float(y0)
    sy = float(sy)
    ty = float(ty)

    hold_frames = int(round(hold_sec / dt))
    out: List[Vec3] = [(float(x0), y0, float(z0)) for _ in range(max(0, hold_frames))]

    # 1) initial -> src : xz linear, y: y0 -> sy
    p0_xz = (float(x0), float(z0))
    src_xz = (float(sx), float(sz))
    seg1 = resample_linear_2d(p0_xz, src_xz, speed_mps=float(move_speed), dt=float(dt))

    if out and math.hypot(out[-1][0] - seg1[0][0], out[-1][2] - seg1[0][1]) < 1e-9:
        seg1 = seg1[1:]

    n1 = len(seg1)
    if n1 == 1:
        out.append((seg1[0][0], sy, seg1[0][1]))
    else:
        for i, (x, z) in enumerate(seg1):
            a = i / (n1 - 1)
            y = (1.0 - a) * y0 + a * sy
            out.append((float(x), float(y), float(z)))

    # snap to src
    out[-1] = (float(sx), sy, float(sz))

    # 2) src -> tgt : xz trapezoid smoothing
    pts_2d: List[Vec2] = [src_xz, (float(tx), float(tz))]
    seg2 = resample_polyline_trapezoid_with_stops(
        pts_2d,
        dt=float(dt),
        v_max=float(cruise_speed),
        a_max=float(accel_max),
        stop_angle_deg=float(stop_angle_deg),
    )

    if seg2 and math.hypot(out[-1][0] - seg2[0][0], out[-1][2] - seg2[0][1]) < 1e-9:
        seg2 = seg2[1:]

    y_start = float(out[-1][1])

    if seg2:
        n2 = len(seg2)
        if n2 == 1:
            out.append((seg2[0][0], ty, seg2[0][1]))
        else:
            for i, (x, z) in enumerate(seg2):
                a = i / (n2 - 1)
                y = (1.0 - a) * y_start + a * ty
                out.append((float(x), float(y), float(z)))

    # snap to tgt
    out[-1] = (float(tx), ty, float(tz))

    # 3) hold at tgt
    tgt_hold_frames = int(round(tgt_hold_sec / dt))
    if tgt_hold_frames > 0:
        last = out[-1]
        out.extend([last] * tgt_hold_frames)

    assert len(out) >= 2
    return out

# ---------------------------
# Motion IO
# ---------------------------

def pad_motion_to_T(motion: dict, T_new: int, exclude_keys=("fps", "obj_name")) -> None:
    assert isinstance(motion, dict)
    assert isinstance(T_new, int) and T_new > 0

    for k, v in list(motion.items()):
        if k in exclude_keys or v is None:
            continue
        if not hasattr(v, "__len__"):
            continue

        T_old = len(v)
        if T_old == 0 or T_old == T_new:
            continue

        if isinstance(v, np.ndarray):
            if T_old > T_new:
                motion[k] = v[:T_new].copy()
            else:
                pad_n = T_new - T_old
                last_frame = v[-1:]
                pad_block = np.repeat(last_frame, pad_n, axis=0)
                motion[k] = np.concatenate([v, pad_block], axis=0)
        else:
            last = v[-1]
            if T_old > T_new:
                motion[k] = list(v[:T_new])
            else:
                pad_n = T_new - T_old
                motion[k] = list(v) + [last] * pad_n


def save_motion_with_path(
    motion_file: str,
    motion_key: str,
    src_pos: Vec3,
    tgt_pos: Vec3,
    hold_sec: float = 2.5,
    move_speed: float = 1.0,
    cruise_speed: float = 1.0,
    accel_max: float = 2.0,
    rotation_interval_sec: float = 1.5,
    dtheta_max_rad: float = 0.0,
    rot_seed: int = 0,
    out_file: Optional[str] = None,
    tgt_hold_sec: float = 1.0,  # <-- add (default 1s)
) -> Tuple[str, int, int, float]:
    
    data = joblib.load(motion_file)
    assert isinstance(data, dict)
    assert motion_key in data

    motion = data[motion_key]
    assert isinstance(motion, dict)
    assert "obj_pos" in motion
    assert "obj_rot" in motion

    obj_pos = motion["obj_pos"]
    assert hasattr(obj_pos, "__len__") and len(obj_pos) > 0
    obj_pos0 = tuple(obj_pos[0])
    assert len(obj_pos0) == 3

    if "dt" in motion:
        dt = float(motion["dt"])
    elif "fps" in motion:
        fps = float(motion["fps"])
        assert fps > 0
        dt = 1.0 / fps
    else:
        raise AssertionError("No 'dt' or 'fps' found in motion.")
    assert dt > 0

    new_obj_pos = make_obj_pos_hold_src_tgt(
        obj_pos0=(float(obj_pos0[0]), float(obj_pos0[1]), float(obj_pos0[2])),
        dt=float(dt),
        hold_sec=float(hold_sec),
        move_speed=float(move_speed),
        src_pos=src_pos,
        tgt_pos=tgt_pos,
        cruise_speed=float(cruise_speed),
        accel_max=float(accel_max),
        stop_angle_deg=90.0,
        tgt_hold_sec=float(tgt_hold_sec),  # <-- add
    )

    obj_rot0 = np.asarray(motion["obj_rot"][0], dtype=np.float32)
    assert obj_rot0.shape == (4,)

    new_obj_rot = make_obj_rot_intervals(
        T=len(new_obj_pos),
        dt=float(dt),
        q0_xyzw=obj_rot0,
        rotation_interval_sec=float(rotation_interval_sec),
        dtheta_max_rad=float(dtheta_max_rad),
        rng_seed=int(rot_seed),
        axis_xyz=(0.0, 1.0, 0.0),
    )

    motion["obj_pos"] = new_obj_pos
    motion["obj_rot"] = new_obj_rot.tolist()

    # lift 동작 유지 (기존 로직 그대로)
    assert "table_pos" in motion
    table_pos = np.asarray(motion["table_pos"], dtype=np.float32)
    assert table_pos.ndim == 2 and table_pos.shape[1] == 3

    T_new = len(new_obj_pos)
    if len(table_pos) > T_new:
        table_pos = table_pos[:T_new].copy()

    t = np.arange(len(table_pos), dtype=np.float32) * float(dt)
    table_pos[t >= 3.0, 2] = -10.0
    motion["table_pos"] = table_pos.tolist()

    pad_motion_to_T(motion, T_new=T_new, exclude_keys=("fps", "obj_name"))

    if out_file is None:
        src_tag = _fmt_xz_tag(float(src_pos[0]), float(src_pos[2]))
        tgt_tag = _fmt_xz_tag(float(tgt_pos[0]), float(tgt_pos[2]))
        out_file = f"cabinet_src_{src_tag}_tgt_{tgt_tag}.pkl"

    joblib.dump({motion_key: motion}, out_file)
    print("SAVED TO", out_file)
    return out_file, len(obj_pos), len(new_obj_pos), float(dt)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=float, required=True)
    parser.add_argument("--tgt_x", type=float, required=True)
    parser.add_argument("--src_x", type=float, default=0.65)       # fixed by default
    parser.add_argument("--src_z_offset", type=float, default=0.07) # keep your prior +0.07
    parser.add_argument("--tgt_hold_sec", type=float, default=1.0)
    args = parser.parse_args()

    motion_file = "/data3/minu/GRAB/cubemediums_12_0116.pkl"
    motion_key = "GRAB_s1_cubemedium_pass_1"

    H = float(args.H)
    src_x = 0.65  # enforce fixed
    tgt_x = float(args.tgt_x)

    src_pos = (src_x, 0.0, H + float(args.src_z_offset))
    tgt_pos = (tgt_x, 0.0, H)

    out_file, old_len, new_len, dt = save_motion_with_path(
        motion_file=motion_file,
        motion_key=motion_key,
        src_pos=src_pos,
        tgt_pos=tgt_pos,
        hold_sec=2.5,
        move_speed=1.0,
        cruise_speed=1.0,
        accel_max=2.0,
        rotation_interval_sec=1.5,
        dtheta_max_rad=0.0,
        rot_seed=123,
        out_file=None,
        tgt_hold_sec=float(args.tgt_hold_sec),  # <-- tgt stop
    )

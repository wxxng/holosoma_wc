import numpy as np

MODES = ["lift", "lift_down", "right", "left", "left_down", "left_back_down", "back", "lift_back_left_down", "stay"]
LEFT_TARGET_Y_M = 0.7
RIGHT_TARGET_Y_M = -LEFT_TARGET_Y_M
LEFT_DOWN_LOWER_HEIGHT_M = 0.8
BACK_DISTANCE_M = 0.3
LEFT_BACK_DOWN_BACK_M = 0.1
LIFT_BACK_LEFT_DOWN_BACK_M = 0.1
LIFT_BACK_LEFT_DOWN_LEFT_M = 0.4
LIFT_BACK_LEFT_DOWN_LOWER_M = 0.8


def _linear_ramp(n: int, start: float, end: float) -> np.ndarray:
    """Simple linear interpolation from start to end over n steps."""
    if n <= 0:
        return np.array([], dtype=np.float32)
    if n == 1:
        return np.array([end], dtype=np.float32)
    return np.linspace(start, end, n + 1)[1:].astype(np.float32)


def _trapezoid_ramp(n: int, start: float, end: float, accel_frac: float = 0.25) -> np.ndarray:
    """Trapezoidal velocity profile interpolation from start to end over n steps.

    accel_frac: fraction of total steps spent accelerating (same for decelerating).
    Velocity is zero at both endpoints, ramps up linearly, cruises, ramps down.
    """
    if n <= 0:
        return np.array([], dtype=np.float32)
    if n == 1:
        return np.array([end], dtype=np.float32)

    dist = end - start
    # Number of steps for accel / decel / cruise
    n_acc = max(1, int(round(n * accel_frac)))
    n_dec = max(1, int(round(n * accel_frac)))
    n_cruise = max(0, n - n_acc - n_dec)

    # Build velocity profile (unnormalized): ramp up, flat, ramp down
    v_acc = np.linspace(0.0, 1.0, n_acc + 1)[1:]       # 1..n_acc
    v_cruise = np.ones(n_cruise)
    v_dec = np.linspace(1.0, 0.0, n_dec + 1)[1:]       # 1..n_dec
    v = np.concatenate([v_acc, v_cruise, v_dec])

    # Integrate velocity to get position, then scale to match desired distance
    pos = np.cumsum(v)
    pos = pos / pos[-1] * dist + start  # normalize so final value = end
    return pos.astype(np.float32)


class TrajectoryGenerator:
    def __init__(
        self,
        rl_rate: float,
        hold_time_s: float = 4.0,
        lift_height_m: float = 0.4,
        down_height_m: float = 0.4,
        lift_speed_mps: float = 0.2,
        mode: str = "lift",
        trapezoid: bool = True,
    ):
        self.rl_rate = float(rl_rate)
        self.hold_time_s = float(hold_time_s)
        self.lift_height_m = float(lift_height_m)
        self.down_height_m = float(down_height_m)
        self.lift_speed_mps = float(lift_speed_mps)
        self.mode = mode
        self._ramp = _trapezoid_ramp if trapezoid else _linear_ramp

    def build_gen_traj(
        self,
        start_pos: np.ndarray,
        start_quat_wxyz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        start_pos = np.asarray(start_pos, dtype=np.float32).reshape(3,)
        start_quat_wxyz = np.asarray(start_quat_wxyz, dtype=np.float32).reshape(4,)

        # segment lengths (in frames)
        n_hold = int(round(self.hold_time_s * self.rl_rate))
        n_hold = max(n_hold, 1)

        if self.mode == "stay":
            pos = np.repeat(start_pos.reshape(1, 3), n_hold, axis=0)
            quat = np.repeat(start_quat_wxyz.reshape(1, 4), n_hold, axis=0)
            return pos, quat

        if self.mode == "lift_down":
            lift_time_s = self.lift_height_m / self.lift_speed_mps
            n_lift = int(round(lift_time_s * self.rl_rate))
            n_lift = max(n_lift, 1)

            down_time_s = self.down_height_m / self.lift_speed_mps
            n_down = int(round(down_time_s * self.rl_rate))
            n_down = max(n_down, 1)

            final_z_offset = self.lift_height_m - self.down_height_m

            # z: hold -> lift up -> hold at top -> lower by down_height_m -> hold
            z_hold = np.zeros((n_hold,), dtype=np.float32)
            z_up = self._ramp(n_lift, 0.0, self.lift_height_m)
            z_top_hold = np.full((n_hold,), self.lift_height_m, dtype=np.float32)
            z_down = self._ramp(n_down, self.lift_height_m, final_z_offset)
            z_final_hold = np.full((n_hold,), final_z_offset, dtype=np.float32)
            z = np.concatenate([z_hold, z_up, z_top_hold, z_down, z_final_hold], axis=0)

            pos = np.repeat(start_pos.reshape(1, 3), z.shape[0], axis=0)
            pos[:, 2] = start_pos[2] + z

            quat = np.repeat(start_quat_wxyz.reshape(1, 4), z.shape[0], axis=0)
            return pos, quat

        if self.mode in {"left", "right"}:
            target_y_m = LEFT_TARGET_Y_M if self.mode == "left" else RIGHT_TARGET_Y_M
            lift_time_s = self.lift_height_m / self.lift_speed_mps
            n_lift = int(round(lift_time_s * self.rl_rate))
            n_lift = max(n_lift, 1)

            move_distance_m = abs(target_y_m - float(start_pos[1]))
            move_time_s = move_distance_m / self.lift_speed_mps if self.lift_speed_mps > 0.0 else 0.0
            n_move = int(round(move_time_s * self.rl_rate))
            n_move = max(n_move, 1)

            z_hold = np.zeros((n_hold,), dtype=np.float32)
            z_up = self._ramp(n_lift, 0.0, self.lift_height_m)
            z_move = np.full((n_move,), self.lift_height_m, dtype=np.float32)
            z_final_hold = np.full((n_move,), self.lift_height_m, dtype=np.float32)
            z = np.concatenate([z_hold, z_up, z_move, z_final_hold], axis=0)

            y_hold = np.full((n_hold,), start_pos[1], dtype=np.float32)
            y_lift = np.full((n_lift,), start_pos[1], dtype=np.float32)
            y_move = self._ramp(n_move, start_pos[1], target_y_m)
            y_final_hold = np.full((n_move,), target_y_m, dtype=np.float32)
            y = np.concatenate([y_hold, y_lift, y_move, y_final_hold], axis=0)

            pos = np.repeat(start_pos.reshape(1, 3), z.shape[0], axis=0)
            pos[:, 1] = y
            pos[:, 2] = start_pos[2] + z

            quat = np.repeat(start_quat_wxyz.reshape(1, 4), z.shape[0], axis=0)
            return pos, quat

        if self.mode == "left_down":
            target_y_m = LEFT_TARGET_Y_M
            lower_height_m = LEFT_DOWN_LOWER_HEIGHT_M

            lift_time_s = self.lift_height_m / self.lift_speed_mps
            n_lift = int(round(lift_time_s * self.rl_rate))
            n_lift = max(n_lift, 1)

            move_distance_m = abs(target_y_m - float(start_pos[1]))
            move_time_s = move_distance_m / self.lift_speed_mps if self.lift_speed_mps > 0.0 else 0.0
            n_move = int(round(move_time_s * self.rl_rate))
            n_move = max(n_move, 1)

            lower_time_s = lower_height_m / self.lift_speed_mps
            n_lower = int(round(lower_time_s * self.rl_rate))
            n_lower = max(n_lower, 1)

            # z profile: hold -> lift up -> move laterally (stay high) -> lower down -> hold at final z
            z_hold = np.zeros((n_hold,), dtype=np.float32)
            z_up = self._ramp(n_lift, 0.0, self.lift_height_m)
            z_move = np.full((n_move,), self.lift_height_m, dtype=np.float32)
            final_z_offset = self.lift_height_m - lower_height_m
            z_down = self._ramp(n_lower, self.lift_height_m, final_z_offset)
            z_final_hold = np.full((n_hold,), final_z_offset, dtype=np.float32)
            z = np.concatenate([z_hold, z_up, z_move, z_down, z_final_hold], axis=0)

            # y profile: hold -> stay during lift -> move laterally -> stay during lower -> hold
            y_hold = np.full((n_hold,), start_pos[1], dtype=np.float32)
            y_lift = np.full((n_lift,), start_pos[1], dtype=np.float32)
            y_move = self._ramp(n_move, start_pos[1], target_y_m)
            y_lower = np.full((n_lower,), target_y_m, dtype=np.float32)
            y_final_hold = np.full((n_hold,), target_y_m, dtype=np.float32)
            y = np.concatenate([y_hold, y_lift, y_move, y_lower, y_final_hold], axis=0)

            pos = np.repeat(start_pos.reshape(1, 3), z.shape[0], axis=0)
            pos[:, 1] = y
            pos[:, 2] = start_pos[2] + z

            quat = np.repeat(start_quat_wxyz.reshape(1, 4), z.shape[0], axis=0)
            return pos, quat

        if self.mode == "left_back_down":
            target_y_m = LIFT_BACK_LEFT_DOWN_LEFT_M
            back_distance_m = LEFT_BACK_DOWN_BACK_M
            lower_height_m = LEFT_DOWN_LOWER_HEIGHT_M

            lift_time_s = self.lift_height_m / self.lift_speed_mps
            n_lift = int(round(lift_time_s * self.rl_rate))
            n_lift = max(n_lift, 1)

            # Diagonal move: duration based on the longer axis so both finish together
            y_distance_m = abs(target_y_m - float(start_pos[1]))
            move_distance_m = max(y_distance_m, back_distance_m)
            move_time_s = move_distance_m / self.lift_speed_mps if self.lift_speed_mps > 0.0 else 0.0
            n_move = int(round(move_time_s * self.rl_rate))
            n_move = max(n_move, 1)

            lower_time_s = lower_height_m / self.lift_speed_mps
            n_lower = int(round(lower_time_s * self.rl_rate))
            n_lower = max(n_lower, 1)

            # z profile: hold -> lift up -> move diagonally (stay high) -> lower down -> hold
            z_hold = np.zeros((n_hold,), dtype=np.float32)
            z_up = self._ramp(n_lift, 0.0, self.lift_height_m)
            z_move = np.full((n_move,), self.lift_height_m, dtype=np.float32)
            final_z_offset = self.lift_height_m - lower_height_m
            z_down = self._ramp(n_lower, self.lift_height_m, final_z_offset)
            z_final_hold = np.full((n_hold,), final_z_offset, dtype=np.float32)
            z = np.concatenate([z_hold, z_up, z_move, z_down, z_final_hold], axis=0)

            # y profile: hold -> stay during lift -> move left -> stay during lower -> hold
            y_hold = np.full((n_hold,), start_pos[1], dtype=np.float32)
            y_lift = np.full((n_lift,), start_pos[1], dtype=np.float32)
            y_move = self._ramp(n_move, start_pos[1], target_y_m)
            y_lower = np.full((n_lower,), target_y_m, dtype=np.float32)
            y_final_hold = np.full((n_hold,), target_y_m, dtype=np.float32)
            y = np.concatenate([y_hold, y_lift, y_move, y_lower, y_final_hold], axis=0)

            # x profile: hold -> stay during lift -> move back (-x) -> stay during lower -> hold
            target_x_m = float(start_pos[0]) - back_distance_m
            x_hold = np.full((n_hold,), start_pos[0], dtype=np.float32)
            x_lift = np.full((n_lift,), start_pos[0], dtype=np.float32)
            x_move = self._ramp(n_move, start_pos[0], target_x_m)
            x_lower = np.full((n_lower,), target_x_m, dtype=np.float32)
            x_final_hold = np.full((n_hold,), target_x_m, dtype=np.float32)
            x = np.concatenate([x_hold, x_lift, x_move, x_lower, x_final_hold], axis=0)

            pos = np.repeat(start_pos.reshape(1, 3), z.shape[0], axis=0)
            pos[:, 0] = x
            pos[:, 1] = y
            pos[:, 2] = start_pos[2] + z

            quat = np.repeat(start_quat_wxyz.reshape(1, 4), z.shape[0], axis=0)
            return pos, quat

        if self.mode == "back":
            back_distance_m = BACK_DISTANCE_M

            lift_time_s = self.lift_height_m / self.lift_speed_mps
            n_lift = int(round(lift_time_s * self.rl_rate))
            n_lift = max(n_lift, 1)

            move_time_s = back_distance_m / self.lift_speed_mps if self.lift_speed_mps > 0.0 else 0.0
            n_move = int(round(move_time_s * self.rl_rate))
            n_move = max(n_move, 1)

            # Same as lift: hold -> ramp up -> move back (-x) -> final hold
            z_hold = np.zeros((n_hold,), dtype=np.float32)
            z_up = self._ramp(n_lift, 0.0, self.lift_height_m)
            z_move = np.full((n_move,), self.lift_height_m, dtype=np.float32)
            z_final_hold = np.full((n_lift,), self.lift_height_m, dtype=np.float32)
            z = np.concatenate([z_hold, z_up, z_move, z_final_hold], axis=0)

            # x profile: fixed during hold+lift, then move back (-x), then hold.
            target_x_m = float(start_pos[0]) - back_distance_m
            x_hold = np.full((n_hold,), start_pos[0], dtype=np.float32)
            x_lift = np.full((n_lift,), start_pos[0], dtype=np.float32)
            x_move = self._ramp(n_move, start_pos[0], target_x_m)
            x_final_hold = np.full((n_lift,), target_x_m, dtype=np.float32)
            x = np.concatenate([x_hold, x_lift, x_move, x_final_hold], axis=0)

            pos = np.repeat(start_pos.reshape(1, 3), z.shape[0], axis=0)
            pos[:, 0] = x
            pos[:, 2] = start_pos[2] + z

            quat = np.repeat(start_quat_wxyz.reshape(1, 4), z.shape[0], axis=0)
            return pos, quat

        if self.mode == "lift_back_left_down":
            back_distance_m = LIFT_BACK_LEFT_DOWN_BACK_M
            left_target_y_m = LIFT_BACK_LEFT_DOWN_LEFT_M
            lower_height_m = LIFT_BACK_LEFT_DOWN_LOWER_M

            lift_time_s = self.lift_height_m / self.lift_speed_mps
            n_lift = int(round(lift_time_s * self.rl_rate))
            n_lift = max(n_lift, 1)

            n_back = max(1, int(round((back_distance_m / self.lift_speed_mps) * self.rl_rate))) if self.lift_speed_mps > 0.0 else 1

            left_distance_m = abs(left_target_y_m - float(start_pos[1]))
            n_left = max(1, int(round((left_distance_m / self.lift_speed_mps) * self.rl_rate))) if self.lift_speed_mps > 0.0 else 1

            n_lower = max(1, int(round((lower_height_m / self.lift_speed_mps) * self.rl_rate)))

            final_z_offset = self.lift_height_m - lower_height_m

            # z: hold -> lift -> back (high) -> left (high) -> lower -> hold
            z_hold       = np.zeros((n_hold,), dtype=np.float32)
            z_up         = self._ramp(n_lift, 0.0, self.lift_height_m)
            z_back       = np.full((n_back,), self.lift_height_m, dtype=np.float32)
            z_left       = np.full((n_left,), self.lift_height_m, dtype=np.float32)
            z_down       = self._ramp(n_lower, self.lift_height_m, final_z_offset)
            z_final_hold = np.full((n_hold,), final_z_offset, dtype=np.float32)
            z = np.concatenate([z_hold, z_up, z_back, z_left, z_down, z_final_hold], axis=0)

            # x: fixed -> fixed during lift -> move back (-x) -> fixed during left -> fixed during lower -> hold
            target_x_m = float(start_pos[0]) - back_distance_m
            x_hold       = np.full((n_hold,), start_pos[0], dtype=np.float32)
            x_lift       = np.full((n_lift,), start_pos[0], dtype=np.float32)
            x_back       = self._ramp(n_back, start_pos[0], target_x_m)
            x_left       = np.full((n_left,), target_x_m, dtype=np.float32)
            x_lower      = np.full((n_lower,), target_x_m, dtype=np.float32)
            x_final_hold = np.full((n_hold,), target_x_m, dtype=np.float32)
            x = np.concatenate([x_hold, x_lift, x_back, x_left, x_lower, x_final_hold], axis=0)

            # y: fixed -> fixed during lift -> fixed during back -> move left (+y) -> fixed during lower -> hold
            y_hold       = np.full((n_hold,), start_pos[1], dtype=np.float32)
            y_lift       = np.full((n_lift,), start_pos[1], dtype=np.float32)
            y_back       = np.full((n_back,), start_pos[1], dtype=np.float32)
            y_left       = self._ramp(n_left, start_pos[1], left_target_y_m)
            y_lower      = np.full((n_lower,), left_target_y_m, dtype=np.float32)
            y_final_hold = np.full((n_hold,), left_target_y_m, dtype=np.float32)
            y = np.concatenate([y_hold, y_lift, y_back, y_left, y_lower, y_final_hold], axis=0)

            pos = np.repeat(start_pos.reshape(1, 3), z.shape[0], axis=0)
            pos[:, 0] = x
            pos[:, 1] = y
            pos[:, 2] = start_pos[2] + z

            quat = np.repeat(start_quat_wxyz.reshape(1, 4), z.shape[0], axis=0)
            return pos, quat

        # Default: "lift" mode
        lift_time_s = self.lift_height_m / self.lift_speed_mps
        n_lift = int(round(lift_time_s * self.rl_rate))
        n_lift = max(n_lift, 1)

        # z profile (absolute offset from start z)
        z_hold = np.zeros((n_hold,), dtype=np.float32)
        z_up = self._ramp(n_lift, 0.0, self.lift_height_m)
        z_final_hold = np.full((n_lift,), self.lift_height_m, dtype=np.float32)

        z = np.concatenate([z_hold, z_up, z_final_hold], axis=0)  # [T]

        pos = np.repeat(start_pos.reshape(1, 3), z.shape[0], axis=0)
        pos[:, 2] = start_pos[2] + z

        quat = np.repeat(start_quat_wxyz.reshape(1, 4), z.shape[0], axis=0)
        return pos, quat

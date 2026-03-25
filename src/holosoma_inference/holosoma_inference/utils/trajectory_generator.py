import numpy as np

MODES = ["lift", "right", "left", "left_down", "back", "stay"]
LEFT_TARGET_Y_M = 0.8
RIGHT_TARGET_Y_M = -LEFT_TARGET_Y_M
LEFT_DOWN_LOWER_HEIGHT_M = 0.8

class TrajectoryGenerator:
    def __init__(
        self,
        rl_rate: float,
        hold_time_s: float = 4.0,
        lift_height_m: float = 0.5,
        lift_speed_mps: float = 0.4,
        mode: str = "lift"
    ):
        self.rl_rate = float(rl_rate)
        self.hold_time_s = float(hold_time_s)
        self.lift_height_m = float(lift_height_m)
        self.lift_speed_mps = float(lift_speed_mps)
        self.mode = mode

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
            z_up = np.linspace(0.0, self.lift_height_m, n_lift + 1, dtype=np.float32)[1:]
            z_move = np.full((n_move,), self.lift_height_m, dtype=np.float32)
            z_final_hold = np.full((n_move,), self.lift_height_m, dtype=np.float32)
            z = np.concatenate([z_hold, z_up, z_move, z_final_hold], axis=0)

            y_hold = np.full((n_hold,), start_pos[1], dtype=np.float32)
            y_lift = np.full((n_lift,), start_pos[1], dtype=np.float32)
            y_move = np.linspace(start_pos[1], target_y_m, n_move + 1, dtype=np.float32)[1:]
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
            z_up = np.linspace(0.0, self.lift_height_m, n_lift + 1, dtype=np.float32)[1:]
            z_move = np.full((n_move,), self.lift_height_m, dtype=np.float32)
            final_z_offset = self.lift_height_m - lower_height_m
            z_down = np.linspace(self.lift_height_m, final_z_offset, n_lower + 1, dtype=np.float32)[1:]
            z_final_hold = np.full((n_hold,), final_z_offset, dtype=np.float32)
            z = np.concatenate([z_hold, z_up, z_move, z_down, z_final_hold], axis=0)

            # y profile: hold -> stay during lift -> move laterally -> stay during lower -> hold
            y_hold = np.full((n_hold,), start_pos[1], dtype=np.float32)
            y_lift = np.full((n_lift,), start_pos[1], dtype=np.float32)
            y_move = np.linspace(start_pos[1], target_y_m, n_move + 1, dtype=np.float32)[1:]
            y_lower = np.full((n_lower,), target_y_m, dtype=np.float32)
            y_final_hold = np.full((n_hold,), target_y_m, dtype=np.float32)
            y = np.concatenate([y_hold, y_lift, y_move, y_lower, y_final_hold], axis=0)

            pos = np.repeat(start_pos.reshape(1, 3), z.shape[0], axis=0)
            pos[:, 1] = y
            pos[:, 2] = start_pos[2] + z

            quat = np.repeat(start_quat_wxyz.reshape(1, 4), z.shape[0], axis=0)
            return pos, quat

        lift_time_s = self.lift_height_m / self.lift_speed_mps
        n_lift = int(round(lift_time_s * self.rl_rate))
        n_lift = max(n_lift, 1)

        # z profile (absolute offset from start z)
        # hold: 0
        z_hold = np.zeros((n_hold,), dtype=np.float32)

        # up: 0 -> +H (constant speed in continuous time; discretized linear ramp)
        z_up = np.linspace(0.0, self.lift_height_m, n_lift + 1, dtype=np.float32)[1:]

        # keep the final pose instead of returning to the start point
        z_final_hold = np.full((n_lift,), self.lift_height_m, dtype=np.float32)

        z = np.concatenate([z_hold, z_up, z_final_hold], axis=0)  # [T]

        pos = np.repeat(start_pos.reshape(1, 3), z.shape[0], axis=0)
        pos[:, 2] = start_pos[2] + z

        quat = np.repeat(start_quat_wxyz.reshape(1, 4), z.shape[0], axis=0)
        return pos, quat

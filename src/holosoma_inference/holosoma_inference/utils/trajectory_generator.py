import numpy as np

MODES = ["lift", "right", "left", "back"]

class TrajectoryGenerator:
    def __init__(
        self,
        rl_rate: float,
        hold_time_s: float = 2.5,
        lift_height_m: float = 0.5,
        lift_speed_mps: float = 0.8,
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

        lift_time_s = self.lift_height_m / self.lift_speed_mps
        n_lift = int(round(lift_time_s * self.rl_rate))
        n_lift = max(n_lift, 1)

        # z profile (absolute offset from start z)
        # hold: 0
        z_hold = np.zeros((n_hold,), dtype=np.float32)

        # up: 0 -> +H (constant speed in continuous time; discretized linear ramp)
        z_up = np.linspace(0.0, self.lift_height_m, n_lift + 1, dtype=np.float32)[1:]

        # down: +H -> 0
        z_down = np.linspace(self.lift_height_m, 0.0, n_lift + 1, dtype=np.float32)[1:]

        z = np.concatenate([z_hold, z_up, z_down], axis=0)  # [T]

        pos = np.repeat(start_pos.reshape(1, 3), z.shape[0], axis=0)
        pos[:, 2] = start_pos[2] + z

        quat = np.repeat(start_quat_wxyz.reshape(1, 4), z.shape[0], axis=0)
        return pos, quat
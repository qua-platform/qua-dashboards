from qua_dashboards.utils.qua_types import QuaVariableFloat
from qua_dashboards.video_mode.scan_modes.scan_mode import ScanMode


import numpy as np
from qm.qua import assign, declare, fixed, for_, if_


from typing import Generator, Sequence, Tuple


class SpiralScan(ScanMode):
    """Spiral scan mode.

    The spiral scan mode is a scan mode that scans the grid in a spiral pattern.
    """

    def get_idxs(self, x_points: int, y_points: int) -> Tuple[np.ndarray, np.ndarray]:
        assert x_points == y_points, "Spiral only works for square grids"

        num_half_spirals = x_points
        x_idx = x_points // 2
        y_idx = y_points // 2

        idxs_x = [x_idx]
        idxs_y = [y_idx]

        for half_spiral_idx in range(num_half_spirals):
            initial_direction_RL = "L" if half_spiral_idx % 2 else "R"
            direction_UD = "U" if half_spiral_idx % 2 else "D"
            direction_LR = "R" if half_spiral_idx % 2 else "L"

            if half_spiral_idx:
                x_idx += 1 if initial_direction_RL == "R" else -1
                idxs_x.append(x_idx)
                idxs_y.append(y_idx)

            for _ in range(half_spiral_idx):
                y_idx += 1 if direction_UD == "U" else -1
                idxs_x.append(x_idx)
                idxs_y.append(y_idx)

            for _ in range(half_spiral_idx):
                x_idx += 1 if direction_LR == "R" else -1
                idxs_x.append(x_idx)
                idxs_y.append(y_idx)

        return np.array(idxs_x), np.array(idxs_y)

    def scan(
        self, x_vals: Sequence[float], y_vals: Sequence[float], x_kind, y_kind
    ) -> Generator[Tuple[QuaVariableFloat, QuaVariableFloat], None, None]:
        movement_direction = declare(fixed)
        half_spiral_idx = declare(int)
        k = declare(int)
        x = declare(int) if x_kind == "frequency" else declare(fixed)
        y = declare(int) if y_kind == "frequency" else declare(fixed)
        voltages = {"x": x, "y": y}

        assert len(x_vals) == len(y_vals), (
            f"x_vals and y_vals must have the same length ({len(x_vals)} != {len(y_vals)})"
        )
        num_half_spirals = len(x_vals)
        x_step = x_vals[1] - x_vals[0]
        y_step = y_vals[1] - y_vals[0]

        assign(movement_direction, -1.0)
        assign(x, 0.0)
        assign(y, 0.0)
        yield voltages["x"], voltages["y"]

        with for_(
            half_spiral_idx, 0, half_spiral_idx < num_half_spirals, half_spiral_idx + 1
        ):  # type: ignore
            # First take one step in the opposite XY direction
            with if_(half_spiral_idx > 0):  # type: ignore
                assign(x, x - x_step * movement_direction)  # type: ignore
                yield voltages["x"], voltages["y"]

            with for_(k, 0, k < half_spiral_idx, k + 1):  # type: ignore
                assign(y, y + y_step * movement_direction)  # type: ignore
                yield voltages["x"], voltages["y"]

            with for_(k, 0, k < half_spiral_idx, k + 1):  # type: ignore
                assign(x, x + x_step * movement_direction)  # type: ignore
                yield voltages["x"], voltages["y"]

            assign(movement_direction, -movement_direction)  # type: ignore

        assign(x, 0)
        assign(y, 0)

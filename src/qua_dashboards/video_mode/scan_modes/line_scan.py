
from typing import Generator, Sequence, Tuple
import numpy as np
from qm.qua import declare, fixed, for_each_
from qua_dashboards.video_mode.scan_modes import ScanMode
from qua_dashboards.utils.qua_types import QuaVariableFloat

class LineScan(ScanMode):
    """
    Line scan mode.

    Identifies X and Y coordinates for a linescan.
    """

    def get_idxs(self, x_points: int, y_points: int) -> Tuple[np.ndarray, np.ndarray]:
        # MINIMAL FIX: map N samples onto a 1×N frame (row 0)
        x_idxs = np.arange(x_points, dtype=int)
        y_idxs = np.zeros(x_points, dtype=int)
        return x_idxs, y_idxs

    def scan(
        self, x_vals: Sequence[float], y_vals: Sequence[float]
    ) -> Generator[Tuple[QuaVariableFloat, QuaVariableFloat], None, None]:

        x_list = list(x_vals)
        if len(y_vals) == len(x_vals):
            y_list = list(y_vals)
        else:
            base_y = float(y_vals[0]) if len(y_vals) else 0.0
            y_list = [base_y] * len(x_list)

        qx = declare(fixed)
        qy = declare(fixed)
        with for_each_((qx, qy), (x_list, y_list)):
            yield qx, qy


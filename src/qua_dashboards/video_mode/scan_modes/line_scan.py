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
        self, x_vals: Sequence[float], y_vals: Sequence[float], x_kind, y_kind
    ) -> Generator[Tuple[QuaVariableFloat], None, None]:

        x_list = list(x_vals)

        qx = declare(int) if x_kind == "Frequency" else declare(fixed)
        with for_each_((qx), (x_list)):
            yield qx, None

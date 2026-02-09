from qua_dashboards.utils.qua_types import QuaVariableFloat
from qua_dashboards.video_mode.scan_modes.scan_mode import ScanMode


import numpy as np
from qm.qua import declare, fixed, for_
from qualang_tools.loops import from_array


from typing import Generator, Sequence, Tuple, Callable


class RasterScan(ScanMode):
    """Raster scan mode.

    The raster scan mode is a simple scan mode that scans the grid in a raster pattern.
    """

    def get_idxs(self, x_points: int, y_points: int) -> Tuple[np.ndarray, np.ndarray]:
        x_idxs = np.tile(np.arange(x_points), y_points)
        y_idxs = np.repeat(np.arange(y_points), x_points)
        return x_idxs, y_idxs

    def scan(
        self, x_vals: Sequence[float], y_vals: Sequence[float], x_mode: str = None, y_mode: str = None, compensation_pulse: Callable = None,
    ) -> Generator[Tuple[QuaVariableFloat, QuaVariableFloat], None, None]:
        voltages = {"x": declare(int) if x_mode == "Frequency" else declare(fixed), "y": declare(int) if y_mode == "Frequency" else declare(fixed)}
        

        with for_(*from_array(voltages["y"], y_vals)):  # type: ignore
            with for_(*from_array(voltages["x"], x_vals)):  # type: ignore
                yield voltages["x"], voltages["y"]
            if compensation_pulse is not None: 
                compensation_pulse()

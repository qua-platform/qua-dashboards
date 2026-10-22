from typing import Callable, Generator, Sequence, Tuple, Literal

import numpy as np
from qm.qua import assign, declare, fixed, for_
from qualang_tools.loops import from_array

from qua_dashboards.utils.qua_types import QuaVariableFloat
from qua_dashboards.video_mode.scan_modes.scan_mode import ScanMode


class RasterScan(ScanMode):
    """Raster scan with selectable fast-axis (X) acquisition direction."""

    def __init__(
        self,
        component_id: str = "raster-scan",
        fast_axis_direction: Literal["forward", "reverse"] = "forward"
    ):
        super().__init__(component_id=component_id)
        self.fast_axis_direction = fast_axis_direction

    @property
    def fast_axis_direction(self) -> str:
        return self._fast_axis_direction

    @fast_axis_direction.setter
    def fast_axis_direction(self, value: str) -> None:
        value = str(value).lower()
        if value not in ("forward", "reverse"):
            raise ValueError("fast_axis_direction must be 'forward' or 'reverse'")
        self._fast_axis_direction = value

    @property
    def reverse_fast_axis(self) -> bool:
        return self.fast_axis_direction == "reverse"

    def _x_scan_values(self, x_vals: Sequence[float]) -> np.ndarray:
        values = np.asarray(x_vals)
        return values[::-1] if self.reverse_fast_axis else values

    def get_idxs(self, x_points: int, y_points: int) -> Tuple[np.ndarray, np.ndarray]:
        x_row = np.arange(x_points)
        if self.reverse_fast_axis:
            x_row = x_row[::-1]
        x_idxs = np.tile(x_row, y_points)
        y_idxs = np.repeat(np.arange(y_points), x_points)
        return x_idxs, y_idxs

    def scan(
        self,
        x_vals: Sequence[float],
        y_vals: Sequence[float],
        x_mode: str = None,
        y_mode: str = None,
        compensation_pulse: Callable = None,
        settle_action: Callable = None,
    ) -> Generator[Tuple[QuaVariableFloat, QuaVariableFloat], None, None]:
        voltages = {
            "x": declare(int) if x_mode == "Frequency" else declare(fixed),
            "y": declare(int) if y_mode == "Frequency" else declare(fixed),
        }
        scan_x_vals = self._x_scan_values(x_vals)

        with for_(*from_array(voltages["y"], y_vals)):  # type: ignore
            if settle_action is not None:
                assign(voltages["x"], scan_x_vals[0])
                settle_action(voltages["x"], voltages["y"])
            with for_(*from_array(voltages["x"], scan_x_vals)):  # type: ignore
                yield voltages["x"], voltages["y"]
            if compensation_pulse is not None:
                compensation_pulse()

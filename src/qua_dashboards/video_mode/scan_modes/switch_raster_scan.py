from typing import Callable, Generator, Sequence, Tuple, Literal

import numpy as np
from qm.qua import assign, declare, fixed, for_, for_each_
from qualang_tools.loops import from_array

from qua_dashboards.utils.qua_types import QuaVariableFloat
from qua_dashboards.video_mode.scan_modes.scan_mode import ScanMode


class SwitchRasterScan(ScanMode):
    """Switch-raster scan with selectable fast-axis (X) acquisition direction.

    ``start_from_middle`` controls the slow-axis (Y) ordering.
    ``fast_axis_direction`` independently controls whether every X row is
    acquired low-to-high (forward) or high-to-low (reverse).
    """

    def __init__(
        self,
        component_id: str = "switch-raster-scan",
        start_from_middle: bool = True,
        fast_axis_direction: Literal["forward", "reverse"] = "forward"
    ):
        super().__init__(component_id=component_id)
        self.start_from_middle = start_from_middle
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

    @staticmethod
    def interleave_arr(arr: np.ndarray, start_from_middle: bool = True) -> np.ndarray:
        mid_idx = len(arr) // 2
        if len(arr) % 2:
            interleaved = [arr[mid_idx]]
            arr1 = arr[mid_idx + 1 :]
            arr2 = arr[mid_idx - 1 :: -1]
            interleaved += [elem for pair in zip(arr1, arr2) for elem in pair]
        else:
            arr1 = arr[mid_idx:]
            arr2 = arr[mid_idx - 1 :: -1]
            interleaved = [elem for pair in zip(arr1, arr2) for elem in pair]

        if not start_from_middle:
            interleaved = interleaved[::-1]
        return np.array(interleaved)

    def get_idxs(self, x_points: int, y_points: int) -> Tuple[np.ndarray, np.ndarray]:
        y_idxs = self.interleave_arr(
            np.arange(y_points), start_from_middle=self.start_from_middle
        )
        x_row = np.arange(x_points)
        if self.reverse_fast_axis:
            x_row = x_row[::-1]
        x_idxs = np.tile(x_row, y_points)
        y_idxs = np.repeat(y_idxs, x_points)
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

        with for_each_(
            voltages["y"],
            self.interleave_arr(
                np.asarray(y_vals), start_from_middle=self.start_from_middle
            ),
        ):  # type: ignore
            if settle_action is not None:
                assign(voltages["x"], scan_x_vals[0])
                settle_action(voltages["x"], voltages["y"])
            with for_(*from_array(voltages["x"], scan_x_vals)):  # type: ignore
                yield voltages["x"], voltages["y"]
            if compensation_pulse is not None:
                compensation_pulse()

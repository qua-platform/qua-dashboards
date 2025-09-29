from qua_dashboards.utils.qua_types import QuaVariableFloat
from qua_dashboards.video_mode.scan_modes.scan_mode import ScanMode


import numpy as np
from qm.qua import declare, fixed, for_, for_each_
from qualang_tools.loops import from_array


from typing import Generator, Sequence, Tuple


class SwitchRasterScan(ScanMode):
    """Switch raster scan mode.

    The switch raster scan mode is a scan mode that scans the grid in a raster pattern,
    but the direction of the scan is switched after each row or column.
    This is useful when the scan length is similar to the bias tee frequency.

    Args:
        start_from_middle: Whether to start the scan from the middle of the array.
            For an array centered around 0, the scan will start with 0 and progressively increase in amplitude.
    """

    def __init__(
        self, component_id: str = "switch-raster-scan", start_from_middle: bool = True
    ):
        super().__init__(component_id=component_id)
        self.start_from_middle = start_from_middle

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
        x_idxs = np.tile(np.arange(x_points), y_points)
        y_idxs = np.repeat(y_idxs, x_points)
        return x_idxs, y_idxs

    def scan(
        self, x_vals: Sequence[float], y_vals: Sequence[float], x_kind: str = None, y_kind: str = None
    ) -> Generator[Tuple[QuaVariableFloat, QuaVariableFloat], None, None]:
        voltages = {"x": declare(int) if x_kind == "Frequency" else declare(fixed), "y": declare(int) if y_kind == "Frequency" else declare(fixed)}

        with for_each_(
            voltages["y"],
            self.interleave_arr(y_vals, start_from_middle=self.start_from_middle),
        ):  # type: ignore
            with for_(*from_array(voltages["x"], x_vals)):  # type: ignore
                yield voltages["x"], voltages["y"]
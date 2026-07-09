from typing import Optional, Dict, Any
from qm.qua import wait
from dash import Dash
import numpy as np
from qua_dashboards.core import ModifiedFlags
from qua_dashboards.utils.dash_utils import create_input_field
from .base_sweep_axis import BaseSweepAxis
from qua_dashboards.utils.qua_types import QuaVariableFloat
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from qua_dashboards.video_mode.inner_loop_actions import InnerLoopAction

__all__ = ["TimeSweepAxis"]

DEFAULT_TIME_SPAN = 10000
DEFAULT_TIME_POINTS = 51


class TimeSweepAxis(BaseSweepAxis):
    """Class representing a Time sweep axis. 
    Takes a pulse object. apply() waits for the next pixel. The time span is defined in nanoseconds. 

    Attributes:
        name: Name of the axis.
        span: Span of the axis.
        points: Number of points in the sweep.
        label: Label of the axis.
        units: Units of the axis.
        offset_parameter: Parameter which provides the offset; in this case, should be the pulse object.
    """

    def __init__(
        self,
        name,
        span: Optional[float] = None,
        points: Optional[int] = None,
        component_id: Optional[str] = None,
        offset_parameter=None,
    ):
        super().__init__(
            component_id=component_id,
            name=name,
            span=span or DEFAULT_TIME_SPAN,
            points=points or DEFAULT_TIME_POINTS,
            units="ns",
            offset_parameter=offset_parameter,
        )
        self._coord_name = f"{name}_time"
        self.total_readout_duration = None

    @property
    def offset(self):
        return abs(self.span/2)

    @property
    def sweep_values_with_offset(self):
        """Returns axis sweep values with offset."""
        return self.sweep_values + self.offset
    
    def configure_loop(self, inner_loop_action: "InnerLoopAction"): 
        """Pass the InnerLoopAction here so that we can calculate the per-pixel action"""
        super().configure_loop(inner_loop_action)
        if inner_loop_action.y_axis is not None: 
            raise ValueError("2D Sweep detected, where one axis is Time. Please perform Time sweep as a 1D sweep.")
        _readout_overhead_ns = sum(
            inner_loop_action._pulse_for(ch).length
            for ch in inner_loop_action.selected_readout_channels
        )
        self.total_readout_duration = _readout_overhead_ns

    @property
    def wait_time(self) -> int: 
        """
        We can calculate the wait time here, based on the readout times of the selected readouts. 
        Eventual structure would be: 
            - Calculate the time step size 
            - Subtract the readout lengths
            - Potentially subtract the hold durations? 
        """
        total_pixel_duration = self.sweep_values_with_offset[1] - self.sweep_values_with_offset[0]
        wait_time = total_pixel_duration - self.total_readout_duration

        if wait_time < 0: 
            necessary_span_for_points = self.points * self.total_readout_duration
            raise ValueError(f"Readout time exceeds per-pixel duration based on SPAN and POINTS. Increase span to {necessary_span_for_points} to perform accurate sweep with the same resolution.")

        return int(round(wait_time / 4)) * 4
        
    @property
    def qua_sweep_values(self) -> np.ndarray:
        """Returns the actual array to be processed by the DataAcquirer"""
        return np.array([int(round(k)) for k in self.sweep_values_with_offset])

    def register_callbacks(self, app: Dash) -> None:
        pass

    def apply(self, value: QuaVariableFloat) -> None:
        """
        Applies the update_frequency command.
        """
        wait(self.wait_time // 4)
        return {}

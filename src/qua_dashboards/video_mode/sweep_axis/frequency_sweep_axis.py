from typing import Optional, Dict, Any
from qm.qua import update_frequency
from dash import Dash
import numpy as np
from qua_dashboards.core import ModifiedFlags
from qua_dashboards.utils.dash_utils import create_input_field
from .base_sweep_axis import BaseSweepAxis
from qua_dashboards.utils.qua_types import QuaVariableFloat

__all__ = ["FrequencySweepAxis"]

DEFAULT_FREQ_SPAN = 1e6
DEFAULT_FREQ_POINTS = 51


class FrequencySweepAxis(BaseSweepAxis):
    """Class representing a sweep axis.

    Attributes:
        name: Name of the axis.
        span: Span of the axis.
        points: Number of points in the sweep.
        label: Label of the axis.
        units: Units of the axis.
        offset_parameter: The Pulse object of the Axis which provides the offset.
        attenuation: Attenuation of the axis (0 by default)
    """

    def __init__(
        self,
        name,
        span: Optional[float] = None,
        points: Optional[int] = None,
        component_id: Optional[str] = None,
        offset_parameter = None
    ):
        super().__init__(component_id=component_id, 
                         name = name, 
                         span = span or DEFAULT_FREQ_SPAN, 
                         points = points or DEFAULT_FREQ_POINTS, 
                         units = "Hz", 
                         offset_parameter = offset_parameter)
        self._coord_name = f"{name}_freq"
        
    @property
    def offset(self): 
        return self.offset_parameter.channel.intermediate_frequency

    @property
    def sweep_values_with_offset(self):
        """Returns axis sweep values with offset."""
        return self.sweep_values + self.offset
    
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
        update_frequency(self.name, value)
        return {}

    def create_axis_layout(self, min_span: float, max_span: Optional[float] = None):
        col = super().create_axis_layout(min_span = min_span, max_span = max_span)
        # Use pattern-matching IDs instead of regular IDs
        ids = {
            "offset": {"type": "number-input", "index": f"{self.component_id}::offset"}, 
        }
        offset_input = (
            create_input_field(
                id=ids["offset"],
                label="Offset",
                value=self.offset,
                input_style={"width": "150px"},
                units=self.units if self.units is not None else "",
            )
        )
        body = col.children.children[0]
        body.children.extend([
            offset_input
        ])
        
        return col

    def update_parameters(self, parameters: Dict[str, Dict[str, Any]]) -> ModifiedFlags:
        """
        Updates 2D data acquirer parameters (axes, averages).
        """
        flags = super().update_parameters(parameters)

        if self.component_id not in parameters:
            return flags

        params = parameters[self.component_id]

        if "offset" in params and self.offset != params["offset"]:
            self.offset_parameter.channel.intermediate_frequency = params["offset"]
            flags |= ModifiedFlags.PARAMETERS_MODIFIED | ModifiedFlags.PROGRAM_MODIFIED

        return flags

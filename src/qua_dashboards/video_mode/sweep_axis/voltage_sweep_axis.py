from typing import Optional, Dict, Any
from dash import Dash
from dash.development.base_component import Component
import numpy as np
import dash_bootstrap_components as dbc
from qualang_tools.units.units import unit
from qua_dashboards.core import BaseUpdatableComponent, ModifiedFlags
from qua_dashboards.utils.basic_parameter import BasicParameter
from qua_dashboards.utils.dash_utils import create_input_field
from qua_dashboards.utils.qua_types import QuaVariableFloat
from .base_sweep_axis import BaseSweepAxis
from qm.qua import *

__all__ = ["VoltageSweepAxis"]

DEFAULT_VOLTAGE_SPAN = 0.03
DEFAULT_VOLTAGE_POINTS = 51


class VoltageSweepAxis(BaseSweepAxis):
    """Class representing a sweep axis.

    Attributes:
        name: Name of the axis.
        span: Span of the axis.
        points: Number of points in the sweep.
        label: Label of the axis.
        units: Units of the axis.
        offset_parameter: Offset parameter of the axis.
        attenuation: Attenuation of the axis (0 by default)
    """

    def __init__(
        self,
        name: str,
        offset_parameter = None,
        span: Optional[float] = None,
        points: Optional[int] = None,
        attenuation: float = 0,
        component_id: Optional[str] = None,
    ):
        super().__init__(component_id=component_id, 
                         name = name, 
                         span = span or DEFAULT_VOLTAGE_SPAN, 
                         points = points or DEFAULT_VOLTAGE_POINTS, 
                         units = "V", 
                         offset_parameter = offset_parameter)
        self.attenuation = attenuation
        self._coord_name = f"{name}_volts"

    @property
    def sweep_values_unattenuated(self):
        """Returns axis sweep values without attenuation."""
        return self.sweep_values * 10 ** (self.attenuation / 20)

    @property
    def sweep_values_with_offset(self):
        """Returns axis sweep values with offset."""
        return self.sweep_values_unattenuated + self.offset_parameter() if self.offset_parameter is not None else self.sweep_values_unattenuated
    
    @property 
    def qua_sweep_values(self) -> np.ndarray: 
        """Returns the actual array to be processed by the DataAcquirer"""
        return np.array(self.sweep_values_unattenuated)

    @property
    def scale(self):
        """Returns axis scale factor, calculated from attenuation."""
        return 10 ** (-self.attenuation / 20)
    
    def declare_vars(self): 
        self.last_val = declare(fixed)
        self.slope = declare(fixed)
        self.loop_current = declare(fixed)
        self.loop_past = declare(fixed)

    def gather_contributions(self, target_value: QuaVariableFloat): 
        out: Dict[str, Dict[str, QuaVariableFloat]] = {
            "volt_levels" : {self.name: (target_value>>12) <<12}, 
            "last_levels" : {self.name: (self.last_val>>12) <<12}, 
            "freq_updates" : {}, 
            "amplitude_scales" : {}
        }
        return out
        
    def apply(self, value: QuaVariableFloat): 
        assign(self.last_val, (value>>12)<<12)
        return


    
    def update_parameters(self, parameters: Dict[str, Dict[str, Any]]) -> ModifiedFlags:
        """
        Updates data acquirer parameters (axes, averages).
        """
        flags = super().update_parameters(parameters)

        return flags

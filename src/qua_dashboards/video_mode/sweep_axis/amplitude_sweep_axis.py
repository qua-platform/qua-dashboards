from typing import Optional, Dict, Any
from qm.qua import declare, assign
from dash import Dash
import numpy as np
import dash_bootstrap_components as dbc
from qualang_tools.units.units import unit
from qua_dashboards.core import ModifiedFlags
from qua_dashboards.utils.dash_utils import create_input_field
from .base_sweep_axis import BaseSweepAxis
from qm.qua import *
from qua_dashboards.utils.qua_types import QuaVariableFloat

__all__ = ["AmplitudeSweepAxis"]

DEFAULT_AMP_SPAN = 0.01
DEFAULT_AMP_POINTS = 51


class AmplitudeSweepAxis(BaseSweepAxis):
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
        name,
        span: Optional[float] = None,
        points: Optional[int] = None,
        component_id: Optional[str] = None,
        offset_parameter = None
    ):
        super().__init__(component_id=component_id, 
                         name = name, 
                         span = span or DEFAULT_AMP_SPAN, 
                         points = points or DEFAULT_AMP_POINTS, 
                         units = "V", 
                         offset_parameter = offset_parameter)
        self.dbm: bool = False
        self._coord_name = f"{name}_amp"


    @property
    def offset(self): 
        # Keep the dBm mode as just a mask for the parameter, never actually setting the amplitude to the dBm value. 
        if self.dbm:
            return unit.volts2dBm(self.offset_parameter.amplitude)
        return self.offset_parameter.amplitude

    def _offset_volts(self) -> float:
        return unit.dBm2volts(self.offset) if self.dbm else float(self.offset)
    
    @property
    def sweep_values_with_offset(self):
        """Returns axis sweep values with offset."""
        return self.sweep_values + self.offset
    
    @property 
    def qua_sweep_values(self) -> np.ndarray: 
        """Returns the actual array to be processed by the DataAcquirer"""
        if self.dbm:
            return np.asarray(unit.dBm2volts(self.sweep_values_with_offset))
        return np.asarray(self.sweep_values_with_offset)


    def register_callbacks(self, app: Dash) -> None:
        pass

    def create_axis_layout(self, min_span: float, max_span: Optional[float] = None):
        """Modified to use pattern-matching IDs"""


        col = super().create_axis_layout(min_span = min_span, max_span = max_span)
        # Use pattern-matching IDs instead of regular IDs
        ids = {
            "offset": {"type": "number-input", "index": f"{self.component_id}::offset"}, 
            "dbm-toggle": {"type": "toggle", "index": f"{self.component_id}::dbm-toggle"}
        }
        toggle = dbc.Row(
                    [dbc.Col("V", width="auto", className="me-2"),
                    dbc.Col(
                        dbc.Checklist(
                            id=ids["dbm-toggle"],
                            options=[{"label": "", "value": "on"}],
                            value=["on"] if self.dbm else [],
                            switch=True,
                        ),
                        width="auto",
                    ),
                    dbc.Col("dBm", width="auto", className="ms-2"),],
                    className="align-items-center g-1"
                    )
        
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
            offset_input, 
            toggle
        ])
        
        return col
    
    def declare_vars(self): 
        self.scale_var = declare(fixed)
    
    def apply(self, value: QuaVariableFloat): 
        """ 
        Apply command. Calculates the amplitude scale necessary, and returns it as a dict to be handled by the inner loop. 
        """
        default_amp_v = float(min(max(self._offset_volts(), 0.0), 1.999))
        assign(self.scale_var, value / default_amp_v)
        return {self.name: self.scale_var}

    def update_parameters(self, parameters: Dict[str, Dict[str, Any]]) -> ModifiedFlags:
        """
        Updates data acquirer parameters (axes, averages).
        """
        flags = super().update_parameters(parameters)

        params = parameters[self.component_id]

        if "offset" in params and self.offset != params["offset"]:
            if self.dbm: 
                self.offset_parameter.amplitude = unit.dBm2volts(params["offset"])
            else: 
                self.offset_parameter.amplitude = params["offset"]
            flags |= ModifiedFlags.PARAMETERS_MODIFIED | ModifiedFlags.PROGRAM_MODIFIED

        if "dbm-toggle" in params: 
            raw = params["dbm-toggle"]
            new_dbm = bool(raw) if isinstance(raw, (list, tuple, set)) else (raw == "on")

            if self.dbm != new_dbm: 
                off = self.offset
                spn = self.span

                if new_dbm:
                    # We do not set the offset to the dBm value here, instead we rely on self.offset to mask the value whenever it is called, based on its own dBm bool. 
                    vmin, vmax = off - spn/2, off + spn/2
                    pmin, pmax = unit.volts2dBm(vmin), unit.volts2dBm(vmax)
                    span_db = pmax-pmin
                    self.span = span_db
                else:
                    pmin, pmax = off-spn/2, off+spn/2
                    vmin, vmax = unit.dBm2volts(pmin), unit.dBm2volts(pmax)
                    center_v = (vmax+vmin) / 2
                    span_v = vmax - vmin
                    self.offset_parameter.amplitude = center_v
                    self.span = span_v

            self.dbm = new_dbm
            flags |= ModifiedFlags.PARAMETERS_MODIFIED | ModifiedFlags.PROGRAM_MODIFIED
        return flags

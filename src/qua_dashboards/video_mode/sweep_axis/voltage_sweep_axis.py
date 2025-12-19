from typing import Optional, Dict, Any
import numpy as np
from qua_dashboards.core import ModifiedFlags
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
        offset_parameter=None,
        span: Optional[float] = None,
        points: Optional[int] = None,
        attenuation: float = 0,
        component_id: Optional[str] = None,
        label: str = None,
    ):
        super().__init__(
            component_id=component_id,
            name=name,
            span=span or DEFAULT_VOLTAGE_SPAN,
            points=points or DEFAULT_VOLTAGE_POINTS,
            units="V",
            offset_parameter=offset_parameter,
        )
        self.attenuation = attenuation
        self._coord_name = f"{name}_volts"

    @property
    def sweep_values_unattenuated(self):
        """Returns axis sweep values without attenuation."""
        return self.sweep_values

    @property
    def sweep_values_with_offset(self):
        """Returns axis sweep values with offset."""
        offset = 0.0
        if self.offset_parameter is not None:
            try:
                offset = self.offset_parameter.get_latest()
            except:
                offset = 0.0
        return self.sweep_values_unattenuated + offset

    @property
    def qua_sweep_values(self) -> np.ndarray:
        """Returns the actual array to be processed by the DataAcquirer"""
        return np.array(self.sweep_values_unattenuated)

    @property
    def scale(self):
        """Returns axis scale factor, calculated from attenuation."""
        return 10 ** (-self.attenuation / 20)

    def declare_vars(self):
        self.val = declare(fixed)

    def apply(self, value: QuaVariableFloat) -> None:
        """
        Apply command. Currently just updates the last voltage tracker
        """
        # assign(self.last_val, (value >> 12) << 12)
        assign(self.val, (value >> 12) << 12)
        return {"voltage": {self.name: self.val}}

    def update_parameters(self, parameters: Dict[str, Dict[str, Any]]) -> ModifiedFlags:
        """
        Updates data acquirer parameters (axes, averages).
        """
        flags = super().update_parameters(parameters)

        return flags

from typing import Optional

from dash import Dash
from dash.development.base_component import Component
import numpy as np

from qua_dashboards.core import BaseUpdatableComponent
from qua_dashboards.utils.basic_parameter import BasicParameter
from qua_dashboards.video_mode.utils.dash_utils import create_axis_layout

__all__ = ["SweepAxis"]


class SweepAxis(BaseUpdatableComponent):
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
        span: float,
        points: int,
        label: Optional[str] = None,
        units: Optional[str] = None,
        offset_parameter: Optional[BasicParameter] = None,
        attenuation: float = 0,
        component_id: Optional[str] = None,
    ):
        if component_id is None:
            component_id = f"{name}-axis"
        super().__init__(component_id=component_id)
        self.name = name
        self.span = span
        self.points = points
        self.label = label
        self.units = units
        self.offset_parameter = offset_parameter
        self.attenuation = attenuation

    @property
    def sweep_values(self):
        """Returns axis sweep values using span and points."""
        return np.linspace(-self.span / 2, self.span / 2, self.points)

    @property
    def sweep_values_unattenuated(self):
        """Returns axis sweep values without attenuation."""
        return self.sweep_values * 10 ** (self.attenuation / 20)

    @property
    def sweep_values_with_offset(self):
        """Returns axis sweep values with offset."""
        if self.offset_parameter is None:
            return self.sweep_values_unattenuated
        return self.sweep_values_unattenuated + self.offset_parameter.get_latest()

    @property
    def scale(self):
        """Returns axis scale factor, calculated from attenuation."""
        return 10 ** (-self.attenuation / 20)

    def get_layout(self) -> Component | None:
        return create_axis_layout(
            axis="x",
            component_id=self.component_id,
            span=self.span,
            points=self.points,
            min_span=0.001,
            max_span=None,
            units=self.units,
        )

    def register_callbacks(self, app: Dash) -> None:
        pass

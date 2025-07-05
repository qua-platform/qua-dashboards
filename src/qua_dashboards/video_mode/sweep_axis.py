from typing import Optional, Dict, Any, Union

from dash import Dash
from dash.development.base_component import Component
import numpy as np
import dash_bootstrap_components as dbc

from qua_dashboards.core import BaseUpdatableComponent, ModifiedFlags
from qua_dashboards.utils.basic_parameter import BasicParameter
from qua_dashboards.utils.dash_utils import create_input_field

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
        span: Union[BasicParameter, float],
        points: Union[BasicParameter, int],
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
        # #self.span = span
        # self._span = span
        # self._points = points
        self.label = label
        self.units = units
        self.offset_parameter = offset_parameter
        self.attenuation = attenuation
        self._override_unattenuated: np.ndarray | None = None

        if isinstance(span, BasicParameter):
            self._span = span
        else:
            self._span = BasicParameter(
                name = f'{name}_span',
                label = label or name,
                units = units or '',
                initial_value = span
            )
        if isinstance(points, BasicParameter):
            self._points = points
        else:
            self._points = BasicParameter(
                name = f'{name}_points', 
                label = label or name,
                units = units or '',
                initial_value = points
            )
            
    @property
    def span(self) -> float:
        return self._span._value

    @span.setter
    def span(self, val: float):
        self._span._value = val

    @property
    def points(self):
        return self._points._value

    @points.setter
    def points(self, val):
        self._points._value = val

    @property
    def sweep_values(self):
        """Returns axis sweep values using span and points."""
        return np.linspace(-self.span / 2, self.span / 2, self.points)

    @property
    def sweep_values_unattenuated(self):
        """Returns axis sweep values without attenuation."""
        if self._override_unattenuated is not None: 
            return self._override_unattenuated
        return self.sweep_values * 10 ** (self.attenuation / 20)

    @sweep_values_unattenuated.setter
    def sweep_values_unattenuated(self, values:list[float] | np.ndarray):
        self._override_unattenuated = np.array(values)


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
        return self.create_axis_layout(
            min_span=0.001,
            max_span=None,
        )

    def register_callbacks(self, app: Dash) -> None:
        # Delegate to the BasicParameter instances, which know how to hook into the
        # Dash callback machinery and flip the ModifiedFlags for us.
        self._span.register_callbacks(app)
        self._points.register_callbacks(app)

    def create_axis_layout(
        self,
        min_span: float,
        max_span: Optional[float] = None,
    ):
        header_id = self._get_id('header')

        if not self.name.replace("_", "").isalnum():
            raise ValueError(
                f"Axis {self.name} must only contain alphanumeric characters and underscores."
            )
        ids = {
            "span": self._get_id("span"),
            "points": self._get_id("points"),
        }
        return dbc.Col(
            dbc.Card(
                [
                    #dbc.CardHeader(self.name.upper(), className="text-light"),
                    # dbc.CardHeader(
                    #     id=header_id,
                    #     children=(self.label or self.name).upper(),
                    #     className="text-light",
                    # ),
                    dbc.CardHeader(
                        id= f"{self.component_id}-header-text",
                        children=(self.label or self.name).upper(),
                        className="text-light",
                    ),
                    
                    
                    dbc.CardBody(
                        [
                            create_input_field(
                                id=ids["span"],
                                label="Span",
                                value=self.span,
                                min=min_span,
                                max=max_span,
                                input_style={"width": "100px"},
                                units=self.units if self.units is not None else "",
                            ),
                            create_input_field(
                                id=ids["points"],
                                label="Points",
                                value=self.points,
                                min=1,
                                max=501,
                                step=1,
                            ),
                        ],
                        className="text-light",
                    ),
                ],
                color="dark",
                inverse=True,
                className="h-100 tab-card-dark",
            ),
            md=6,
            className="mb-3",
        )

    def update_parameters(self, parameters: Dict[str, Dict[str, Any]]) -> ModifiedFlags:
        """
        Updates 2D data acquirer parameters (axes, averages).
        """
        flags = super().update_parameters(parameters)

        if self.component_id not in parameters:
            return flags

        params = parameters[self.component_id]

        # X-axis
        if "span" in params and self.span != params["span"]:
            self.span = params["span"]
            flags |= ModifiedFlags.PARAMETERS_MODIFIED | ModifiedFlags.PROGRAM_MODIFIED
        if "points" in params and self.points != params["points"]:
            self.points = params["points"]
            flags |= ModifiedFlags.PARAMETERS_MODIFIED | ModifiedFlags.PROGRAM_MODIFIED
        return flags

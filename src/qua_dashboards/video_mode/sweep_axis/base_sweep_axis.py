from typing import Optional, Dict, Any

from dash import Dash, html
from dash.development.base_component import Component
import numpy as np
import dash_bootstrap_components as dbc
from qua_dashboards.core import BaseUpdatableComponent, ModifiedFlags
from qua_dashboards.utils.dash_utils import create_input_field

__all__ = ["BaseSweepAxis"]

DEFAULT_SPAN = 0.03
DEFAULT_POINTS = 51


class BaseSweepAxis(BaseUpdatableComponent):
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
        span: Optional[float] = None,
        points: Optional[int] = None,
        label: Optional[str] = None,
        units: Optional[str] = None,
        offset_parameter=None,
        component_id: Optional[str] = None,
    ):
        if component_id is None:
            component_id = f"{name}-axis"
        super().__init__(component_id=component_id)
        self.name = name
        self.span = span or DEFAULT_SPAN
        self.points = points or DEFAULT_POINTS
        self.label = label
        self.units = units
        self.offset_parameter = offset_parameter
        self.dbm: bool = False
        self._coord_name = name

    @property
    def sweep_values(self):
        """Returns axis sweep values using span and points."""
        return np.linspace(-self.span / 2, self.span / 2, self.points)

    @property
    def coord_name(self) -> str:
        return self._coord_name

    @property
    def sweep_values_with_offset(self):
        """Returns axis sweep values with offset."""
        return self.sweep_values + self.offset_parameter()

    @property
    def qua_sweep_values(self):
        """Returns the actual array to be processed by the DataAcquirer"""
        return self.sweep_values

    def get_layout(self) -> Component | None:
        return self.create_axis_layout(
            min_span=0.001,
            max_span=None,
        )

    def register_callbacks(self, app: Dash) -> None:
        pass

    def declare_vars(self):
        """Declare the relevant QUA variables"""
        pass

    def apply(self, value, past_value):
        """Apply the correct QUA command"""
        pass

    def _create_parameter_row(
        self,
        *,
        label: str,
        input_id,
        value,
        units: str = "",
        min_value=None,
        max_value=None,
        step=None,
    ):
        return dbc.Row(
            [
                dbc.Col(
                    html.Div(label),
                    width="auto",
                    style={"width": "72px", "flex": "0 0 72px"},
                ),
                dbc.Col(
                    dbc.Input(
                        id=input_id,
                        type="number",
                        value=value,
                        min=min_value,
                        max=max_value,
                        step=step,
                        size="sm",
                        style={"width": "150px"},
                    ),
                    width="auto",
                    className="px-0",
                ),
                dbc.Col(
                    html.Span(units),
                    width="auto",
                    style={"width": "42px", "flex": "0 0 42px"},
                ),
            ],
            className="g-2 align-items-center mb-2 flex-nowrap",
        )

    def create_axis_layout(self, min_span: float, max_span: Optional[float] = None):
        """Build a compact, aligned axis-parameter layout."""
        if not self.name.replace("_", "").isalnum():
            raise ValueError(
                f"Axis {self.name} must only contain alphanumeric characters and underscores."
            )

        ids = {
            "span": {"type": "number-input", "index": f"{self.component_id}::span"},
            "points": {"type": "number-input", "index": f"{self.component_id}::points"},
        }

        rows = [
            self._create_parameter_row(
                label="Span",
                input_id=ids["span"],
                value=self.span,
                units=self.units or "",
                min_value=min_span,
                max_value=max_span,
                step=0.001,
            ),
            self._create_parameter_row(
                label="Points",
                input_id=ids["points"],
                value=self.points,
                min_value=1,
                max_value=501,
                step=1,
            ),
        ]

        return dbc.Col(
            html.Div(rows, className="py-2"),
            md=6,
            className="mb-2",
        )

    def update_parameters(self, parameters: Dict[str, Dict[str, Any]]) -> ModifiedFlags:
        """
        Updates 2D data acquirer parameters (axes, averages).
        """
        flags = super().update_parameters(parameters)

        if self.component_id not in parameters:
            return flags

        params = parameters[self.component_id]

        if "span" in params and self.span != params["span"]:
            self.span = params["span"]
            flags |= ModifiedFlags.PARAMETERS_MODIFIED | ModifiedFlags.PROGRAM_MODIFIED
        if "points" in params and self.points != params["points"]:
            self.points = params["points"]
            flags |= ModifiedFlags.PARAMETERS_MODIFIED | ModifiedFlags.PROGRAM_MODIFIED

        return flags

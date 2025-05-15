import logging
from typing import Any, List, Dict
import xarray as xr
import numpy as np

from dash import html
import dash_bootstrap_components as dbc

from qua_dashboards.video_mode.data_acquirers.base_data_acquirer import BaseDataAcquirer
from qua_dashboards.video_mode.sweep_axis import SweepAxis
from qua_dashboards.core import ModifiedFlags
from qua_dashboards.video_mode.utils.dash_utils import create_axis_layout


logger = logging.getLogger(__name__)

__all__ = ["Base2DDataAcquirer"]


class Base2DDataAcquirer(BaseDataAcquirer):
    """
    Abstract base class for 2D data acquirers.

    Inherits from BaseDataAcquirer and adds explicit handling for an X and Y
    sweep axis. This class is intended for acquirers that produce 2D grid data.
    The raw data returned by `perform_actual_acquisition` in subclasses
    is typically expected to be a 2D numpy array or similar structure that
    can be mapped to an xarray.DataArray by the consumer.
    """

    def __init__(
        self,
        *,
        x_axis: SweepAxis,
        y_axis: SweepAxis,
        component_id: str,
        acquisition_interval_s: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the Base2DDataAcquirer.

        Args:
            x_axis: The X sweep axis.
            y_axis: The Y sweep axis.
            component_id: Unique ID for Dash component namespacing.
            acquisition_interval_s: Target acquisition interval for background thread.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            component_id=component_id,
            acquisition_interval_s=acquisition_interval_s,
            **kwargs,
        )
        self.x_axis: SweepAxis = x_axis
        self.y_axis: SweepAxis = y_axis

    def update_parameters(self, parameters: Dict[str, Dict[str, Any]]) -> ModifiedFlags:
        """
        Updates 2D data acquirer parameters (axes, averages).
        """
        flags = super().update_parameters(parameters)

        if self.component_id in parameters:
            params = parameters[self.component_id]
            # X-axis
            if "x-span" in params and self.x_axis.span != params["x-span"]:
                self.x_axis.span = params["x-span"]
                flags |= ModifiedFlags.PARAMETERS_MODIFIED
            if "x-points" in params and self.x_axis.points != params["x-points"]:
                self.x_axis.points = params["x-points"]
                flags |= ModifiedFlags.PARAMETERS_MODIFIED
            # Y-axis
            if "y-span" in params and self.y_axis.span != params["y-span"]:
                self.y_axis.span = params["y-span"]
                flags |= ModifiedFlags.PARAMETERS_MODIFIED
            if "y-points" in params and self.y_axis.points != params["y-points"]:
                self.y_axis.points = params["y-points"]
                flags |= ModifiedFlags.PARAMETERS_MODIFIED

        return flags

    def get_dash_components(self, include_subcomponents: bool = True) -> List[Any]:
        """
        Returns Dash UI components for X and Y axis configuration.
        """
        components = super().get_dash_components(include_subcomponents)

        axis_ui = [
            html.Div(
                [
                    dbc.Row(
                        [
                            create_axis_layout(
                                axis="x",
                                component_id=self.component_id,
                                span=self.x_axis.span,
                                points=self.x_axis.points,
                                min_span=0.001,
                                max_span=None,
                                units=self.x_axis.units,
                            ),
                            create_axis_layout(
                                axis="y",
                                component_id=self.component_id,
                                span=self.y_axis.span,
                                points=self.y_axis.points,
                                min_span=0.001,
                                max_span=None,
                                units=self.y_axis.units,
                            ),
                        ],
                        className="g-0",  # No gutters
                    ),
                ]
            )
        ]
        components.extend(axis_ui)
        return components

    def get_latest_data(self) -> Dict[str, Any]:
        """
        Retrieves the latest processed data, converts it to an xarray.DataArray,
        and includes status/error information.
        """
        # Get the processed data numpy array from the parent class
        processed_output = super().get_latest_data()

        data_np = processed_output.get("data")
        error = processed_output.get("error")
        status = processed_output.get("status")

        if error is not None or data_np is None:
            # If there's an error or no data, return the original output
            return processed_output

        if not isinstance(data_np, np.ndarray) or data_np.ndim != 2:
            dim_str = data_np.ndim if hasattr(data_np, "ndim") else "N/A"
            logger.warning(
                f"{self.component_id}: Expected a 2D numpy array for xarray conversion "
                f"but got {type(data_np)} with {dim_str} dimensions. "
                f"Returning raw data."
            )
            return processed_output

        # Ensure shapes match the axes
        expected_shape = (self.y_axis.points, self.x_axis.points)
        if data_np.shape != expected_shape:
            logger.warning(
                f"{self.component_id}: Shape mismatch. Data shape is {data_np.shape}, "
                f"but expected {expected_shape} based on axis points. "
                f"Returning raw data. Ensure acquisition logic and axis points align."
            )
            return {
                "data": None,  # Or a default empty xarray
                "error": ValueError(
                    f"Shape mismatch: data {data_np.shape}, expected {expected_shape}"
                ),
                "status": "error",
            }

        try:
            # Convert the numpy array to an xarray.DataArray
            data_xr = xr.DataArray(
                data_np,
                coords=[
                    (self.y_axis.name, self.y_axis.sweep_values_with_offset),
                    (self.x_axis.name, self.x_axis.sweep_values_with_offset),
                ],
                attrs={"long_name": "Signal"},
            )

            for axis in [self.x_axis, self.y_axis]:
                attrs = {"label": axis.label or axis.name}
                if axis.units is not None:
                    attrs["units"] = axis.units
                data_xr.coords[axis.name].attrs.update(attrs)

            return {
                "data": data_xr,
                "error": None,
                "status": status,
            }

        except Exception as e:
            logger.error(
                f"Error converting numpy data to xarray.DataArray in "
                f"{self.component_id}: {e}"
            )
            return {
                "data": None,
                "error": e,
                "status": "error",
            }

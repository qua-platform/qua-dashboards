import logging
from typing import Any, List, Dict
import xarray as xr
import numpy as np

from dash import html
import dash_bootstrap_components as dbc

from qua_dashboards.video_mode.data_acquirers.base_data_acquirer import BaseDataAcquirer
from qua_dashboards.video_mode.sweep_axis import SweepAxis
from qua_dashboards.core import ModifiedFlags, BaseUpdatableComponent

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
        sweep_axes: List[SweepAxis],
        x_axis_name: str,
        y_axis_name: str,
        component_id: str,
        acquisition_interval_s: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the Base2DDataAcquirer.

        Args:
            sweep_axes: The list of available sweep axes.
            x_axis_name: Name of the selected X sweep axis.
            y_axis_name: Name of the selected Y sweep axis.
            component_id: Unique ID for Dash component namespacing.
            acquisition_interval_s: Target acquisition interval for background thread.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            component_id=component_id,
            acquisition_interval_s=acquisition_interval_s,
            **kwargs,
        )
        # Store all axes and resolve selected X/Y by name
        self.sweep_axes: List[SweepAxis] = sweep_axes

        if x_axis_name == y_axis_name:
            raise ValueError("x_axis_name and y_axis_name must be different")
        self.x_axis_name = x_axis_name
        self.y_axis_name = y_axis_name

    @property
    def x_axis(self) -> SweepAxis:
        inner_loop = getattr(self, "qua_inner_loop_action", None)
        if inner_loop is not None: 
            inner_loop.x_axis_name = self.x_axis_name
        return self.find_sweepaxis(self.x_axis_name)
    @property
    def y_axis(self) -> SweepAxis:
        inner_loop = getattr(self, "qua_inner_loop_action", None)
        if inner_loop is not None: 
            inner_loop.y_axis_name = self.y_axis_name
        return self.find_sweepaxis(self.y_axis_name)
    
    def find_sweepaxis(self, axis_name:str) -> SweepAxis:
        names = [ax.name for ax in self.sweep_axes]
        if axis_name not in names:
            raise ValueError(
                f"axis_name '{axis_name}' not found in available axes: {names}"
            )
        return next(ax for ax in self.sweep_axes if ax.name == axis_name)

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
                            self.x_axis.get_layout(),
                            self.y_axis.get_layout(),
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

    def get_components(self) -> List[BaseUpdatableComponent]:
        components = super().get_components()
        components.extend([self.x_axis, self.y_axis])
        return components

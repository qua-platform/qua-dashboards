import logging
from typing import Any, List, Dict
import xarray as xr
import numpy as np

from dash import html, dcc
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
        amp_sweep_axes: List[SweepAxis] = [], 
        freq_sweep_axes: List[SweepAxis] = [], 
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
        self.amp_sweep_axes: List[SweepAxis] = amp_sweep_axes
        self.freq_sweep_axes: List[SweepAxis] = freq_sweep_axes

        if x_axis_name == y_axis_name:
            raise ValueError("x_axis_name and y_axis_name must be different")
        self.x_axis_name = x_axis_name
        self.y_axis_name = y_axis_name
        self._is_1d = y_axis_name is None
        self._dummy_axis = SweepAxis(name="dummy", span=0.0, points=1, units="", attenuation=0, component_id=f"{self.component_id}-dummy")
        self.post_processing_functions = {
            "Raw_data": lambda da: da, 
            "x_derivative": lambda da: da.differentiate(self.x_axis_name, edge_order = 1), 
            "y_derivative": lambda da: (da if (self.y_axis_name is None or self.y_axis_name not in da.dims) else da.differentiate(self.y_axis_name, edge_order = 1)), 
        }
        self.selected_function = self.post_processing_functions["Raw_data"]

        self.x_mode = "Voltage"
        self.y_mode = "Voltage"
    
    @property
    def _display_x_sweep_axes(self): 
        if self.x_mode == "Voltage":
            return self.sweep_axes
        elif self.x_mode == "Frequency": 
            return self.freq_sweep_axes
        elif self.x_mode == "Amplitude": 
            return self.amp_sweep_axes
        
    @property
    def _display_y_sweep_axes(self): 
        if self.y_mode == "Voltage":
            return self.sweep_axes
        elif self.y_mode == "Frequency": 
            return self.freq_sweep_axes
        elif self.y_mode == "Amplitude": 
            return self.amp_sweep_axes

    @property
    def x_axis(self) -> SweepAxis:
        inner_loop = getattr(self, "qua_inner_loop_action", None)
        if inner_loop is not None: 
            inner_loop.x_mode = self.x_mode
            inner_loop.x_axis_name = self.x_axis_name
        return self.find_sweepaxis(self.x_axis_name, self.x_mode)
    @property
    def y_axis(self) -> SweepAxis:
        inner_loop = getattr(self, "qua_inner_loop_action", None)
        if inner_loop is not None: 
            if self.y_axis_name is None:
                self._is_1d = True
                inner_loop.y_axis_name = self._dummy_axis.name
                return self._dummy_axis
            self._is_1d = False
            inner_loop.y_mode = self.y_mode
            inner_loop.y_axis_name = self.y_axis_name
        return self.find_sweepaxis(self.y_axis_name, self.y_mode)
    
    def find_sweepaxis(self, axis_name:str, mode) -> SweepAxis:
        if axis_name is None: 
            return self._dummy_axis
        if mode == "Voltage":
            axes = self.sweep_axes
        if mode == "Amplitude":
            axes = self.amp_sweep_axes
        if mode == "Frequency": 
            axes = self.freq_sweep_axes

        names = [ax.name for ax in axes]
        if axis_name not in names:
            raise ValueError(
                f"axis_name '{axis_name}' not found in available axes: {names}"
            )
        return next(ax for ax in axes if ax.name == axis_name)
    
    def add_processing_function(self, name: str, fn, overwrite:bool = False): 
        """
        Register a new post processing function. 
        Expected format: fn(da: xr.DataArray) -> xr.DataArray
        """
        if not callable(fn):
            raise TypeError("fn must be callable")
        
        if (name in self.post_processing_functions) and not overwrite: 
            raise ValueError(f"Function '{name}' already exists. User overwrite = True to repalce.")
        
        def wrapped(da: xr.DataArray):
            try:
                out = fn(da)
                if not isinstance(out, xr.DataArray):
                    out = xr.DataArray(out, dims=da.dims, coords=da.coords, attrs=da.attrs)
                return out
            except Exception as e:
                logger.warning("Post-processing '%s' failed: %s. Returning input.", name, e)
                return da

        self.post_processing_functions[name] = wrapped
        return name

    def get_dash_components(self, include_subcomponents: bool = True) -> List[Any]:
        """
        Returns Dash UI components for X and Y axis configuration.
        """
        components = super().get_dash_components(include_subcomponents)

        mode_selection_ui = [
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col([
                                html.H6("Select X Mode"),
                                dcc.Dropdown(
                                    id = self._get_id("x-mode"),
                                    options = [{"label" : mode, "value" : mode} for mode in ["Voltage", "Amplitude", "Frequency"]],
                                    value = self.x_mode, 
                                    style = {"color":"black"},
                                    className="mb-2", 
                                    clearable=False
                                ),
                            ]),
                            dbc.Col([
                                html.H6("Select Y Mode"),
                                dcc.Dropdown(
                                    id = self._get_id("y-mode"),
                                    options = [{"label" : mode, "value" : mode} for mode in ["Voltage", "Amplitude", "Frequency"]],
                                    value = self.y_mode, 
                                    style = {"color":"black"},
                                    className="mb-2", 
                                    placeholder="None",
                                    clearable = False
                                ),
                            ]),
                        ]
                    )
                ]
            )
        ]

        selection_ui = [
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col([
                                html.H6("Select X Axis"),
                                dcc.Dropdown(
                                    id = self._get_id("gate-select-x"),
                                    options = [{"label" : gate_name, "value" : gate_name} for gate_name in [axis.name for axis in self._display_x_sweep_axes]],
                                    value = self.x_axis_name, 
                                    style = {"color":"black"},
                                    className="mb-2", 
                                    clearable=False
                                ),
                            ]),
                            dbc.Col([
                                html.H6("Select Y Axis"),
                                dcc.Dropdown(
                                    id = self._get_id("gate-select-y"),
                                    options = [{"label" : gate_name, "value" : gate_name} for gate_name in [axis.name for axis in self._display_y_sweep_axes]],
                                    value = self.y_axis_name, 
                                    style = {"color":"black"},
                                    className="mb-2", 
                                    placeholder="None",
                                ),
                            ]),
                        ]
                    )
                ]
            )
        ]
        components = components + mode_selection_ui + selection_ui 
        row = [self.x_axis.get_layout(), self.y_axis.get_layout()] if self.y_axis_name is not None else [self.x_axis.get_layout()]
        axis_ui = [
            html.Div(
                [
                    dbc.Row(
                        row,
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
            logger.info(
                f"{self.component_id}: Shape mismatch. Data shape is {data_np.shape}, "
                f"but expected {expected_shape} based on axis points. "
                f"Ignoring stale frames."
            )
            return {
                "data": None,  # Or a default empty xarray
                "error": None,
                "status": "pending",
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
            data_xr = self.selected_function(data_xr)

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
        return components

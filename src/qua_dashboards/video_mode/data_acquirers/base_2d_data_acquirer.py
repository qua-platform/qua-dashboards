import logging
from typing import Any, List, Dict
import xarray as xr
import numpy as np

from dash import html, dcc
import dash_bootstrap_components as dbc

from qua_dashboards.video_mode.data_acquirers.base_data_acquirer import BaseDataAcquirer
from qua_dashboards.video_mode.sweep_axis import BaseSweepAxis, VoltageSweepAxis
from qua_dashboards.core import BaseUpdatableComponent

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
        sweep_axes: Dict[str, List[BaseSweepAxis]],
        x_axis_name: str,
        y_axis_name: str,
        component_id: str,
        acquisition_interval_s: float = 0.1,
        x_mode: str = "Voltage", 
        y_mode: str = "Voltage",
        **kwargs: Any,
    ) -> None:
        """
        Initializes the Base2DDataAcquirer.

        Args:
            sweep_axes: The Dict of available sweep axis lists. Format should be {'Voltage': [VoltageSweepAxis], 'Frequency': [FrequencySweepAxis], 'Amplitude': [AmplitudeSweepAxiis]}
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
        self.sweep_axes = sweep_axes

        if x_axis_name == y_axis_name and x_mode == y_mode:
            raise ValueError("x_axis_name and y_axis_name must be different")
        self.x_axis_name = x_axis_name
        self.y_axis_name = y_axis_name

        self._dummy_axis = VoltageSweepAxis(
            name="dummy",
            span=0.0,
            points=1,
            attenuation=0,
            component_id=f"{self.component_id}-dummy",
        )
        self.post_processing_functions = {
            "Raw_data": lambda da: da,
            "x_derivative": lambda da: da.differentiate(
                self.x_axis.coord_name, edge_order=1
            ),
            "y_derivative": lambda da: (
                da
                if (
                    self.y_axis.coord_name is None
                    or self.y_axis.coord_name not in da.dims
                )
                else da.differentiate(self.y_axis.coord_name, edge_order=1)
            ),
        }
        self.selected_function = self.post_processing_functions["Raw_data"]

        self.x_mode = x_mode
        self.y_mode = y_mode

    @property
    def _is_1d(self) -> bool:
        return self.y_axis_name is None

    @property
    def _display_x_sweep_axes(self) -> List[BaseSweepAxis]:
        return self.sweep_axes[self.x_mode]

    @property
    def _display_y_sweep_axes(self) -> List[BaseSweepAxis]:
        return self.sweep_axes[self.y_mode]

    @property
    def x_axis(self) -> BaseSweepAxis:
        try:
            return self.find_sweepaxis(self.x_axis_name, self.x_mode)
        except ValueError:
            valid_axes = self._display_x_sweep_axes
            self.x_axis_name = valid_axes[0].name
            return valid_axes[0]

    @property
    def y_axis(self) -> BaseSweepAxis:
        if self.y_axis_name is None:
            return self._dummy_axis
        try:
            return self.find_sweepaxis(self.y_axis_name, self.y_mode)
        except ValueError:
            self.y_axis_name = None
            return self._dummy_axis

    def find_sweepaxis(self, axis_name: str, mode) -> BaseSweepAxis:
        if axis_name is None:
            return self._dummy_axis
        axes = self.sweep_axes[mode]
        names = [ax.name for ax in axes]
        if axis_name not in names:
            raise ValueError(
                f"axis_name '{axis_name}' not found in available axes: {names}"
            )
        return next(ax for ax in axes if ax.name == axis_name)

    def add_processing_function(self, name: str, fn, overwrite: bool = False):
        """
        Register a new post processing function.
        Expected format: fn(da: xr.DataArray) -> xr.DataArray
        """
        if not callable(fn):
            raise TypeError("fn must be callable")

        if (name in self.post_processing_functions) and not overwrite:
            raise ValueError(
                f"Function '{name}' already exists. User overwrite = True to repalce."
            )

        def wrapped(da: xr.DataArray):
            try:
                out = fn(da)
                if not isinstance(out, xr.DataArray):
                    out = xr.DataArray(
                        out, dims=da.dims, coords=da.coords, attrs=da.attrs
                    )
                return out
            except Exception as e:
                logger.warning(
                    "Post-processing '%s' failed: %s. Returning input.", name, e
                )
                return da

        self.post_processing_functions[name] = wrapped
        return name
    
    def _build_dropdown_options(self, _display_sweep_axis):
        options = [
            {"label": gate_name, "value": gate_name}
            for gate_name in [
                axis.name
                for axis in _display_sweep_axis
            ]
        ] 
        return options

    def _axis_number_input(self, axis, parameter, value, units="", **kwargs):
        input_id = {"type": "number-input", "index": f"{axis.component_id}::{parameter}"}
        input_component = dbc.Input(
            id=input_id,
            type="number",
            value=value,
            size="sm",
            **kwargs,
        )

        if units:
            return dbc.InputGroup(
                [
                    input_component,
                    dbc.InputGroupText(units),
                ],
                size="sm",
            )

        return input_component

    def _axis_parameter_rows(self):
        x_axis = self.x_axis
        y_axis = self.y_axis if self.y_axis_name is not None else None

        rows = [
            (
                "Span",
                self._axis_number_input(
                    x_axis,
                    "span",
                    x_axis.span,
                    x_axis.units or "",
                    min=0.001,
                    step=0.001,
                ),
                self._axis_number_input(
                    y_axis,
                    "span",
                    y_axis.span,
                    y_axis.units or "",
                    min=0.001,
                    step=0.001,
                ) if y_axis is not None else None,
            ),
            (
                "Points",
                self._axis_number_input(
                    x_axis,
                    "points",
                    x_axis.points,
                    min=1,
                    max=501,
                    step=1,
                ),
                self._axis_number_input(
                    y_axis,
                    "points",
                    y_axis.points,
                    min=1,
                    max=501,
                    step=1,
                ) if y_axis is not None else None,
            ),
        ]

        x_has_offset = hasattr(x_axis, "offset")
        y_has_offset = y_axis is not None and hasattr(y_axis, "offset")

        if x_has_offset or y_has_offset:
            rows.append(
                (
                    "Offset",
                    self._axis_number_input(
                        x_axis,
                        "offset",
                        x_axis.offset,
                        x_axis.units or "",
                    ) if x_has_offset else None,
                    self._axis_number_input(
                        y_axis,
                        "offset",
                        y_axis.offset,
                        y_axis.units or "",
                    ) if y_has_offset else None,
                )
            )

        x_has_dbm = hasattr(x_axis, "_offset_volts")
        y_has_dbm = y_axis is not None and hasattr(y_axis, "_offset_volts")

        if x_has_dbm or y_has_dbm:
            def dbm_toggle(axis):
                if axis is None or not hasattr(axis, "_offset_volts"):
                    return None
                return dbc.Row(
                    [
                        dbc.Col("V", width="auto"),
                        dbc.Col(
                            dbc.Checklist(
                                id={
                                    "type": "toggle",
                                    "index": f"{axis.component_id}::dbm-toggle",
                                },
                                options=[{"label": "", "value": "on"}],
                                value=["on"] if axis.dbm else [],
                                switch=True,
                                className="mb-0",
                            ),
                            width="auto",
                        ),
                        dbc.Col("dBm", width="auto"),
                    ],
                    className="g-1 align-items-center",
                )

            rows.append(("Units", dbm_toggle(x_axis), dbm_toggle(y_axis)))

        return [
            dbc.Row(
                [
                    dbc.Col(label, width=2, className="d-flex align-items-center"),
                    dbc.Col(x_control if x_control is not None else html.Div(), width=5),
                    dbc.Col(y_control if y_control is not None else html.Div(), width=5),
                ],
                className="g-2 mb-2 align-items-center",
            )
            for label, x_control, y_control in rows
        ]

    def get_dash_components(self, include_subcomponents: bool = True) -> List[Any]:
        """Returns a compact X/Y parameter table for the selected sweep axes."""
        components = super().get_dash_components(include_subcomponents)
        keys = self.sweep_axes.keys()
        y_mode_keys = getattr(self, "_y_mode_keys_override", None) or keys
        x_mode_keys = getattr(self, "_x_mode_keys_override", None) or keys

        components.append(
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col("", width=2),
                            dbc.Col(html.Strong("X"), width=5),
                            dbc.Col(html.Strong("Y"), width=5),
                        ],
                        className="g-2 mb-1",
                    ),
                    dbc.Row(
                        [
                            dbc.Col("Mode", width=2, className="d-flex align-items-center"),
                            dbc.Col(
                                dcc.Dropdown(
                                    id=self._get_id("x-mode"),
                                    options=[{"label": mode, "value": mode} for mode in x_mode_keys],
                                    value=self.x_mode,
                                    style={"color": "black"},
                                    clearable=False,
                                ),
                                width=5,
                            ),
                            dbc.Col(
                                dcc.Dropdown(
                                    id=self._get_id("y-mode"),
                                    options=[{"label": mode, "value": mode} for mode in y_mode_keys],
                                    value=self.y_mode,
                                    style={"color": "black"},
                                    placeholder="None",
                                    clearable=False,
                                ),
                                width=5,
                            ),
                        ],
                        className="g-2 mb-2 align-items-center",
                    ),
                    dbc.Row(
                        [
                            dbc.Col("Axis", width=2, className="d-flex align-items-center"),
                            dbc.Col(
                                dcc.Dropdown(
                                    id=self._get_id("gate-select-x"),
                                    options=self._build_dropdown_options(self._display_x_sweep_axes),
                                    value=self.x_axis_name,
                                    style={"color": "black"},
                                    clearable=False,
                                ),
                                width=5,
                            ),
                            dbc.Col(
                                dcc.Dropdown(
                                    id=self._get_id("gate-select-y"),
                                    options=self._build_dropdown_options(self._display_y_sweep_axes),
                                    value=self.y_axis_name,
                                    style={"color": "black"},
                                    placeholder="None",
                                ),
                                width=5,
                            ),
                        ],
                        className="g-2 mb-2 align-items-center",
                    ),
                    *self._axis_parameter_rows(),
                ]
            )
        )

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
                    (self.y_axis.coord_name, self.y_axis.sweep_values_with_offset),
                    (self.x_axis.coord_name, self.x_axis.sweep_values_with_offset),
                ],
                attrs={"long_name": "Signal",
                       "units": self.result_unit},
            )
            data_xr = self.selected_function(data_xr)

            for axis in [self.x_axis, self.y_axis]:
                attrs = {"label": axis.label or axis.name}
                if axis.units is not None:
                    attrs["units"] = axis.units
                data_xr.coords[axis.coord_name].attrs.update(attrs)

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

# qua_dashboards/video_mode/data_acquirers/random_data_acquirer.py
import logging
from time import sleep
from typing import Any, Dict, List, Optional, Tuple, Literal
import xarray as xr

import numpy as np
from dash import html

from qua_dashboards.core import ModifiedFlags, BaseUpdatableComponent
from qua_dashboards.utils.dash_utils import create_input_field
from qua_dashboards.video_mode.data_acquirers.base_2d_data_acquirer import (
    Base2DDataAcquirer,
    BaseDataAcquirer
)
from quam.components.pulses import ReadoutPulse
from qua_dashboards.video_mode.scan_modes import ScanMode, LineScan
from qua_dashboards.video_mode.inner_loop_actions import InnerLoopAction
from qua_dashboards.video_mode.sweep_axis import BaseSweepAxis, VoltageSweepAxis, AmplitudeSweepAxis, FrequencySweepAxis
from quam_builder.architecture.quantum_dots.components import VoltageGate, GateSet


logger = logging.getLogger(__name__)

__all__ = ["GateSetDataAcquirer"]


class BaseGateSetDataAcquirer(Base2DDataAcquirer):
    """Base Data Acquirer that is built upon GateSet. 

    Inherits from Base2DDataAcquirer and builds around the GateSet and VirtualGateSet to built the UI.
    The sweep is not configured here, since this can stay agnostic of actual OPX usage. 
    """

    result_types_default = ["I", "Q", "amplitude", "phase"]

    def __init__(
        self,
        *,
        x_axis_name: str,
        y_axis_name: str,
        scan_modes: Dict[str, ScanMode],
        component_id: str = "gate-set-data-acquirer",
        num_software_averages: int = 1, 
        acquisition_interval_s: float = 0.1,
        gate_set: GateSet = None,
        result_type: Literal["I", "Q", "amplitude", "phase"] = "I",
        # Other parameters like num_software_averages, acquisition_interval_s
        # are passed via **kwargs to Base2DDataAcquirer and then to BaseDataAcquirer.
        available_readout_pulses: List[ReadoutPulse] = [],
        **kwargs: Any,

    ) -> None:
        """Initializes the GateSetDataAcquirer.

        Args:
            component_id: Unique ID for Dash elements.
            sweep_axes: The list of available sweep axes.
            x_axis_name: Name of the X sweep axis.
            y_axis_name: Name of the Y sweep axis.
            gate_set: The GateSet object containing voltage channels.
            scan_modes: A dictionary of scan modes defining how the 2D grid is traversed.
            x_axis_name: Name of the X sweep axis (must match a GateSet channel or virtual gate).
            y_axis_name: Name of the Y sweep axis (must match a GateSet channel or virtual gate).
            available_readout_pulses: A list of the QUAM Pulse objects to measure.
            num_software_averages: Number of raw snapshots for software averaging.
            acquisition_interval_s: Target interval for acquiring a new raw snapshot.
            result_type: The type of result to derive from I/Q data.
            **kwargs: Additional arguments for Base2DDataAcquirer, including
                num_software_averages and acquisition_interval_s for
                BaseDataAcquirer.
        """
        logger.debug(
            f"Initializing GateSetDataAcquirer (ID: {component_id})."
        )
        sweep_axes: List[BaseSweepAxis] = self._generate_sweep_axes(gate_set=gate_set, available_pulses=available_readout_pulses)

        super().__init__(
            component_id=component_id,
            sweep_axes=sweep_axes,
            x_axis_name=x_axis_name,
            y_axis_name=y_axis_name,
            num_software_averages=num_software_averages,
            acquisition_interval_s=acquisition_interval_s,
            **kwargs,
        )
        self.inner_loop_action: Optional[InnerLoopAction] = None
        self._compilation_flags: ModifiedFlags = ModifiedFlags.NONE
        self.scan_modes = scan_modes
        self.scan_2d: ScanMode = next(iter(self.scan_modes.values()))
        self.scan_1d: ScanMode = LineScan()
        self.result_types = self.result_types_default
        self.result_type: str = result_type
        self.inner_functions_dict = {}
        self.gate_set = gate_set
        self.available_readout_pulses = available_readout_pulses
        self._configure_readout()
        self._scan_idx_cache = None


    @property
    def current_scan_mode(self) -> str:
        for name, mode in self.scan_modes.items():
            if self.scan_2d == mode:
                return name
        return next(iter(self.scan_modes.keys()))

    def set_scan_mode(self, name: str) -> None:
        if self.scan_2d is self.scan_modes[name]:
            return
        self.scan_2d = self.scan_modes[name]
        self._scan_idx_cache = None
        self._halt_acquisition()
        self._compilation_flags |= ModifiedFlags.PROGRAM_MODIFIED

    def _halt_acquisition(self): 
        pass

    @property
    def scan_mode(self) -> ScanMode:
        if self._is_1d:
            return self.scan_1d
        else:
            return self.scan_2d
        
    @property
    def x_axis(self) -> BaseSweepAxis:
        inner_loop = getattr(self, "inner_loop_action", None)
        if inner_loop is not None:
            inner_loop.x_mode = self.x_mode
        try:
            axis = self.find_sweepaxis(self.x_axis_name, self.x_mode)
            if inner_loop is not None:
                inner_loop.x_axis = axis
            return axis
        except ValueError:
            valid_axes = self._display_x_sweep_axes
            self.x_axis_name = valid_axes[0].name
            return valid_axes[0]

    @property
    def y_axis(self) -> BaseSweepAxis:
        inner_loop = getattr(self, "inner_loop_action", None)
        if self.y_axis_name is None:
            return self._dummy_axis
        if inner_loop is not None:
            inner_loop.y_mode = self.y_mode
            inner_loop.y_axis_name = self.y_axis_name
        try:
            axis = self.find_sweepaxis(self.y_axis_name, self.y_mode)
            if inner_loop is not None:
                inner_loop.y_axis = axis
            return axis
        except ValueError:
            self.y_axis_name = None
            return self._dummy_axis
    
    def get_scan_indices(self, x_pts:int, y_pts:int) -> Tuple[np.ndarray, np.ndarray]: 
        """
        Cache the scan mode indices, so that during run-time, it does not keep querying the scan mode classes. 
        """
        key = (x_pts, y_pts, id(self.scan_mode))
        if self._scan_idx_cache is None or self._scan_idx_cache[0] != key: 
            x_idx, y_idx = self.scan_mode.get_idxs(x_points=x_pts, y_points=y_pts)
            self._scan_idx_cache = (
                key,
                np.asarray(x_idx, dtype=np.intp),
                np.asarray(y_idx, dtype=np.intp),
            )
        return self._scan_idx_cache[1], self._scan_idx_cache[2]
    
    def ensure_axis(self) -> None:
        gs = self.gate_set
        have = {axis.name for ax in self.sweep_axes.values() for axis in ax}
        for nm in gs.valid_channel_names:
            if nm not in have:
                offset_parameter = None
                if self.voltage_control_component is not None: 
                    params = self.voltage_control_component.voltage_parameters_by_name
                    if nm in params: 
                        offset_parameter = params[nm]
                self.sweep_axes["Voltage"].append(VoltageSweepAxis(name=nm, offset_parameter=offset_parameter))
        
    def _build_dropdown_options(self, _display_sweep_axis):
        """Build dropdown options with physical/virtual grouping."""
        options = []
        available_names = [axis.name for axis in _display_sweep_axis]
        physical_names = set(self.gate_set.channels.keys())
        virtual_names = [n for n in available_names if n not in physical_names]
        
        # Physical gates section
        if physical_names:
            options.append({"label": "── Physical Gates ──", "value": "__physical_header__", "disabled": True})
            for name in sorted(physical_names):
                options.append({"label": name, "value": name})
        
        # Virtual gates section  
        if virtual_names:
            options.append({"label": "── Virtual Gates ──", "value": "__virtual_header__", "disabled": True})
            for name in virtual_names:
                options.append({"label": name, "value": name})
        return options
    
    def _configure_readout(self) -> None:
        """
        Searches the channels and finds the appropriate readout channels.
        """
        self.available_readout_channels = {}
        self.readout_pulse_mapping = {}
        for p in self.available_readout_pulses:
            self.available_readout_channels[p.channel.name] = p.channel
            self.readout_pulse_mapping[p.channel.name] = p
        
        if self.inner_loop_action is not None: 
            self.inner_loop_action.readout_pulse_mapping = self.readout_pulse_mapping

        self.selected_readout_channels = (
            [
                self.available_readout_channels[
                    self.available_readout_pulses[0].channel.name
                ]
            ]
            if self.available_readout_channels
            else []
        )
        if self.inner_loop_action is not None: 
            self.inner_loop_action.selected_readout_channels = (
                self.selected_readout_channels
            )
    
    def _generate_sweep_axes(
        self, gate_set, available_pulses
    ) -> Dict[str, List[BaseSweepAxis]]:
        voltage_axes: List[VoltageSweepAxis] = []
        for channel_name in gate_set.valid_channel_names:
            if channel_name in gate_set.channels:
                channel = gate_set.channels[channel_name]
                if isinstance(channel, VoltageGate):
                    attenuation = channel.attenuation
                    offset_parameter = channel.offset_parameter
                else:
                    # Regular SingleChannel -> no attenuation or offset
                    attenuation = 0
                    offset_parameter = None
            else:
                # Virtual gate -> no channel -> no attenuation or offset
                attenuation = 0
                if hasattr(self, "voltage_control_component") and self.voltage_control_component is not None: 
                    params_by_name = self.voltage_control_component.voltage_parameters_by_name
                    if channel_name in params_by_name:
                        offset_parameter = params_by_name[channel_name]
                else:
                    offset_parameter = None
            voltage_axes.append(
                VoltageSweepAxis(
                    name=channel_name,
                    offset_parameter=offset_parameter,
                    attenuation=attenuation,
                    component_id=f"{channel_name}_volt",
                )
            )
        drive_axes: List[AmplitudeSweepAxis] = []
        freq_axes: List[FrequencySweepAxis] = []
        for pulse in available_pulses:
            channel_name = pulse.channel.name
            freq_axes.append(
                FrequencySweepAxis(
                    name=channel_name,
                    offset_parameter=pulse,
                    component_id=f"{channel_name}_freq",
                )
            )
            drive_axes.append(
                AmplitudeSweepAxis(
                    name=channel_name,
                    offset_parameter=pulse,
                    component_id=f"{channel_name}_amp",
                )
            )
        return {
            "Voltage": voltage_axes,
            "Frequency": freq_axes,
            "Amplitude": drive_axes,
        }

    def update_parameters(self, parameters: Dict[str, Dict[str, Any]]) -> ModifiedFlags:
        flags = super().update_parameters(parameters)
        try:
            for axes in self.sweep_axes.values():
                for ax in axes:
                    child_flags = ax.update_parameters(parameters)
                    flags |= child_flags
        except Exception as e:
            logger.warning("Axis dispatch error: %s", e)
        if self.inner_loop_action is not None: 
            try:
                il_flags = self.inner_loop_action.update_parameters(parameters)
                flags |= il_flags
            except Exception as e:
                logger.warning("Inner-loop dispatch error: %s", e)
                raise

        if self.component_id in parameters:
            params = parameters[self.component_id]
            if "result-type" in params and self.result_type != params["result-type"]:
                self.result_type = params["result-type"]
                flags |= ModifiedFlags.PARAMETERS_MODIFIED
            if self.inner_loop_action is not None: 
                if (
                    "ramp_duration" in params
                    and self.inner_loop_action.ramp_duration != params["ramp_duration"]
                ):
                    self.inner_loop_action.ramp_duration = params["ramp_duration"]
                    flags |= (
                        ModifiedFlags.PARAMETERS_MODIFIED | ModifiedFlags.PROGRAM_MODIFIED
                    )
                if (
                    "point_duration" in params
                    and self.inner_loop_action.point_duration != params["point_duration"]
                ):
                    self.inner_loop_action.point_duration = params["point_duration"]
                    flags |= (
                        ModifiedFlags.PARAMETERS_MODIFIED | ModifiedFlags.PROGRAM_MODIFIED
                    )
                if (
                    "pre_measurement_delay" in params
                    and self.inner_loop_action.pre_measurement_delay != params["pre_measurement_delay"]
                ):
                    self.inner_loop_action.pre_measurement_delay = params["pre_measurement_delay"]
                    flags |= (
                        ModifiedFlags.PARAMETERS_MODIFIED | ModifiedFlags.PROGRAM_MODIFIED
                    )
            if (
                "gate-select-x" in params
                and params["gate-select-x"] != self.x_axis_name
            ):
                self.x_axis_name = params["gate-select-x"]
                flags |= (
                    ModifiedFlags.PARAMETERS_MODIFIED | ModifiedFlags.PROGRAM_MODIFIED
                )
            if "gate-select-y" in params:
                new_y = params["gate-select-y"]
                new_y = None if new_y in (None, "", "null", "dummy") else new_y
                if new_y != self.y_axis_name:
                    self.y_axis_name = new_y
                    flags |= (
                        ModifiedFlags.PARAMETERS_MODIFIED
                        | ModifiedFlags.PROGRAM_MODIFIED
                    )

            if "readouts" in params:
                new_names = [
                    n
                    for n in params["readouts"]
                    if n in self.available_readout_channels
                ]
                new_objs = [self.available_readout_channels[n] for n in new_names]
                if [ch.name for ch in self.selected_readout_channels] != new_names:
                    self.selected_readout_channels = new_objs
                    if self.inner_loop_action is not None: 
                        self.inner_loop_action.selected_readout_channels = (
                            self.selected_readout_channels
                        )

                    flags |= ModifiedFlags.PROGRAM_MODIFIED
                    flags |= ModifiedFlags.PARAMETERS_MODIFIED

            if "x-mode" in params and params["x-mode"] != self.x_mode:
                self.x_mode = params["x-mode"]
                flags |= (
                    ModifiedFlags.PROGRAM_MODIFIED | ModifiedFlags.PARAMETERS_MODIFIED
                )

            if "y-mode" in params and params["y-mode"] != self.y_mode:
                self.y_mode = params["y-mode"]
                flags |= (
                    ModifiedFlags.PROGRAM_MODIFIED | ModifiedFlags.PARAMETERS_MODIFIED
                )

        self._compilation_flags |= flags & (
            ModifiedFlags.PROGRAM_MODIFIED | ModifiedFlags.CONFIG_MODIFIED
        )
        if flags & (ModifiedFlags.PROGRAM_MODIFIED | ModifiedFlags.CONFIG_MODIFIED):
            with self._data_lock:
                self._data_history_raw.clear()
        return flags

    def get_dash_components(
        self,
        include_subcomponents: bool = True,
        *,
        include_inner_loop_controls: bool = False,
    ) -> List[Any]:
        components = super().get_dash_components(include_subcomponents)

        if include_subcomponents:
            if hasattr(self.scan_mode, "get_dash_components"):
                components.extend(
                    self.scan_mode.get_dash_components(include_subcomponents)
                )
            if include_inner_loop_controls and hasattr(
                self.inner_loop_action, "get_dash_components"
            ):
                components.extend(
                    self.inner_loop_action.get_dash_components(
                        include_subcomponents
                    )
                )
        return components

    def get_latest_data(self) -> Dict[str, Any]:
        """
        Retrieves the latest processed data, converts it to an xarray.DataArray,
        and includes status/error information.
        """
        if len(self.selected_readout_channels) <= 1:
            return super().get_latest_data()

        processed = BaseDataAcquirer.get_latest_data(self)
        data_np = processed.get("data")
        error = processed.get("error")
        status = processed.get("status")

        if error is not None or data_np is None:
            return processed
        if not isinstance(data_np, np.ndarray) or data_np.ndim not in (2, 3):
            dim_str = data_np.ndim if hasattr(data_np, "ndim") else "N/A"
            logger.warning(
                f"{self.component_id}: Expected a 2D or 3D numpy array for xarray conversion "
                f"but got {type(data_np)} with {dim_str} dimensions. Returning raw data."
            )
            return processed
        try:
            labels = [ch.name for ch in self.selected_readout_channels]
            if data_np.shape[0] != len(labels):
                labels = ["" for i in range(data_np.shape[0])]
            if (
                isinstance(data_np, np.ndarray)
                and data_np.ndim == 3
                and data_np.shape[1] == 1
            ):
                data_np = data_np[:, 0, :]

            actual_is_1d = data_np.ndim == 2

            if actual_is_1d:
                x_len = data_np.shape[-1]
                x_coords = list(self.x_axis.sweep_values_with_offset)
                if len(x_coords) != x_len:
                    if len(x_coords) >= 2:
                        x_coords = np.linspace(x_coords[0], x_coords[-1], x_len)
                    else:
                        x_coords = np.arange(x_len)

                data_xr = xr.DataArray(
                    data_np,
                    dims=("readout", self.x_axis.coord_name),
                    coords={"readout": labels, self.x_axis.coord_name: x_coords},
                    attrs={"long_name": "Signal"},
                )
                data_xr = self.selected_function(data_xr)
                x_attrs = {"label": self.x_axis.label or self.x_axis.coord_name}
                if self.x_axis.units is not None:
                    x_attrs["units"] = self.x_axis.units
                data_xr.coords[self.x_axis.coord_name].attrs.update(x_attrs)
                return {"data": data_xr, "error": None, "status": status}

            x_len = data_np.shape[-1]
            y_len = data_np.shape[-2]
            x_coords = list(self.x_axis.sweep_values_with_offset)
            y_coords = list(self.y_axis.sweep_values_with_offset)

            if len(x_coords) != x_len:
                if len(x_coords) >= 2:
                    x_coords = np.linspace(x_coords[0], x_coords[-1], x_len)
                else:
                    x_coords = np.arange(x_len)

            if len(y_coords) != y_len:
                if len(y_coords) >= 2:
                    y_coords = np.linspace(y_coords[0], y_coords[-1], y_len)
                else:
                    y_coords = np.arange(y_len)

            data_xr = xr.DataArray(
                data_np,
                dims=("readout", self.y_axis.coord_name, self.x_axis.coord_name),
                coords={
                    "readout": labels,
                    self.y_axis.coord_name: y_coords,
                    self.x_axis.coord_name: x_coords,
                },
                attrs={"long_name": "Signal"},
            )
            data_xr = self.selected_function(data_xr)
            for axis in [self.x_axis, self.y_axis]:
                attrs = {"label": axis.label or axis.coord_name}
                if axis.units is not None:
                    attrs["units"] = (
                        self.x_axis.units if axis is self.x_axis else self.y_axis.units
                    )
                data_xr.coords[axis.coord_name].attrs.update(attrs)

            return {"data": data_xr, "error": None, "status": status}
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
        if self.inner_loop_action is not None: 
            components.extend(self.inner_loop_action.get_components())
        return components

    def mark_virtual_layer_changed(self, *, affects_config: bool = False):
        """Call this when a virtual-gate matrix was edited."""
        if affects_config:
            self._compilation_flags |= ModifiedFlags.CONFIG_MODIFIED
        else:
            self._compilation_flags |= ModifiedFlags.PROGRAM_MODIFIED


    def _flat_to_2d(self, flat: np.ndarray) -> np.ndarray:
        flat = np.asarray(flat).ravel()
        y_pts, x_pts = int(self.y_axis.points), int(self.x_axis.points)
        output_data_2d = np.full((y_pts, x_pts), np.nan, dtype=flat.dtype)

        x_indices, y_indices = self.get_scan_indices(
            x_pts=x_pts, y_pts=y_pts,
        )
        n = min(flat.size, len(x_indices))
        if n:
            output_data_2d[y_indices[:n], x_indices[:n]] = flat[:n]
        return output_data_2d

    
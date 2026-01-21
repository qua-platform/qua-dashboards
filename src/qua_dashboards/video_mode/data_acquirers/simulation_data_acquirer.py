# qua_dashboards/video_mode/data_acquirers/random_data_acquirer.py
import logging
from time import sleep
from typing import Any, Dict, List, Optional, Tuple
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
from qua_dashboards.voltage_control.voltage_control_component import VoltageControlComponent
from qua_dashboards.video_mode.inner_loop_actions import SimulatedInnerLoopAction
from qua_dashboards.video_mode.sweep_axis import BaseSweepAxis, VoltageSweepAxis, AmplitudeSweepAxis, FrequencySweepAxis
from quam_builder.architecture.quantum_dots.components import VoltageGate, GateSet


logger = logging.getLogger(__name__)

__all__ = ["SimulationDataAcquirer"]


class SimulationDataAcquirer(Base2DDataAcquirer):
    """Data acquirer that generates random 2D data for simulation purposes.

    Inherits from Base2DDataAcquirer and simulates a delay for data acquisition.
    The actual data generation is a simple random number matrix.
    """

    def __init__(
        self,
        machine,
        *,
        x_axis_name: str,
        y_axis_name: str,
        scan_modes: Dict[str, ScanMode],
        component_id: str = "random-data-acquirer",
        acquire_time: float = 0.05,  # Simulate 50ms acquisition time per frame
        num_software_averages: int = 1, 
        acquisition_interval_s: float = 0.1,
        gate_set: GateSet = None,
        # Other parameters like num_software_averages, acquisition_interval_s
        # are passed via **kwargs to Base2DDataAcquirer and then to BaseDataAcquirer.
        available_readout_pulses: List[ReadoutPulse] = [],
        voltage_control_component: Optional[VoltageControlComponent] = None,
        **kwargs: Any,

    ) -> None:
        """Initializes the SimulationDataAcquirer.

        Args:
            component_id: Unique ID for Dash elements.
            sweep_axes: The list of available sweep axes.
            x_axis_name: Name of the X sweep axis.
            y_axis_name: Name of the Y sweep axis.
            acquire_time: Simulated time in seconds to 'acquire' one raw data frame.
            **kwargs: Additional arguments for Base2DDataAcquirer, including
                num_software_averages and acquisition_interval_s for
                BaseDataAcquirer.
        """
        self.acquire_time: float = acquire_time
        self._first_acquisition: bool = True
        logger.debug(
            f"Initializing SimulationDataAcquirer (ID: {component_id}) with "
            f"acquire_time: {self.acquire_time}s"
        )
        self.voltage_control_component = voltage_control_component
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
        self.inner_loop_action = SimulatedInnerLoopAction(
            gate_set = gate_set, 
            x_axis = self.x_axis, 
            y_axis = self.y_axis, 
        )
        self._compilation_flags: ModifiedFlags = ModifiedFlags.NONE
        self.scan_modes = scan_modes
        self.scan_2d: ScanMode = next(iter(self.scan_modes.values()))
        self.scan_1d: ScanMode = LineScan()
        self.result_types = ["I", "Q", "amplitude", "phase"]
        self.result_type = self.result_types[0]
        self.inner_functions_dict = {}
        self.gate_set = gate_set
        self.available_readout_pulses = available_readout_pulses
        self._configure_readout()
        self.machine = machine
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
        self._halt_acquisition()
        self._compilation_flags |= ModifiedFlags.PROGRAM_MODIFIED

    @property
    def scan_mode(self) -> ScanMode:
        if self._is_1d:
            return self.scan_1d
        else:
            return self.scan_2d
        
    
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
                if self.voltage_control_component is not None: 
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

    def perform_actual_acquisition(self) -> np.ndarray:
        """Simulates data acquisition by sleeping and returning random data.

        This method is called by the background thread in BaseDataAcquirer.

        Returns:
            A 2D numpy array of random float values between 0 and 1, with
            dimensions (y_axis.points, x_axis.points).
        """
        if self._first_acquisition:
            self._first_acquisition = False
        else:
            sleep(self.acquire_time)
        logger.debug(
            f"RandomDataAcquirer (ID: {self.component_id}): "
            f"Generating random data for {self.y_axis.points}x{self.x_axis.points}"
        )
        # Ensure y_axis.points and x_axis.points are positive integers
        if self.y_axis.points <= 0 or self.x_axis.points <= 0:
            logger.warning(
                f"RandomDataAcquirer (ID: {self.component_id}): Invalid points "
                f"({self.y_axis.points}x{self.x_axis.points}). Returning empty array."
            )
            return np.array([[]])  # Return a 2D empty array to avoid downstream errors
        
        num_readouts = len(self.selected_readout_channels)

        x_axis_vals = self.x_axis.sweep_values_with_offset
        y_axis_vals = self.y_axis.sweep_values_with_offset

        results = tuple(self.inner_loop_action(x_axis_vals, y_axis_vals))

        return self._process_fetched_results(results)

    def update_parameters(self, parameters: Dict[str, Dict[str, Any]]) -> ModifiedFlags:
        flags = super().update_parameters(parameters)
        try:
            for axes in self.sweep_axes.values():
                for ax in axes:
                    child_flags = ax.update_parameters(parameters)
                    flags |= child_flags
        except Exception as e:
            logger.warning("Axis dispatch error: %s", e)

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
        components.extend(self.inner_loop_action.get_components())
        return components

    def _halt_acquisition(self) -> None:
        pass

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

    def _process_fetched_results(self, fetched_results: Tuple) -> np.ndarray:
        """
        Processes the raw tuple from QUA's fetch_all into a dictionary of named arrays,
        then derives the final 2D array based on self.result_type.
        """
        if fetched_results is None:
            return np.full((self.y_axis.points, self.x_axis.points), np.nan)
        num_sel = len(self.selected_readout_channels)
        is_multi_readout = num_sel > 1
        expected_points = self.x_axis.points * self.y_axis.points

        def _normalize_flat(flat: np.ndarray) -> np.ndarray:
            """
            Trim or pad a 1D stream of samples to exactly one 2D frame and reshape.
            - If more than one frame is concatenated, then keep only the most recent full frame
            - If fewer samples are available than a full frame, returns a 2D array filled with NaNs
            - Otherwise, reshape samples into the appropriate dimensions
            """
            flat = np.asarray(flat).ravel()
            # keep last full frame if concatenated
            if flat.size > expected_points:
                flat = np.asarray(flat)[-expected_points:]
            # if not enough, return placeholder (so viewer waits gracefully)
            if flat.size < expected_points:
                return np.full(
                    (self.y_axis.points, self.x_axis.points),
                    np.nan,
                    dtype=np.asarray(flat).dtype,
                )
            return self._flat_to_2d(flat)

        if not is_multi_readout:
            I, Q = fetched_results[0], fetched_results[1]
            if self.result_type == "I":
                flat = I
            elif self.result_type == "Q":
                flat = Q
            elif self.result_type == "amplitude":
                flat = np.abs(
                    I + 1j * Q
                )
            elif self.result_type == "phase":
                flat = np.angle(
                    I + 1j * Q
                )
            else:
                raise ValueError(
                    f"Invalid result_type: '{self.result_type}'.")
            return _normalize_flat(flat)

        else:
            channel_names = [ch.name for ch in self.selected_readout_channels]
            # Multi-readout: return (R, X) in 1D, (R, Y, X) in 2D
            is_1d = self._is_1d
            expected_points = self.x_axis.points * (1 if is_1d else self.y_axis.points)

            output_layers = []
            #names = [ch.name for ch in self.selected_readout_channels]
            for i in range(num_sel):
                I = fetched_results[2 * i]
                Q = fetched_results[2 * i + 1]
                if self.result_type == "I":
                    flat = np.asarray(I)
                elif self.result_type == "Q":
                    flat = np.asarray(Q)
                elif self.result_type == "amplitude":
                    flat = np.abs(
                        np.asarray(I)
                        + 1j * np.asarray(Q)
                    )
                elif self.result_type == "phase":
                    flat = np.angle(
                        np.asarray(I)
                        + 1j * np.asarray(Q)
                    )
                else:
                    raise ValueError(f"Invalid result_type '{self.result_type}'.")

                # trim/pad per-layer, similar to _normalize_flat
                if flat.size > expected_points:
                    flat = flat[-expected_points:]
                if flat.size < expected_points:
                    if is_1d:
                        return np.full(
                            (len(channel_names), self.x_axis.points), np.nan, dtype=flat.dtype
                        )
                    else:
                        return np.full(
                            (len(channel_names), self.y_axis.points, self.x_axis.points),
                            np.nan,
                            dtype=flat.dtype,
                        )

                # shape per-layer
                if is_1d:
                    layer = flat.reshape(self.x_axis.points)
                else:
                    layer = self._flat_to_2d(flat)
                output_layers.append(layer)

            return np.stack(output_layers, axis=0)
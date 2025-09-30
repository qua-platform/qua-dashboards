import logging
import time
from typing import Any, Dict, List, Literal, Optional, Tuple
from qualang_tools.units.units import unit
import numpy as np
from qm import QuantumMachinesManager, Program
from qm.jobs.running_qm_job import RunningQmJob
from quam.components import Channel
from qm.qua import (
    program,
    declare_stream,
    infinite_loop_,
    save,
    stream_processing,
    wait,
)
from quam.components.pulses import ReadoutPulse
import xarray as xr
from qua_dashboards.video_mode.scan_modes import LineScan
from qua_dashboards.core.base_updatable_component import BaseUpdatableComponent
from qua_dashboards.video_mode.data_acquirers.base_2d_data_acquirer import (
    Base2DDataAcquirer,
    BaseDataAcquirer,
)
from qua_dashboards.core import ModifiedFlags
from qua_dashboards.video_mode.sweep_axis import SweepAxis
from qua_dashboards.video_mode.scan_modes import ScanMode

from qua_dashboards.video_mode.inner_loop_actions.inner_loop_action import (
    InnerLoopAction,
)
from qua_dashboards.video_mode.inner_loop_actions.basic_inner_loop_action import (
    BasicInnerLoopAction,
)
from quam_builder.architecture.quantum_dots.components import GateSet, VoltageGate


logger = logging.getLogger(__name__)

__all__ = ["OPXDataAcquirer"]


class OPXDataAcquirer(Base2DDataAcquirer):
    """
    Data acquirer for OPX devices using a QUAM Machine object.

    This class handles communication with the Quantum Orchestration Platform (QOP)
    by generating QUA programs based on a QUAM machine configuration,
    executing them, and processing the results for 2D video mode display.
    It leverages a background thread (from BaseDataAcquirer) for continuous
    data acquisition and software averaging.
    """

    stream_vars_default = ["I", "Q"]  # Default stream variables expected from QUA
    result_types_default = ["I", "Q", "amplitude", "phase"]

    def __init__(
        self,
        *,
        qmm: QuantumMachinesManager,
        machine: Any,
        gate_set: GateSet,
        x_axis_name: str,
        y_axis_name: str,
        scan_modes: Dict[str, ScanMode],
        available_readout_pulses: List[ReadoutPulse],
        qua_inner_loop_action: Optional[InnerLoopAction] = None,
        component_id: str = "opx-data-acquirer",
        num_software_averages: int = 1,
        acquisition_interval_s: float = 0.1,
        result_type: Literal["I", "Q", "amplitude", "phase"] = "I",
        initial_delay_s: Optional[float] = None,
        stream_vars: Optional[List[str]] = None,
        inner_loop_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """
        Initializes the OPXDataAcquirer.

        Args:
            qmm: The QuantumMachinesManager instance.
            machine: The QUAM Machine instance to use for generating the QUA config.
            gate_set: The GateSet object containing voltage channels.
            scan_modes: A dictionary of scan modes defining how the 2D grid is traversed.
            x_axis_name: Name of the X sweep axis (must match a GateSet channel or virtual gate).
            y_axis_name: Name of the Y sweep axis (must match a GateSet channel or virtual gate).
            available_readout_pulses: A list of the QUAM Pulse objects to measure.
            qua_inner_loop_action: Optional custom QUA inner loop action. If not provided,
                                   BasicInnerLoopAction will be created automatically.
            component_id: Unique ID for Dash elements.
            num_software_averages: Number of raw snapshots for software averaging.
            acquisition_interval_s: Target interval for acquiring a new raw snapshot.
            result_type: The type of result to derive from I/Q data.
            initial_delay_s: Initial delay in seconds before starting each full scan in QUA.
            stream_vars: List of stream variables (e.g., ["I", "Q"]) expected from QUA.
                         Defaults to ["I", "Q"].
            inner_loop_kwargs: Additional arguments for BasicInnerLoopAction creation.
            **kwargs: Additional arguments for Base2DDataAcquirer.
        """
        sweep_axes = self._generate_sweep_axes(gate_set)
        super().__init__(
            sweep_axes=sweep_axes,
            x_axis_name=x_axis_name,
            y_axis_name=y_axis_name,
            component_id=component_id,
            num_software_averages=num_software_averages,
            acquisition_interval_s=acquisition_interval_s,
            **kwargs,
        )

        self.qmm: QuantumMachinesManager = qmm
        self.machine: Any = machine
        self.gate_set: GateSet = gate_set
        self.qua_config: Dict[str, Any] = self.machine.generate_config()
        self.qm: Any = None

        # Create BasicInnerLoopAction if not provided
        if qua_inner_loop_action is None:
            inner_loop_kwargs = inner_loop_kwargs or {}
            self.qua_inner_loop_action = BasicInnerLoopAction(
                gate_set=gate_set,
                x_axis_name=self.x_axis.name,
                y_axis_name=self.y_axis.name,
                **inner_loop_kwargs,
            )
        else:
            self.qua_inner_loop_action = qua_inner_loop_action
        self.scan_modes = scan_modes
        self.scan_2d: ScanMode = next(iter(self.scan_modes.values()))
        self.scan_1d: ScanMode = LineScan()

        self.initial_delay_s: Optional[float] = initial_delay_s
        self.qua_program: Optional[Program] = None
        self.qm_job: Optional[RunningQmJob] = None
        self._compilation_flags: ModifiedFlags = ModifiedFlags.NONE

        self.result_type: str = result_type
        self._raw_qua_results: Dict[str, np.ndarray] = {}
        self.stream_vars: List[str] = stream_vars or self.stream_vars_default
        self.result_types: List[str] = self.result_types_default
        self.available_readout_pulses = available_readout_pulses
        self.non_voltage_pulses = self.available_readout_pulses
        self.freq_sweep_axes, self.amp_sweep_axes = self._generate_non_voltage_axes(self.non_voltage_pulses)
        self._ensure_pulse_names()
        self._configure_readout()
        self._rebuild_stream_vars()
        

    def _ensure_pulse_names(self):
        """
        Check for pulse name "half_max_square" in each gate_set channel, necessary for operation.
        GateSet.new_sequence() does this check, but this function added to have a conditional regeneration of config and reinitialisation
        """
        for ch in self.gate_set.channels.values():
            if "half_max_square" not in ch.operations.keys():
                from quam.components import pulses

                if hasattr(ch.opx_output, "output_mode"):
                    if ch.opx_output.output_mode == "amplified":
                        ch.operations["half_max_square"] = pulses.SquarePulse(
                            amplitude=1.25, length=16
                        )
                    else:
                        ch.operations["half_max_square"] = pulses.SquarePulse(
                            amplitude=0.25, length=16
                        )
                else:
                    ch.operations["half_max_square"] = pulses.SquarePulse(
                        amplitude=0.25, length=16
                    )
                logger.info(
                    f"Default pulse not in channel '{ch.name}'. Adding operation 'half_max_square' and regenerating config"
                )
                self.qua_config = self.machine.generate_config()
                self.qm = None
                self.initialize_qm()

    def _configure_readout(self):
        """
        Searches the machine channels and finds the appropriate readout channels.
        """
        self.available_readout_channels = {}
        self.readout_pulse_mapping = {}
        for p in self.available_readout_pulses: 
            self.available_readout_channels[p.channel.name] = p.channel
            self.readout_pulse_mapping[p.channel.name] = p

        self.qua_inner_loop_action.readout_pulse_mapping = self.readout_pulse_mapping

        self.selected_readout_channels = (
            [self.available_readout_channels[self.available_readout_pulses[0].channel.name]]
            if self.available_readout_channels
            else []
        )

        self.qua_inner_loop_action.selected_readout_channels = (
            self.selected_readout_channels
        )

    def _rebuild_stream_vars(self):
        """
        Build the number of relevant Stream Vars, based on how many readout channels to acquire.
        - If only one readout channel is selected, then stream vars is the default (I, Q)
        - If more than one readout channel is selected, then it builds the appropriate number of vars (I:..., Q:...) for each readout channel, effectively creating 2xN
            stream vars for N readout channels
        """
        if len(self.selected_readout_channels) <= 1:
            self.stream_vars = self.stream_vars_default.copy()
        else:
            svars = []
            for channel in self.selected_readout_channels:
                svars = svars + [f"I:{channel.name}", f"Q:{channel.name}"]
            self.stream_vars = svars

    @property
    def current_scan_mode(self) -> str:
        for (name, mode) in self.scan_modes.items():
            if self.scan_2d == mode: 
                return name
        return next(iter(self.scan_modes.keys()))
    
    def set_scan_mode(self, name:str) -> None:
        if self.scan_2d is self.scan_modes[name]:
            return
        self.scan_2d = self.scan_modes[name]
        self._halt_acquisition()
        self._compilation_flags |= ModifiedFlags.PROGRAM_MODIFIED


    @property
    def scan_mode(self):
        if self._is_1d:
            return self.scan_1d
        else:
            return self.scan_2d

    def ensure_axis(self) -> None:
        gs = self.gate_set
        have = {ax.name for ax in self.sweep_axes}
        for nm in gs.valid_channel_names:
            if nm not in have:
                self.sweep_axes.append(SweepAxis(name=nm, units = "V"))


    @staticmethod
    def _generate_non_voltage_axes(available_pulses):
        """
        Checks through the available readout pulses and creates sweepaxis lists based off of them. 
        If you have qubit elements, be sure to add them here; the frequency and drive power sweepaxes will be created
        """
        drive_axes: List[SweepAxis] = []
        freq_axes: List[SweepAxis] = []
        default_freq_span = 20e6
        default_points = 51
        default_amp_span = 0.01
        for pulse in available_pulses: 
            channel_name = pulse.channel.name
            freq_axes.append(
                SweepAxis(name = f"{channel_name}_frequency", 
                          offset_parameter = None, 
                          non_voltage_offset=pulse.channel.intermediate_frequency,
                          span = default_freq_span, 
                          points = default_points, 
                          units = "Hz"
                )
            )
            drive_axes.append(
                SweepAxis(name = f"{channel_name}_drive", 
                          offset_parameter = None,
                          non_voltage_offset=pulse.amplitude, 
                          span = default_amp_span, 
                          points = default_points, 
                )
            )

        return freq_axes, drive_axes

    @staticmethod
    def _generate_sweep_axes(gate_set: GateSet) -> List[SweepAxis]:
        sweep_axes: List[SweepAxis] = []
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
                offset_parameter = None
            sweep_axes.append(
                SweepAxis(
                    name=channel_name,
                    offset_parameter=offset_parameter,
                    attenuation=attenuation,
                    units = "V"
                )
            )
        return sweep_axes
    
    @staticmethod
    def _sweep_vals_validator(vals, mode, dbm):
        """
        Validate the sweep vals based on the X or Y mode. 
        - If mode is 'Frequency', then round the sweepvalues to the nearest int
        - If mode if 'Drive', then checks for dbm bool and converts to volts if necessary
        """
        vals = np.array(vals)
        if mode == "Frequency": 
            vals = [int(round(float(v))) for v in vals]
        if mode == "Drive":
            if dbm: 
                vals = unit.dBm2volts(vals)
        return vals

    def generate_qua_program(self) -> Program:
        """
        Generates the QUA program for the 2D scan.
        """
        x_qua_values = self._sweep_vals_validator(list(self.x_axis.sweep_values_unattenuated), self.x_mode, self.x_axis.dbm)
        y_qua_values = self._sweep_vals_validator(list(self.y_axis.sweep_values_unattenuated), self.y_mode, self.y_axis.dbm)
            
        self.qua_inner_loop_action.selected_readout_channels = (
            self.selected_readout_channels
        )
        with program() as prog:
            qua_streams = {var: declare_stream() for var in self.stream_vars}

            with infinite_loop_():
                self.qua_inner_loop_action.initial_action()
                if self.initial_delay_s is not None and self.initial_delay_s > 0:
                    wait(int(self.initial_delay_s * 1e9 / 4))

                for x_qua_var, y_qua_var in self.scan_mode.scan(
                    x_vals=x_qua_values,
                    y_vals=y_qua_values,
                    x_mode = self.x_mode, 
                    y_mode = self.y_mode  # type: ignore
                ):
                    measured_qua_values = self.qua_inner_loop_action(
                        x_qua_var, y_qua_var
                    )

                    if not isinstance(measured_qua_values, tuple):
                        measured_qua_values = tuple(
                            measured_qua_values,
                        )

                    if len(measured_qua_values) != len(self.stream_vars):
                        raise ValueError(
                            f"Number of values returned by qua_inner_loop_action ({len(measured_qua_values)}) "
                            f"does not match number of stream_vars ({len(self.stream_vars)})."
                        )

                    for var_name, qua_value_to_save in zip(
                        self.stream_vars, measured_qua_values
                    ):
                        save(qua_value_to_save, qua_streams[var_name])

                self.qua_inner_loop_action.final_action()

            num_points_total = self.x_axis.points * self.y_axis.points
            with stream_processing():
                buffered_streams = {
                    var: qua_streams[var].buffer(num_points_total)
                    for var in self.stream_vars
                }

                combined_qua_stream = buffered_streams[self.stream_vars[0]]
                for i in range(1, len(self.stream_vars)):
                    combined_qua_stream = combined_qua_stream.zip(
                        buffered_streams[self.stream_vars[i]]
                    )  # type: ignore

                combined_qua_stream.save("all_streams_combined")

        self.qua_program = prog
        return prog

    def initialize_qm(self):
        if self.qm_job is not None:
            try:
                if self.qm_job.status == "running":
                    logger.info(f"Halting existing QM job for {self.component_id}.")
                    self.qm_job.halt()
                self.qm_job = None
            except Exception as e:
                logger.warning(f"Error halting previous QM job: {e}")

        if self.qua_config is None:
            self.qua_config = self.machine.generate_config()

        if self.qm is None:
            self.qm = self.qmm.open_qm(self.qua_config)  # type: ignore

    def execute_program(self, validate_running: bool = True):
        if self.qua_program is None:
            logger.info(f"Generating QUA program for {self.component_id}.")
            self.generate_qua_program()

        logger.info(f"Executing QUA program for {self.component_id}.")
        if self.qm is None:
            self.initialize_qm()
        self.qm_job = self.qm.execute(self.qua_program)  # type: ignore

        if validate_running:
            try:
                handle = self.qm_job.result_handles.get("all_streams_combined")
                handle.wait_for_values(1, timeout=20)
                logger.info(f"QM job for {self.component_id} started successfully.")
            except Exception as e:
                logger.error(
                    f"QM job for {self.component_id} failed to start or produce initial values: {e}"
                )
                raise

    def _flat_to_2d(self, flat: np.ndarray) -> np.ndarray:
        """
        Takes a flat numpy array of data, and build a 2D plot based on the appropriate shape and scan mode indices.
        """
        shape = (self.y_axis.points, self.x_axis.points)
        output_data_2d = np.zeros(shape, dtype=flat.dtype)
        x_indices, y_indices = self.scan_mode.get_idxs(
            x_points=self.x_axis.points, y_points=self.y_axis.points
        )
        for i, (y, x) in enumerate(zip(y_indices, x_indices)):
            output_data_2d[y, x] = flat[i]
        return output_data_2d

    def _process_fetched_results(self, fetched_qua_results: Tuple) -> np.ndarray:
        """
        Processes the raw tuple from QUA's fetch_all into a dictionary of named arrays,
        then derives the final 2D array based on self.result_type.
        """
        if len(fetched_qua_results) != len(self.stream_vars):
            logger.warning(
                f"Fetched {len(fetched_qua_results)} streams, but expected {len(self.stream_vars)}. "
                f"Expected stream vars: {self.stream_vars}. Likely indicates configuration change in progress - resetting compilation flags."
            )
            self._compilation_flags |= ModifiedFlags.PROGRAM_MODIFIED

        self._raw_qua_results = dict(zip(self.stream_vars, fetched_qua_results))
        expected_points = self.x_axis.points * self.y_axis.points
        num_sel = len(self.selected_readout_channels)

        def _normalize_flat(flat: np.ndarray) -> np.ndarray:
            """
            Trim or pad a 1D stream of samples to exactly one 2D frame and reshape.
            - If more than one frame is concatenated, then keep only the most recent full frame
            - If fewer samples are available than a full frame, returns a 2D array filled with NaNs
            - Otherwise, reshape samples into the appropriate dimensions
            """
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

        if num_sel <= 1:
            if self.result_type == "I" and "I" in self._raw_qua_results:
                flat = self._raw_qua_results["I"]
            elif self.result_type == "Q" and "Q" in self._raw_qua_results:
                flat = self._raw_qua_results["Q"]
            elif self.result_type == "amplitude":
                if "I" not in self._raw_qua_results or "Q" not in self._raw_qua_results:
                    raise ValueError(
                        "Cannot calculate amplitude without 'I' and 'Q' streams."
                    )
                flat = np.abs(
                    self._raw_qua_results["I"] + 1j * self._raw_qua_results["Q"]
                )
            elif self.result_type == "phase":
                if "I" not in self._raw_qua_results or "Q" not in self._raw_qua_results:
                    raise ValueError(
                        "Cannot calculate phase without 'I' and 'Q' streams."
                    )
                flat = np.angle(
                    self._raw_qua_results["I"] + 1j * self._raw_qua_results["Q"]
                )
            else:
                raise ValueError(
                    f"Invalid result_type: '{self.result_type}'. "
                    f"Must be one of {self.result_types} or a direct stream var: {self.stream_vars}"
                )
            return _normalize_flat(flat)

        else:
            # Multi-readout: return (R, X) in 1D, (R, Y, X) in 2D
            is_1d = (self.y_axis_name is None) or (self.y_axis.points == 1)
            expected_points = self.x_axis.points * (1 if is_1d else self.y_axis.points)

            output_layers = []
            names = [ch.name for ch in self.selected_readout_channels]
            for name in names:
                if self.result_type == "I":
                    key = f"I:{name}"
                    flat = np.asarray(self._raw_qua_results[key])
                elif self.result_type == "Q":
                    key = f"Q:{name}"
                    flat = np.asarray(self._raw_qua_results[key])
                elif self.result_type == "amplitude":
                    kI, kQ = f"I:{name}", f"Q:{name}"
                    flat = np.abs(
                        np.asarray(self._raw_qua_results[kI])
                        + 1j * np.asarray(self._raw_qua_results[kQ])
                    )
                elif self.result_type == "phase":
                    kI, kQ = f"I:{name}", f"Q:{name}"
                    flat = np.angle(
                        np.asarray(self._raw_qua_results[kI])
                        + 1j * np.asarray(self._raw_qua_results[kQ])
                    )
                else:
                    raise ValueError(f"Invalid result_type '{self.result_type}'.")

                # trim/pad per-layer, similar to _normalize_flat
                if flat.size > expected_points:
                    flat = flat[-expected_points:]
                if flat.size < expected_points:
                    if is_1d:
                        return np.full(
                            (len(names), self.x_axis.points), np.nan, dtype=flat.dtype
                        )
                    else:
                        return np.full(
                            (len(names), self.y_axis.points, self.x_axis.points),
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

    def perform_actual_acquisition(self) -> np.ndarray:
        if self._compilation_flags & ModifiedFlags.CONFIG_MODIFIED:
            logger.info(f"Config regeneration triggered for {self.component_id}.")
            self._regenerate_config_and_reopen_qm()
            self._compilation_flags = ModifiedFlags.NONE
        elif self._compilation_flags & ModifiedFlags.PROGRAM_MODIFIED:
            self._halt_acquisition()
            self.execute_program()
            self._compilation_flags = ModifiedFlags.NONE

        if self.qm_job is None or self.qm_job.status != "running":
            logger.warning(
                f"QM job for {self.component_id} is not running or None. Attempting to re-initialize."
            )
            self.initialize_qm()
            self.execute_program()
            if self.qm_job is None:
                raise RuntimeError(
                    f"Failed to initialize QM job for {self.component_id}."
                )

        start_time = time.perf_counter()
        try:
            result_handle = self.qm_job.result_handles.get("all_streams_combined")
            fetched_results_tuple = result_handle.fetch_all()
        except Exception as e:
            logger.error(
                f"Error fetching results from QM job for {self.component_id}: {e}"
            )
            raise

        acquisition_time = (time.perf_counter() - start_time) * 1000
        logger.debug(
            f"{self.component_id}: Fetched QUA results in {acquisition_time:.2f} ms."
        )

        return self._process_fetched_results(fetched_results_tuple)

    def _regenerate_config_and_reopen_qm(self) -> None:
        logger.info(f"Regenerating QUA config for {self.component_id} from machine.")
        self.qua_config = self.machine.generate_config()
        # Necessary to force qm to re-open
        self.qm = None
        self.initialize_qm()
        self.execute_program()

    def update_parameters(self, parameters: Dict[str, Dict[str, Any]]) -> ModifiedFlags:
        flags = super().update_parameters(parameters)
        try:
            for ax in self.sweep_axes + self.freq_sweep_axes + self.amp_sweep_axes:
                child_flags = ax.update_parameters(parameters)
                flags |= child_flags
        except Exception as e:
            logger.warning("Axis dispatch error: %s", e)

        try:
            il_flags = self.qua_inner_loop_action.update_parameters(parameters)
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
                and self.qua_inner_loop_action.ramp_duration != params["ramp_duration"]
            ):
                self.qua_inner_loop_action.ramp_duration = params["ramp_duration"]
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
                    old_stream_vars = self.stream_vars.copy()
                    self.selected_readout_channels = new_objs
                    self.qua_inner_loop_action.selected_readout_channels = (
                        self.selected_readout_channels
                    )

                    self._rebuild_stream_vars()

                    if old_stream_vars != self.stream_vars:
                        logger.info(
                            f"Stream vars changed from {old_stream_vars} to {self.stream_vars}, forcing program recompile"
                        )
                        self._halt_acquisition()
                        self.qm_job = None
                        flags |= ModifiedFlags.PROGRAM_MODIFIED
                    else:
                        flags |= ModifiedFlags.PARAMETERS_MODIFIED

            if "x-mode" in params and params["x-mode"] != self.x_mode: 
                self.x_mode = params["x-mode"]
                flags |= ModifiedFlags.PROGRAM_MODIFIED | ModifiedFlags.PARAMETERS_MODIFIED

            if "y-mode" in params and params["y-mode"] != self.y_mode: 
                self.y_mode = params["y-mode"]
                flags |= ModifiedFlags.PROGRAM_MODIFIED | ModifiedFlags.PARAMETERS_MODIFIED


        self._compilation_flags |= flags & (
            ModifiedFlags.PROGRAM_MODIFIED | ModifiedFlags.CONFIG_MODIFIED
        )
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
                self.qua_inner_loop_action, "get_dash_components"
            ):
                components.extend(
                    self.qua_inner_loop_action.get_dash_components(
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
                    dims=("readout", self.x_axis.name),
                    coords={"readout": labels, self.x_axis.name: x_coords},
                    attrs={"long_name": "Signal"},
                )
                data_xr = self.selected_function(data_xr)
                x_attrs = {"label": self.x_axis.label or self.x_axis.name}
                if self.x_axis.units is not None:
                    x_attrs["units"] = self.x_axis.units
                data_xr.coords[self.x_axis.name].attrs.update(x_attrs)
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
                dims=("readout", self.y_axis.name, self.x_axis.name),
                coords={
                    "readout": labels,
                    self.y_axis.name: y_coords,
                    self.x_axis.name: x_coords,
                },
                attrs={"long_name": "Signal"},
            )
            data_xr = self.selected_function(data_xr)
            for axis in [self.x_axis, self.y_axis]:
                attrs = {"label": axis.label or axis.name}
                if axis.units is not None:
                    attrs["units"] = (
                        self.x_axis.units if axis is self.x_axis else self.y_axis.units
                    )
                data_xr.coords[axis.name].attrs.update(attrs)

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
        components.extend(self.qua_inner_loop_action.get_components())
        return components

    def _halt_acquisition(self) -> None:
        logger.info(f"Program recompile triggered for {self.component_id}.")
        if (
            self.qm_job is not None
            and getattr(self.qm_job, "status", None) == "running"
        ):
            try:
                self.qm_job.halt()
            except Exception as e:
                logger.warning(f"Halting previous QM job failed: {e}")
        self.qua_program = None

    def stop_acquisition(self) -> None:
        logger.info(f"OPXDataAcquirer ({self.component_id}) attempting to halt QM job.")
        if self.qm_job and self.qm_job.status == "running":
            try:
                self.qm_job.halt()
                self.qm_job = None
                logger.info(f"QM job for {self.component_id} halted.")
            except Exception as e:
                logger.warning(f"Error halting QM job for {self.component_id}: {e}")
        super().stop_acquisition()

    def mark_virtual_layer_changed(self, *, affects_config: bool = False):
        """Call this when a virtual-gate matrix was edited."""
        if affects_config:
            self._compilation_flags |= ModifiedFlags.CONFIG_MODIFIED
        else:
            self._compilation_flags |= ModifiedFlags.PROGRAM_MODIFIED


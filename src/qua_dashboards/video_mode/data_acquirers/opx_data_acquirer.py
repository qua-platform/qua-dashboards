import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import threading
import queue

from qm import QuantumMachinesManager, Program
from qm.jobs.running_qm_job import RunningQmJob
from qm.qua import (
    program,
    declare_stream,
    infinite_loop_,
    save,
    stream_processing,
    wait,
)

from qua_dashboards.video_mode.data_acquirers.base_gate_set_data_acquirer import BaseGateSetDataAcquirer
from qua_dashboards.core import ModifiedFlags
from qua_dashboards.video_mode.sweep_axis import (
    BaseSweepAxis,
    VoltageSweepAxis,

)
from qua_dashboards.voltage_control import VoltageControlComponent
from qua_dashboards.video_mode.inner_loop_actions.inner_loop_action import (
    InnerLoopAction,
)
from qua_dashboards.video_mode.inner_loop_actions.basic_inner_loop_action import (
    BasicInnerLoopAction,
)

logger = logging.getLogger(__name__)

__all__ = ["OPXDataAcquirer"]


class OPXDataAcquirer(BaseGateSetDataAcquirer):
    """
    Data acquirer for OPX devices using a QUAM Machine object.

    This class handles communication with the Quantum Orchestration Platform (QOP)
    by generating QUA programs based on a QUAM machine configuration,
    executing them, and processing the results for 2D video mode display.
    It leverages a background thread (from BaseDataAcquirer) for continuous
    data acquisition and software averaging.
    """

    stream_vars_default = ["I", "Q"]  # Default stream variables expected from QUA

    def __init__(
        self,
        *,
        qmm: QuantumMachinesManager,
        machine: Any,
        inner_loop_action: Optional[InnerLoopAction] = None,
        component_id: str = "opx-data-acquirer",
        initial_delay_s: Optional[float] = None,
        stream_vars: Optional[List[str]] = None,
        inner_loop_kwargs: Optional[Dict[str, Any]] = None,
        inner_functions_dict: Optional[Dict] = None,
        apply_compensation_pulse: bool = True, 
        voltage_control_component: Optional["VoltageControlComponent"] = None,
        mid_scan_compensation: bool = False, 
        buffer_frames: int = 20,
        calibrations: Optional[Dict] = None,
        **kwargs: Any,
    ):
        """
        Initializes the OPXDataAcquirer.

        Args:
            qmm: The QuantumMachinesManager instance.
            machine: The QUAM Machine instance to use for generating the QUA config.
            inner_loop_action: Optional custom QUA inner loop action. If not provided,
                                   BasicInnerLoopAction will be created automatically.
            component_id: Unique ID for Dash elements.
            initial_delay_s: Initial delay in seconds before starting each full scan in QUA.
            stream_vars: List of stream variables (e.g., ["I", "Q"]) expected from QUA.
                         Defaults to ["I", "Q"].
            inner_loop_kwargs: Additional arguments for BasicInnerLoopAction creation.
            **kwargs: Additional arguments for Base2DDataAcquirer.
        """
        self.voltage_control_component = voltage_control_component

        super().__init__(
            component_id=component_id,
            **kwargs,
        )

        self.qmm: QuantumMachinesManager = qmm
        self.machine: Any = machine
        self.qua_config: Dict[str, Any] = self.machine.generate_config()
        self.qm: Any = None

        # Create BasicInnerLoopAction if not provided
        if inner_loop_action is None:
            inner_loop_kwargs = inner_loop_kwargs or {}
            self.inner_loop_action = BasicInnerLoopAction(
                gate_set=self.gate_set,
                x_axis=self.x_axis,
                y_axis=self.y_axis,
                apply_compensation=apply_compensation_pulse,
                **inner_loop_kwargs,
            )
        else:
            self.inner_loop_action = inner_loop_action
        self._compiled_xy = None
        # Caching the scan mode indices for faster python side operation
        self._scan_idx_cache: Optional[Tuple[Tuple[int,int,int], np.ndarray, np.ndarray]] = None

        self.initial_delay_s: Optional[float] = initial_delay_s
        self.qua_program: Optional[Program] = None
        self.qm_job: Optional[RunningQmJob] = None

        self._raw_qua_results: Dict[str, np.ndarray] = {}
        self.stream_vars: List[str] = stream_vars or self.stream_vars_default
        self._ensure_pulse_names()
        self._rebuild_stream_vars()
        self._configure_readout()
        self._compiled_stream_vars: Optional[List[str]] = None
        self.inner_functions_dict = inner_functions_dict or {}
        self.mid_scan_compensation = mid_scan_compensation
        self.buffer_frames = buffer_frames
        self.calibrations: Optional[Dict] = calibrations
        self._frame_queue = queue.Queue(maxsize=30)
        self._fetch_thread = None
        self._fetch_event = threading.Event()  # set = fetch thread running
        # Number of times UI actually displayed fallback single-frame output.
        # Used to skip duplicated leading frames when buffered batches start arriving.
        self._single_frames_shown = 0
        self._last_frame = None
        # Counts only truly fresh frames consumed from the queue (not repeated last_frame).
        self._fresh_frame_seq = 0

    def _ensure_pulse_names(self) -> None:
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

    def _rebuild_stream_vars(self) -> None:
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

    def reset(self) -> None: 
        """Reset the acquirer with QM job"""
        if self.qm_job is not None: 
            try:
                if self.qm_job.status == "running":
                    self.qm_job.halt()
            except Exception as e:
                logger.warning(f"Error halting QM job during reset: {e}")
            self.qm_job = None
        self.qua_program = None
        self._compilation_flags |= ModifiedFlags.PROGRAM_MODIFIED
        super().reset()

    def shutdown(self) -> None: 
        """Fully shut down Video Mode"""
        logger.info(f"Shutdown requested on component {self.component_id}")
        self.stop_acquisition()
        if self.qm_job is not None: 
            try: 
                if self.qm_job.status == "running": 
                    self.qm_job.cancel()
            except Exception as e:
                logger.warning(f"Error halting QM Job: {e}")
            self.qm_job = None
        if self.qm is not None: 
            try: 
                self.qm.close()
            except Exception as e: 
                logger.warning(f"Error closing QM: {e}")
            self.qm = None

        self.qua_program = None
        logger.info(f"{self.component_id} shutdown complete")

    def generate_qua_program(self) -> Program:
        """
        Generates the QUA program for the 2D scan.
        """
        x_qua_values = self.x_axis.qua_sweep_values
        if self._is_1d:
            y_qua_values = None
        else:
            y_qua_values = self.y_axis.qua_sweep_values
        self._compiled_xy = (int(self.x_axis.points), (1 if self._is_1d else int(self.y_axis.points)), self._is_1d)

        self.inner_loop_action.selected_readout_channels = (
            self.selected_readout_channels
        )
        self._rebuild_stream_vars()
        with program() as prog:
            qua_streams = {var: declare_stream() for var in self.stream_vars}
            self._compiled_stream_vars = self.stream_vars.copy()

            with infinite_loop_():
                self.inner_loop_action.initial_action()
                if self.initial_delay_s is not None and self.initial_delay_s > 0:
                    wait(int(self.initial_delay_s * 1e9 / 4))

                for x_qua_var, y_qua_var in self.scan_mode.scan(
                    x_vals=x_qua_values,
                    y_vals=(y_qua_values if not self._is_1d else None),
                    x_mode=self.x_mode,
                    y_mode=(self.y_mode if not self._is_1d else None),  # type: ignore
                    compensation_pulse = None if self.mid_scan_compensation is False else self.inner_loop_action.voltage_sequence.apply_compensation_pulse,
                ):
                    measured_qua_values = self.inner_loop_action(
                        x_qua_var, y_qua_var
                    )

                    if not isinstance(measured_qua_values, tuple):
                        measured_qua_values = tuple(
                            measured_qua_values,
                        )

                    if len(measured_qua_values) != len(self.stream_vars):
                        raise ValueError(
                            f"Number of values returned by inner_loop_action ({len(measured_qua_values)}) "
                            f"does not match number of stream_vars ({len(self.stream_vars)})."
                        )

                    for var_name, qua_value_to_save in zip(
                        self.stream_vars, measured_qua_values
                    ):
                        save(qua_value_to_save, qua_streams[var_name])

                self.inner_loop_action.final_action()

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

                combined_qua_stream.save("latest_frame")  # Unbuffered - first frame fast
                active_buffer = self._resolve_buffer_frames(int(self.x_axis.points), int(self.y_axis.points))
                combined_qua_stream.buffer(active_buffer).save("all_streams_combined")

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

        if self.qm is not None: 
            try: 
                self.qm.close()
                self.qm = None
                logger.info(f"Closed QM for {self.component_id}")
            except Exception as e: 
                logger.warning(f"Error closing QM: {e}")

        if self.qua_config is None:
            self.qua_config = self.machine.generate_config()

        if self.qm is None:
            self.qm = self.qmm.open_qm(self.qua_config)  # type: ignore

    def execute_program(self, validate_running: bool = False, startup_timeout_s: float = 0.1):
        if self.qua_program is None:
            logger.info(f"Generating QUA program for {self.component_id}.")
            self.generate_qua_program()

        logger.info(f"Executing QUA program for {self.component_id}.")
        if self.qm is None:
            self.initialize_qm()
        self.qm_job = self.qm.execute(self.qua_program)  # type: ignore

        if validate_running and startup_timeout_s > 0:
            try:
                handle = self.qm_job.result_handles.get("all_streams_combined")
                handle.wait_for_values(1, timeout=startup_timeout_s)
                logger.info(f"QM job for {self.component_id} successfully produced initial values.")
            except Exception as e:
                logger.error(
                    f"QM job for {self.component_id} failed to start or produce initial values: {e}"
                )
                #raise

    def _process_fetched_results(self, fetched_qua_results: Tuple) -> np.ndarray:
        """
        Processes the raw tuple from QUA's fetch_all into a dictionary of named arrays,
        then derives the final 2D array based on self.result_type.
        """
        if fetched_qua_results is None:
            return np.full((self.y_axis.points, self.x_axis.points), np.nan)
        compiled = self._compiled_stream_vars or self.stream_vars
        is_multi_readout = len(compiled) > 2

        if len(fetched_qua_results) != len(compiled):
            logger.warning(
                f"Fetched {len(fetched_qua_results)} streams, but expected {len(self.stream_vars)}. "
                f"Expected stream vars: {self.stream_vars}. Likely indicates configuration change in progress - resetting compilation flags."
            )
            self._compilation_flags |= ModifiedFlags.PROGRAM_MODIFIED
            return np.full((self.y_axis.points, self.x_axis.points), np.nan)

        self._raw_qua_results = dict(zip(compiled, fetched_qua_results))
        expected_points = self.x_axis.points * self.y_axis.points
        num_sel = len(self.selected_readout_channels)

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
            channel_names = [ch.name for ch in self.selected_readout_channels]
            # Multi-readout: return (R, X) in 1D, (R, Y, X) in 2D
            is_1d = self._is_1d
            expected_points = self.x_axis.points * (1 if is_1d else self.y_axis.points)

            output_layers = []
            #names = [ch.name for ch in self.selected_readout_channels]
            for name in channel_names:
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

    def _resolve_buffer_frames(self, nx: int, ny: int) -> int:
        """Return the ideal buffer_frames for (nx, ny) via calibration interpolation.

        Falls back to self.buffer_frames if no calibration data is loaded.
        """
        if self.calibrations is None:
            return self.buffer_frames
        try:
            from scipy.interpolate import RegularGridInterpolator
            nx_vals = np.array(self.calibrations["nx_vals"])
            ny_vals = np.array(self.calibrations["ny_vals"])
            ideal_buffers = np.array(self.calibrations["ideal_buffers"], dtype=float)
            interp = RegularGridInterpolator(
                (nx_vals, ny_vals),
                ideal_buffers,
                method="linear",
                bounds_error=False,
                fill_value=None,  # nearest-edge extrapolation outside the grid
            )
            val = float(interp([[float(nx), float(ny)]])[0])
            return max(1, int(round(val)))
        except Exception as e:
            logger.warning(f"Calibration lookup failed: {e}. Using default buffer_frames={self.buffer_frames}.")
            return self.buffer_frames

    def _calculate_optimal_buffer_frames(self) -> int:
        if self._fetch_time_ms is None:
            return self.buffer_frames
        optimal = int(self._fetch_time_ms / self._plot_time_ms)
        return max(self._min_buffer_frames, min(self._max_buffer_frames, optimal))
    
    def _fetch_loop(self):
        """Continuous background fetcher using two QM streams.

        Uses latest_frame (unbuffered) for fast single-frame updates while the
        OPX fills its buffer.  Once all_streams_combined delivers a full batch,
        we skip any frames the user has already seen via the single-frame path
        (_single_frames_shown), then serve only the new frames.
        """
        while self._fetch_event.is_set():
            try:
                if self.qm_job is None or self.qm_job.status != "running":
                    time.sleep(0.01)
                    continue

                buffered_handle = self.qm_job.result_handles.get("all_streams_combined")
                latest_handle = self.qm_job.result_handles.get("latest_frame")

                if buffered_handle is None or latest_handle is None:
                    logger.debug("Result handles not available yet")
                    time.sleep(0.05)
                    continue

                buffered_results = buffered_handle.fetch_all()

                if buffered_results is not None and len(buffered_results) > 0:
                    # Skip frames the user already saw via the single-frame path.
                    # Clamp to len(batch)-1 so we always pass at least one frame from
                    # the first buffered batch and avoid a visible freeze.
                    start_idx = min(self._single_frames_shown, max(len(buffered_results) - 1, 0))
                    self._single_frames_shown = 0  # reset; subsequent batches shown in full
                    for frame_idx in range(start_idx, len(buffered_results)):
                        if not self._fetch_event.is_set():
                            break
                        frame = buffered_results[frame_idx]
                        processed = self._process_fetched_results(tuple(frame))
                        try:
                            self._frame_queue.put(processed, timeout=0.1)
                        except queue.Full:
                            pass
                else:
                    # Buffer not yet full — show individual frames as they arrive.
                    latest_result = latest_handle.fetch_all()
                    if latest_result is not None:
                        processed = self._process_fetched_results(tuple(latest_result))
                        try:
                            self._frame_queue.put(processed, timeout=0.1)
                        except queue.Full:
                            pass
                    else:
                        time.sleep(0.01)

            except Exception as e:
                logger.warning(f"Fetch loop error: {e}")
                time.sleep(0.1)
        
    def _clear_queue(self) -> None:
        """Clear all frames from the queue and reset tracking state."""
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        self._last_frame = None
        self._single_frames_shown = 0
        self._fresh_frame_seq = 0

    def perform_actual_acquisition(self) -> np.ndarray:
        if self._acquisition_status == "stopped":
            return np.full((self.y_axis.points, self.x_axis.points), np.nan)
        cur = (int(self.x_axis.points), 1 if self._is_1d else int(self.y_axis.points), self._is_1d)
        if self._compiled_xy is not None and cur != self._compiled_xy:
            logger.info(f"Scan shape changed {self._compiled_xy} -> {cur}. Forcing recompile.")
            self._clear_queue()
            self._compiled_stream_vars = None
            self._compilation_flags |= ModifiedFlags.PROGRAM_MODIFIED
        if self._compiled_stream_vars is not None:
            if len(self.selected_readout_channels) <= 1:
                expected_vars = self.stream_vars_default.copy()
            else:
                expected_vars = []
                for channel in self.selected_readout_channels:
                    expected_vars.extend([f"I:{channel.name}", f"Q:{channel.name}"])
            if self._compiled_stream_vars != expected_vars:
                logger.warning(f"Stream vars mismatch! Compiled: {self._compiled_stream_vars}, Expected: {expected_vars}. Regenerating program.")
                self._halt_acquisition()
                self._compilation_flags |= ModifiedFlags.PROGRAM_MODIFIED
        if self._compilation_flags & ModifiedFlags.CONFIG_MODIFIED:
            self._fetch_event.clear()
            self._clear_queue()
            logger.info(f"Config regeneration triggered for {self.component_id}.")
            self._regenerate_config_and_reopen_qm()
            self._compilation_flags = ModifiedFlags.NONE
        elif self._compilation_flags & ModifiedFlags.PROGRAM_MODIFIED:
            self._fetch_event.clear()
            self._clear_queue()
            self._halt_acquisition()
            self.execute_program()
            self._compilation_flags = ModifiedFlags.NONE

        if self.qm_job is None or self.qm_job.status != "running":
            if self._acquisition_status != "stopped":
                logger.warning(
                    f"QM job for {self.component_id} is not running or None. Attempting to re-initialize."
                )
                self.initialize_qm()
                self.execute_program()
                if self.qm_job is None:
                    raise RuntimeError(
                        f"Failed to initialize QM job for {self.component_id}."
                    )
            else:
                return np.full((self.y_axis.points, self.x_axis.points), np.nan)
            
        
        if self._fetch_thread is None or not self._fetch_thread.is_alive():
            self._fetch_event.set()
            self._fetch_thread = threading.Thread(target=self._fetch_loop, daemon=True)
            self._fetch_thread.start()

        try:
            frame = self._frame_queue.get_nowait()
            self._last_frame = frame
            # Fresh queued frame is now visible; any previous fallback-display count
            # should no longer be applied.
            self._single_frames_shown = 0
            self._fresh_frame_seq += 1
            return frame
        except queue.Empty:
            if self._last_frame is not None:
                # Count only frames that were actually displayed as fallback.
                self._single_frames_shown += 1
                return self._last_frame
            return np.full((self.y_axis.points, self.x_axis.points), np.nan)

    def get_latest_data(self) -> Dict[str, Any]:
        """
        Return latest data plus a fresh-frame sequence.

        Base 2D/GateSet wrappers can drop `seq` during xarray conversion. For
        smooth plotting cadence we expose a monotonic sequence that advances only
        when a genuinely new frame is consumed from the queue.
        """
        out = super().get_latest_data()
        out["seq"] = int(self._fresh_frame_seq)
        return out

    def _regenerate_config_and_reopen_qm(self) -> None:
        logger.info(f"Regenerating QUA config for {self.component_id} from machine.")
        self.qua_config = self.machine.generate_config()
        # Necessary to force qm to re-open
        self.qm = None
        self.initialize_qm()
        self.execute_program()


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
        self._fetch_event.clear()
        logger.info(f"OPXDataAcquirer ({self.component_id}) attempting to halt QM job.")
        super().stop_acquisition()
        if self.qm_job and self.qm_job.status == "running":
            try:
                self.qm_job.halt()
                self.qm_job = None
                logger.info(f"QM job for {self.component_id} halted.")
            except Exception as e:
                logger.warning(f"Error halting QM job for {self.component_id}: {e}")

    def update_parameters(self, parameters: Dict[str, Dict[str, Any]]) -> ModifiedFlags:
        """Update parameters and clear queue if anything changed."""
        flags = super().update_parameters(parameters)
        
        if flags != ModifiedFlags.NONE:
            self._clear_queue()
        
        return flags

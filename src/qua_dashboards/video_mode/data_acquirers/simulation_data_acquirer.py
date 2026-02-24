# qua_dashboards/video_mode/data_acquirers/random_data_acquirer.py
import logging
from time import sleep
from typing import Any, Dict, Optional, Tuple, Type

import numpy as np

from qua_dashboards.video_mode.data_acquirers.base_gate_set_data_acquirer import BaseGateSetDataAcquirer
from qua_dashboards.voltage_control.voltage_control_component import VoltageControlComponent
from qua_dashboards.video_mode.inner_loop_actions import InnerLoopAction, SimulatedInnerLoopAction
from qua_dashboards.video_mode.inner_loop_actions.simulators import BaseSimulator
from qua_dashboards.core import ModifiedFlags

logger = logging.getLogger(__name__)

__all__ = ["SimulationDataAcquirer", "SimulationDataAcquirerOPXOutput"]


class SimulationDataAcquirer(BaseGateSetDataAcquirer):
    """Data acquirer that generates random 2D data for simulation purposes.

    Inherits from Base2DDataAcquirer and simulates a delay for data acquisition.
    The actual data generation is a simple random number matrix.
    """

    def __init__(
        self,
        machine,
        simulator: BaseSimulator,
        acquire_time: float = 0.1,
        *,
        inner_loop_action: Optional[InnerLoopAction] = None,
        # inner_loop_action_cls: Optional[Type[InnerLoopAction]] = None,
        # inner_loop_action_kwargs: Optional[Dict[str, Any]] = None,
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
        self.voltage_control_component = voltage_control_component
        super().__init__(
            **kwargs,
        )
        self.acquire_time = acquire_time
        self._first_acquisition: bool = True
        logger.debug(
            f"Initializing SimulationDataAcquirer (ID: {self.component_id}) with "
            f"acquire_time: {self.acquire_time}s"
        )
        

        if inner_loop_action is None:
            inner_loop_action = SimulatedInnerLoopAction(
                gate_set=self.gate_set,
                x_axis=self.x_axis,
                y_axis=self.y_axis,
                simulator = simulator
            )
        self.inner_loop_action = inner_loop_action
        self._configure_readout()
        self.machine = machine

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

        # Ensure y_axis.points and x_axis.points are positive integers
        if self.y_axis.points <= 0 or self.x_axis.points <= 0:
            logger.warning(
                f"{self.component_id} (ID: {self.component_id}): Invalid points "
                f"({self.y_axis.points}x{self.x_axis.points}). Returning empty array."
            )
            return np.array([[]])  # Return a 2D empty array to avoid downstream errors
        
        num_readouts = len(self.selected_readout_channels)

        x_axis_vals = self.x_axis.sweep_values_with_offset
        y_axis_vals = self.y_axis.sweep_values_with_offset

        results = tuple(self.inner_loop_action(x_axis_vals, y_axis_vals))

        return self._process_fetched_results(results)

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

        def _to_scan_flat(data: np.ndarray) -> np.ndarray:
            """Convert a 2D frame into a 1D stream that follows the scan order."""
            data = np.asarray(data)
            if data.ndim != 2:
                return data.ravel()

            y_pts, x_pts = int(self.y_axis.points), int(self.x_axis.points)
            if x_pts != y_pts and data.shape == (x_pts, y_pts):
                data = data.T
            elif data.shape != (y_pts, x_pts):
                return data.ravel()

            x_indices, y_indices = self.get_scan_indices(x_pts=x_pts, y_pts=y_pts)
            return data[y_indices, x_indices]

        def _normalize_flat(flat: np.ndarray) -> np.ndarray:
            """
            Trim or pad a 1D stream of samples to exactly one 2D frame and reshape.
            - If more than one frame is concatenated, then keep only the most recent full frame
            - If fewer samples are available than a full frame, returns a 2D array filled with NaNs
            - Otherwise, reshape samples into the appropriate dimensions
            """
            flat = _to_scan_flat(flat)
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
                flat = _to_scan_flat(flat)
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


from .opx_data_acquirer import OPXDataAcquirer
class SimulationDataAcquirerOPXOutput(OPXDataAcquirer):
    def __init__(self, simulator, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.simulator = simulator
        self._first_acquisition = True 
        self.acquire_time = 10e-3
    
    def perform_actual_acquisition(self) -> np.ndarray:
        """Simulates data acquisition by sleeping and returning random data.

        This method is called by the background thread in BaseDataAcquirer.

        Returns:
            A 2D numpy array of random float values between 0 and 1, with
            dimensions (y_axis.points, x_axis.points).
        """
        if self._acquisition_status == "stopped":
            return np.full((self.y_axis.points, self.x_axis.points), np.nan)
        cur = (int(self.x_axis.points), 1 if self._is_1d else int(self.y_axis.points), self._is_1d)
        if self._compiled_xy is not None and cur != self._compiled_xy:
            logger.info(f"Scan shape changed {self._compiled_xy} -> {cur}. Forcing recompile.")
            self._halt_acquisition()
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
            logger.info(f"Config regeneration triggered for {self.component_id}.")
            self._regenerate_config_and_reopen_qm()
            self._compilation_flags = ModifiedFlags.NONE
        elif self._compilation_flags & ModifiedFlags.PROGRAM_MODIFIED:
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

        I, Q = self.simulator.measure_data(self.x_axis.name, self.y_axis.name, x_axis_vals, y_axis_vals, num_readouts)
        result = []
        for i in range(num_readouts): 
            result.extend([I[i], Q[i]])
        return self._process_fetched_results(result)

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

        def _to_scan_flat(data: np.ndarray) -> np.ndarray:
            """Convert a 2D frame into a 1D stream that follows the scan order."""
            data = np.asarray(data)
            if data.ndim != 2:
                return data.ravel()

            y_pts, x_pts = int(self.y_axis.points), int(self.x_axis.points)
            if x_pts != y_pts and data.shape == (x_pts, y_pts):
                data = data.T
            elif data.shape != (y_pts, x_pts):
                return data.ravel()

            x_indices, y_indices = self.get_scan_indices(x_pts=x_pts, y_pts=y_pts)
            return data[y_indices, x_indices]

        def _normalize_flat(flat: np.ndarray) -> np.ndarray:
            """
            Trim or pad a 1D stream of samples to exactly one 2D frame and reshape.
            - If more than one frame is concatenated, then keep only the most recent full frame
            - If fewer samples are available than a full frame, returns a 2D array filled with NaNs
            - Otherwise, reshape samples into the appropriate dimensions
            """
            flat = _to_scan_flat(flat)
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
                flat = _to_scan_flat(flat)
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
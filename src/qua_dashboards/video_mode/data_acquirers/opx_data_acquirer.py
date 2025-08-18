import logging
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
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
from dash import html
import dash_bootstrap_components as dbc

from qua_dashboards.core.base_updatable_component import BaseUpdatableComponent
from qua_dashboards.video_mode.data_acquirers.base_2d_data_acquirer import (
    Base2DDataAcquirer,
)
from qua_dashboards.core import ModifiedFlags
from qua_dashboards.video_mode.sweep_axis import SweepAxis
from qua_dashboards.video_mode.scan_modes import ScanMode

from qua_dashboards.video_mode.inner_loop_actions.inner_loop_action import (
    InnerLoopAction,
)


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
        qua_inner_loop_action: InnerLoopAction,
        scan_mode: ScanMode,
        x_axis: SweepAxis,
        y_axis: SweepAxis,
        component_id: str = "opx-data-acquirer",
        num_software_averages: int = 1,
        acquisition_interval_s: float = 0.1,
        result_type: Literal["I", "Q", "amplitude", "phase"] = "I",
        initial_delay_s: Optional[float] = None,
        stream_vars: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """
        Initializes the OPXDataAcquirer.

        Args:
            qmm: The QuantumMachinesManager instance.
            machine: The QUAM Machine instance to use for generating the QUA config.
            qua_inner_loop_action: The QUA inner loop action to execute at each scan point.
            scan_mode: The scan mode defining how the 2D grid is traversed.
            x_axis: The X sweep axis.
            y_axis: The Y sweep axis.
            component_id: Unique ID for Dash elements.
            num_software_averages: Number of raw snapshots for software averaging.
            acquisition_interval_s: Target interval for acquiring a new raw snapshot.
            result_type: The type of result to derive from I/Q data.
            initial_delay_s: Initial delay in seconds before starting each full scan in QUA.
            stream_vars: List of stream variables (e.g., ["I", "Q"]) expected from QUA.
                         Defaults to ["I", "Q"].
            **kwargs: Additional arguments for Base2DDataAcquirer.
        """
        super().__init__(
            x_axis=x_axis,
            y_axis=y_axis,
            component_id=component_id,
            num_software_averages=num_software_averages,
            acquisition_interval_s=acquisition_interval_s,
            **kwargs,
        )

        self.qmm: QuantumMachinesManager = qmm
        self.machine: Any = machine
        self.qua_config: Dict[str, Any] = self.machine.generate_config()
        self.qm: Any = None

        self.qua_inner_loop_action: InnerLoopAction = qua_inner_loop_action
        self.scan_mode: ScanMode = scan_mode

        self.initial_delay_s: Optional[float] = initial_delay_s
        self.qua_program: Optional[Program] = None
        self.qm_job: Optional[RunningQmJob] = None
        self._compilation_flags: ModifiedFlags = ModifiedFlags.NONE

        self.result_type: str = result_type
        self._raw_qua_results: Dict[str, np.ndarray] = {}
        self.stream_vars: List[str] = stream_vars or self.stream_vars_default
        self.result_types: List[str] = self.result_types_default

    def generate_qua_program(self) -> Program:
        """
        Generates the QUA program for the 2D scan.
        """
        x_qua_values = list(self.x_axis.sweep_values_unattenuated)
        y_qua_values = list(self.y_axis.sweep_values_unattenuated)

        with program() as prog:
            qua_streams = {var: declare_stream() for var in self.stream_vars}

            with infinite_loop_():
                self.qua_inner_loop_action.initial_action()
                if self.initial_delay_s is not None and self.initial_delay_s > 0:
                    wait(int(self.initial_delay_s * 1e9 / 4))

                for x_qua_var, y_qua_var in self.scan_mode.scan(
                    x_vals=x_qua_values,
                    y_vals=y_qua_values,  # type: ignore
                ):
                    measured_qua_values = self.qua_inner_loop_action(
                        x_qua_var, y_qua_var
                    )

                    if not isinstance(measured_qua_values, tuple):
                        measured_qua_values = (measured_qua_values,)

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
        logger.info(f"Generating QUA program for {self.component_id}.")
        self.qua_program = None
        self.generate_qua_program()
        logger.info(f"Executing QUA program for {self.component_id}.")
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

    def _process_fetched_results(self, fetched_qua_results: Tuple) -> np.ndarray:
        """
        Processes the raw tuple from QUA's fetch_all into a dictionary of named arrays,
        then derives the final 2D array based on self.result_type.
        """
        if len(fetched_qua_results) != len(self.stream_vars):
            raise ValueError(
                f"Fetched {len(fetched_qua_results)} streams, but expected {len(self.stream_vars)}. "
                f"Stream vars: {self.stream_vars}"
            )

        self._raw_qua_results = dict(zip(self.stream_vars, fetched_qua_results))

        if self.result_type == "I" and "I" in self._raw_qua_results:
            output_data_flat = self._raw_qua_results["I"]
        elif self.result_type == "Q" and "Q" in self._raw_qua_results:
            output_data_flat = self._raw_qua_results["Q"]
        elif self.result_type == "amplitude":
            if "I" not in self._raw_qua_results or "Q" not in self._raw_qua_results:
                raise ValueError(
                    "Cannot calculate amplitude without 'I' and 'Q' streams."
                )
            output_data_flat = np.abs(
                self._raw_qua_results["I"] + 1j * self._raw_qua_results["Q"]
            )
        elif self.result_type == "phase":
            if "I" not in self._raw_qua_results or "Q" not in self._raw_qua_results:
                raise ValueError("Cannot calculate phase without 'I' and 'Q' streams.")
            output_data_flat = np.angle(
                self._raw_qua_results["I"] + 1j * self._raw_qua_results["Q"]
            )
        else:
            part1 = "Invalid result_type: '{}'.".format(self.result_type)
            part2 = " Must be one of {} ".format(self.result_types)
            part3 = "or a direct stream variable name from {}.".format(self.stream_vars)
            error_msg = part1 + part2 + part3
            raise ValueError(error_msg)

        shape = (self.y_axis.points, self.x_axis.points)
        output_data_2d = np.zeros(shape, dtype=output_data_flat.dtype)
        x_indices, y_indices = self.scan_mode.get_idxs(
            x_points=self.x_axis.points, y_points=self.y_axis.points
        )
        for i, (y, x) in enumerate(zip(y_indices, x_indices)):
            output_data_2d[y, x] = output_data_flat[i]

        return output_data_2d

    def perform_actual_acquisition(self) -> np.ndarray:
        if self._compilation_flags & ModifiedFlags.CONFIG_MODIFIED:
            logger.info(f"Config regeneration triggered for {self.component_id}.")
            self._regenerate_config_and_reopen_qm()
            self._compilation_flags = ModifiedFlags.NONE
        elif self._compilation_flags & ModifiedFlags.PROGRAM_MODIFIED:
            logger.info(f"Program recompile triggered for {self.component_id}.")
            self.qua_program = None  # Clear the program to force a re-generation
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
        self.initialize_qm()
        self.execute_program()

    def update_parameters(self, parameters: Dict[str, Dict[str, Any]]) -> ModifiedFlags:
        flags = super().update_parameters(parameters)

        if self.component_id in parameters:
            params = parameters[self.component_id]
            if "result-type" in params and self.result_type != params["result-type"]:
                self.result_type = params["result-type"]
                flags |= ModifiedFlags.PARAMETERS_MODIFIED

        self._compilation_flags = flags
        return flags

    def get_dash_components(self, include_subcomponents: bool = True) -> List[Any]:
        components = super().get_dash_components(include_subcomponents)

        result_type_selector = dbc.Row(
            [
                dbc.Label("Result Type", width="auto", className="col-form-label"),
                dbc.Col(
                    dbc.Select(
                        id=self._get_id("result-type"),
                        options=[
                            {"label": rt, "value": rt} for rt in self.result_types
                        ],
                        value=self.result_type,
                    ),
                    width=True,
                ),
            ],
            className="mb-2 align-items-center",
        )
        components.append(result_type_selector)

        if include_subcomponents:
            if hasattr(self.scan_mode, "get_dash_components"):
                components.extend(
                    self.scan_mode.get_dash_components(include_subcomponents)
                )
            if hasattr(self.qua_inner_loop_action, "get_dash_components"):
                components.extend(
                    self.qua_inner_loop_action.get_dash_components(
                        include_subcomponents
                    )
                )

        return components

    def get_components(self) -> List[BaseUpdatableComponent]:
        components = super().get_components()
        components.extend(self.scan_mode.get_components())
        components.extend(self.qua_inner_loop_action.get_components())
        return components

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

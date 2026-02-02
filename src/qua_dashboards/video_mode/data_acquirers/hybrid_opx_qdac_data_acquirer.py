import logging
from typing import Dict, List, Union
import numpy as np

from qm.qua import (
    program,
    declare_stream,
    infinite_loop_,
    save,
    stream_processing,
    wait,
    for_,
    declare,
    align,
)
from qua_dashboards.video_mode.data_acquirers.opx_data_acquirer import OPXDataAcquirer
from qua_dashboards.video_mode.sweep_axis import VoltageSweepAxis
from quam_builder.architecture.quantum_dots.components import VirtualDCSet
from qua_dashboards.core import ModifiedFlags

logger = logging.getLogger(__name__)


__all__ = ["HybridOPXQDACDataAcquirer"]


class HybridOPXQDACDataAcquirer(OPXDataAcquirer): 
    """
    A class which combines a slow Qdac triggered DC list (Y axis) and a fast OPX sweep (X axis). For a defined voltages list, 
    this class will: 
        - Set up as many dc_lists in the Qdac as the number of outgoing physical ports. 
        - Trigger the Qdac on the outer loop, and sweep the OPX on the inner loop. 

    Requirements: 
        - Configure both a VirtualGateSet and VirtualDCSet with the same layered structure. This means that 
        the virtual gate names and virtual gate structures must match. The actual matrices themselves can be different. 
        - For each VirtualDCSet physical channel, the channel must be a VoltageGate object, with a configured QdacSpec. The 
        QdacSpec must define the qdac_output_port, and it must define an opx_trigger_out. 
        - Ensure that the outgoing digital trigger of the OPX is mapped to one of the qdac's external trigger inputs. The 
        external trigger input port can be specified here as the attribute qdac_ext_trigger_input_port. 
    
    Args: 
        dc_set: The VirtualDCSet object containing the VoltageGate channels, and their DC virtualisation. 
        qdac: The qdac object. This class assumes the use of qcodes_contrib_drivers.drivers.QDevil.QDAC2.Qdac2
        qdac_dwell_time_us: The amount of time the qdac should dwell on each point. 
    
    """
    
    def __init__(
        self, 
        dc_set: VirtualDCSet, 
        qdac,
        qdac_ext_trigger_input_port: int,
        qdac_dwell_time_us:int = 1000,
        *args, 
        **kwargs
    ) -> None:
        kwargs["_show_y_mode_selector"] = False
        super().__init__(*args, **kwargs)

        self.dc_set = dc_set
        self.qdac_dwell_time_us = qdac_dwell_time_us

        self.qdac = qdac
        self.qdac_ext_trigger_input_port = qdac_ext_trigger_input_port

        self._last_dc_list_y_offset = None

    def _find_physical_dc_lists(
        self, 
        qdac_sweep_axis: VoltageSweepAxis,
    ) -> Dict[str, Union[List, np.ndarray]]: 
        """Use the sweep axis abnd scab mode to yield a dictionary of physical dc_lists to use for the Qdac"""

        axis_name = qdac_sweep_axis.name
        axis_values = qdac_sweep_axis.sweep_values_with_offset

        _, y_idxs = self.scan_mode.get_idxs(x_points = 1, y_points = len(axis_values))
        ordered_axis_values = axis_values[y_idxs]

        dc_set = self.dc_set
        full_physical_dicts = {name: [] for name in dc_set.channels.keys()}

        for value in axis_values: 
            virtual_dict = {axis_name: float(value)}
            physical_dict = dc_set.resolve_voltages(virtual_dict)

            for physical_gate in dc_set.channels.keys(): 
                full_physical_dicts[physical_gate].append(physical_dict[physical_gate])

        return {name: np.array(l) for name, l in full_physical_dicts.items()}
    
    def _prepare_qdac_sweeps(
        self,
    ) -> None: 
        """
        Prepares the DC list attributes for the QDAC channel. This function assumes the use of the 
        Qdac2 driver from qcodes_contrib_drivers. This also assumes that the VoltageGate objects have 
        their QdacSpec objects configured with the qdac_output_port and opx_trigger_out. 
        """
        if not isinstance(self.y_axis, VoltageSweepAxis): 
            raise ValueError("Qdac Sweep Axis must be mode 'Voltage'")
        
        self._last_dc_list_y_offset = self.y_axis.offset_parameter.get_latest()
        physical_dc_lists = self._find_physical_dc_lists(self.y_axis)

        for name, voltages in physical_dc_lists.items():
            dc_list = self.qdac.channel(self.dc_set.channels[name].qdac_spec.qdac_output_port).dc_list(
                voltages = voltages, 
                dwell_s = self.qdac_dwell_time_us/1e6, 
                stepped = True,
            )
            dc_list.start_on_external(trigger = self.qdac_ext_trigger_input_port)

    def generate_qua_program(self): 
        """Generate the 1d X axis QUA sweep, and Y triggered by the Qdac."""

        x_qua_values = self.x_axis.qua_sweep_values
        y_points = 1 if self._is_1d else int(self.y_axis.points)

        self._prepare_qdac_sweeps()
        trigger_channel = None
        for ch in self.dc_set.channels.values(): 
            if ch.qdac_spec is not None and ch.qdac_spec.opx_trigger_out is not None: 
                trigger_channel = ch.qdac_spec.opx_trigger_out
                break
        if trigger_channel is None: 
            raise ValueError("No VoltageGate channel has a configured qdac_spec with opx_trigger_out.")
        
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

                y_idx = declare(int)
                with for_(y_idx, 0, y_idx < y_points, y_idx + 1): 
                    
                    # This should automatically the Qdac to step to the next set of voltages, since all the 
                    # dc lists are mapped to the same external trigger. 
                    trigger_channel.play("trigger")
                    align()

                    for x_qua_var, y_qua_var in self.scan_1d.scan(
                        x_vals=x_qua_values,
                        y_vals=None,
                        x_mode=self.x_mode,
                        y_mode=None,  # type: ignore
                    ):
                        measured_qua_values = self.inner_loop_action(
                            x_qua_var, None
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
                combined_qua_stream.buffer(self.buffer_frames).save("all_streams_combined")
                # combined_qua_stream.buffer(self.buffer_frames).save("all_streams_combined")

        self.qua_program = prog
        return prog
    
    def perform_actual_acquisition(self) -> np.ndarray:
            # Check if Y offset has changed since DC lists were prepared
            if (
                self._last_dc_list_y_offset is not None 
                and not self._is_1d
                and isinstance(self.y_axis, VoltageSweepAxis)
            ):
                current_y_offset = self.y_axis.offset_parameter.get_latest()
                if current_y_offset != self._last_dc_list_y_offset:
                    logger.info(
                        f"Y axis offset changed ({self._last_dc_list_y_offset} -> {current_y_offset}), "
                        "triggering DC list regeneration"
                    )
                    self._compilation_flags |= ModifiedFlags.PROGRAM_MODIFIED
            
            return super().perform_actual_acquisition()
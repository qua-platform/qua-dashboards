from qua_dashboards.core.base_updatable_component import ModifiedFlags
from qua_dashboards.utils.dash_utils import create_input_field
from dash import html
from qm.qua.lib import Cast, Math
from qualang_tools.units.units import unit
from qua_dashboards.video_mode.inner_loop_actions.inner_loop_action import (
    InnerLoopAction,
)
from quam_builder.architecture.quantum_dots.virtual_gates.virtual_gate_set import (
    VirtualGateSet
)
from qua_dashboards.utils.qua_types import QuaVariableFloat
from qm.qua import (
    align,
    assign,
    declare,
    demod,
    else_,
    fixed,
    if_,
    measure,
    play,
    ramp,
    ramp_to_zero,
    set_dc_offset,
    wait,
    amp
)

from typing import Any, Dict, List, Tuple

from qm.qua import (
    align,
    assign,
    declare,
    fixed,
    ramp_to_zero,
    wait,
)
import logging
logger = logging.getLogger(__name__)
from typing import List

class VirtualGateInnerLoopAction(InnerLoopAction):
    """
    Replacement for BasicInnerLoopAction which does a multi-axis scan based on 
    virtualisation matrices
    Args:
        VirtualGateSet with channels 
        x_element: The QUAM Channel object along the x-axis; defined also within the VirtualGateSet
        y_element: The QUAM Channel object along the y-axis; defined also within the VirtualGateSet
        readout_pulse: The QUAM Pulse object to measure.
        pre_measurement_delay: The optional delay before the measurement.
        ramp_rate: Th
        e ramp rate for voltage changes (optional).
        use_dBm: Whether to use dBm for amplitude (optional).
    """

    def __init__(
        self,
        gateset: VirtualGateSet,
        x_element,
        y_element,
        readout_pulse,
        pre_measurement_delay: float = 0.0,
        ramp_rate: float = 0.0,
        use_dBm=False,
    ):
        
        super().__init__()
        self.gateset = gateset
        self.x_elem = x_element
        self.y_elem = y_element
        self.readout_pulse = readout_pulse
        self.pre_measurement_delay = pre_measurement_delay
        self.ramp_rate = ramp_rate
        self.use_dBm = use_dBm


        self._last_x_voltage = None
        self._last_y_voltage = None
        self.reached_voltage = None

        self.sequence = gateset.new_sequence(track_integrated_voltage=False)
    def perform_ramp(self, levels):
        """Performs a ramp on a single channel
        Currently hard-coded to be 1us ramp time, but in future will convert to a QUA calculation
        """
        ramp_duration_ns = declare(int)
        assign(ramp_duration_ns, 1000)

        self.sequence.ramp_to_level(levels, 
                                    ramp_duration = ramp_duration_ns, 
                                    duration = 1000)
    

    def set_dc_offsets(self, x: QuaVariableFloat, y: QuaVariableFloat):

        x_name = getattr(self.x_elem, "name", self.x_elem)
        y_name = getattr(self.y_elem, "name", self.y_elem)
        levels = {x_name: x, y_name:y}

        if self.ramp_rate > 0:
            raise NotImplementedError('Please set ramp_rate to 0 and use a low OPX span')

        else: 
            phys = self.gateset.resolve_voltages(levels)
            for gate_name, qua_V in phys.items():
                set_dc_offset(gate_name, "single", qua_V)
            self.readout_pulse.channel.play("step", amplitude_scale=0, duration = 4)
                
    def __call__(
        self, x: QuaVariableFloat, y: QuaVariableFloat
    ) -> Tuple[QuaVariableFloat, QuaVariableFloat]:
        self.set_dc_offsets(x, y)
        align()

        pre_measurement_delay_cycles = int(self.pre_measurement_delay * 1e9 // 4)
        if pre_measurement_delay_cycles >= 4:
            wait(pre_measurement_delay_cycles)

        I, Q = self.readout_pulse.channel.measure(self.readout_pulse.id)
        align()

        return I, Q

    def initial_action(self):
        self._last_x_voltage = declare(fixed, 0.0)
        self._last_y_voltage = declare(fixed, 0.0)
        self.set_dc_offsets(0, 0)
        align()

    def final_action(self):
        if self.ramp_rate > 0:
            if getattr(self.x_elem, "sticky", None) is None:
                raise RuntimeError("Ramp rate is not supported for non-sticky elements")
            if getattr(self.y_elem, "sticky", None) is None:
                raise RuntimeError("Ramp rate is not supported for non-sticky elements")

            self.set_dc_offsets(0.0,0.0)
        else:
            self.set_dc_offsets(0.0,0.0)
            
        align()

    def get_dash_components(self, include_subcomponents: bool = True) -> List[html.Div]:
        components = super().get_dash_components(include_subcomponents)

        additional_components = [
            create_input_field(
                id=self._get_id("readout_frequency"),
                label="Readout frequency",
                value=self.readout_pulse.channel.intermediate_frequency,
                input_style={"width": "200px"},
                units="Hz",
            ),
            create_input_field(
                id=self._get_id("readout_duration"),
                label="Readout duration",
                value=self.readout_pulse.length,
                units="ns",
                step=10,
            ),
        ]

        if self.use_dBm:
            additional_components.append(
                create_input_field(
                    id=self._get_id("readout_power"),
                    label="Readout power",
                    value=unit.volts2dBm(self.readout_pulse.amplitude),
                    units="dBm",
                ),
            )
        else:
            additional_components.append(
                create_input_field(
                    id=self._get_id("readout_amplitude"),
                    label="Readout amplitude",
                    value=self.readout_pulse.amplitude,
                    units="V",
                ),
            )

        components.append(html.Div(additional_components))

        return components

    def update_parameters(self, parameters: Dict[str, Dict[str, Any]]) -> ModifiedFlags:
        """Update the data acquirer's attributes based on the input values."""
        if self.component_id not in parameters:
            return ModifiedFlags.NONE

        params = parameters[self.component_id]

        flags = ModifiedFlags.NONE
        if (
            self.readout_pulse.channel.intermediate_frequency
            != params["readout_frequency"]
        ):
            self.readout_pulse.channel.intermediate_frequency = params[
                "readout_frequency"
            ]
            flags |= ModifiedFlags.PARAMETERS_MODIFIED
            flags |= ModifiedFlags.PROGRAM_MODIFIED
            flags |= ModifiedFlags.CONFIG_MODIFIED

        if self.readout_pulse.length != params["readout_duration"]:
            self.readout_pulse.length = params["readout_duration"]
            flags |= ModifiedFlags.PARAMETERS_MODIFIED
            flags |= ModifiedFlags.PROGRAM_MODIFIED
            flags |= ModifiedFlags.CONFIG_MODIFIED

        if self.use_dBm:
            if unit.volts2dBm(self.readout_pulse.amplitude) != params["readout_power"]:
                self.readout_pulse.amplitude = unit.dBm2volts(params["readout_power"])
                flags |= ModifiedFlags.PARAMETERS_MODIFIED
                flags |= ModifiedFlags.PROGRAM_MODIFIED
                flags |= ModifiedFlags.CONFIG_MODIFIED
        else:
            if self.readout_pulse.amplitude != params["readout_amplitude"]:
                self.readout_pulse.amplitude = params["readout_amplitude"]
                flags |= ModifiedFlags.PARAMETERS_MODIFIED
                flags |= ModifiedFlags.PROGRAM_MODIFIED
                flags |= ModifiedFlags.CONFIG_MODIFIED

        return flags

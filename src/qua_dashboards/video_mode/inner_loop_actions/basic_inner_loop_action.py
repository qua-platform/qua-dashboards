from qua_dashboards.core.base_updatable_component import ModifiedFlags
from qua_dashboards.utils.dash_utils import create_input_field
from dash import html
from qualang_tools.units.units import unit
from qua_dashboards.video_mode.inner_loop_actions.inner_loop_action import (
    InnerLoopAction,
)
from qua_dashboards.utils.qua_types import QuaVariableFloat
from quam_builder.architecture.quantum_dots import GateSet

from qm import qua


from typing import Any, Dict, List, Tuple


class BasicInnerLoopAction(InnerLoopAction):
    """Inner loop action for the video mode: set voltages and measure using GateSet.

    Args:
        gate_set: The GateSet object containing voltage channels.
        readout_pulse: The QUAM Pulse object to measure.
        x_axis_name: Name of the X-axis channel in the GateSet.
        y_axis_name: Name of the Y-axis channel in the GateSet.
        pre_measurement_delay: The optional delay before the measurement in ns..
        track_integrated_voltage: Whether to track integrated voltage (optional).
        use_dBm: Whether to use dBm for amplitude (optional).
    """

    def __init__(
        self,
        gate_set: GateSet,
        readout_pulse,
        x_axis_name: str,
        y_axis_name: str,
        pre_measurement_delay: int = 0,
        track_integrated_voltage: bool = False,
        use_dBm=False,
    ):
        super().__init__()
        self.gate_set = gate_set
        self.readout_pulse = readout_pulse
        self.x_axis_name = x_axis_name
        self.y_axis_name = y_axis_name
        self.pre_measurement_delay = pre_measurement_delay
        self.track_integrated_voltage = track_integrated_voltage
        self.use_dBm = use_dBm
        self.voltage_sequence = None
        self.ramp_duration = 16

    def __call__(
        self, x: QuaVariableFloat, y: QuaVariableFloat
    ) -> Tuple[QuaVariableFloat, QuaVariableFloat]:
        # Map sweep values to named channels via axis names
        if y is None: 
            levels = {self.x_axis_name: x}
            last_resolved_voltages = self.gate_set.resolve_voltages({self.x_axis_name: self.last_v_x})
        else:
            levels = {self.x_axis_name: x, self.y_axis_name: y}
            last_resolved_voltages = self.gate_set.resolve_voltages({self.x_axis_name: self.last_v_x, self.y_axis_name: self.last_v_y})
        resolved_voltages = self.gate_set.resolve_voltages(levels)

        duration = self.readout_pulse.length
        if self.pre_measurement_delay > 0:
            duration += self.pre_measurement_delay
        
        ramp_time = qua.fixed(1/self.ramp_duration/4)
        for ch_string in self.gate_set.channels.keys():
            applied_voltage = (resolved_voltages[ch_string] - last_resolved_voltages[ch_string])
            qua.assign(self.slope, applied_voltage)
            qua.assign(self.slope, (self.slope >> 12) << 12)
            qua.assign(self.slope, self.slope * ramp_time)
            qua.play(qua.ramp(self.slope), ch_string, duration = self.ramp_duration)

        if self.pre_measurement_delay > 0:
            self.readout_pulse.channel.wait(self.pre_measurement_delay)

        qua.align()
        I, Q = self.readout_pulse.channel.measure(self.readout_pulse.id)
        qua.align()

        qua.assign(self.last_v_x, x)
        if y is not None: 
            qua.assign(self.last_v_y, y)

        qua.assign(self.last_v_x, (self.last_v_x >> 12) << 12)
        if y is not None:
            qua.assign(self.last_v_y, (self.last_v_y >> 12) << 12)
        qua.ramp_to_zero(self.readout_pulse.channel.name, duration = self.ramp_duration)
        qua.align()

        return I, Q

    def initial_action(self):
        # Create VoltageSequence within QUA program context
        self.voltage_sequence = self.gate_set.new_sequence(
            track_integrated_voltage=self.track_integrated_voltage
        )
        self.slope = qua.declare(qua.fixed)
        self.last_v_x = qua.declare(qua.fixed)
        self.last_v_y = qua.declare(qua.fixed)
        qua.assign(self.last_v_x, 0)
        qua.assign(self.last_v_y, 0)

        # Initialize all channels to zero
        self.voltage_sequence.ramp_to_zero()

    def final_action(self):
        # Use GateSet's built-in ramp to zero
        self.voltage_sequence.ramp_to_zero()

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
        if "readout_frequency" in params and (
                self.readout_pulse.channel.intermediate_frequency != params["readout_frequency"]
            ):
            self.readout_pulse.channel.intermediate_frequency = params[
                "readout_frequency"
            ]
            flags |= ModifiedFlags.PARAMETERS_MODIFIED
            flags |= ModifiedFlags.PROGRAM_MODIFIED
            flags |= ModifiedFlags.CONFIG_MODIFIED

        if "readout_duration" in params and self.readout_pulse.length != params["readout_duration"]:
            self.readout_pulse.length = params["readout_duration"]
            flags |= ModifiedFlags.PARAMETERS_MODIFIED
            flags |= ModifiedFlags.PROGRAM_MODIFIED
            flags |= ModifiedFlags.CONFIG_MODIFIED

        if self.use_dBm:
            if "readout_power" in params and unit.volts2dBm(self.readout_pulse.amplitude) != params["readout_power"]:
                self.readout_pulse.amplitude = unit.dBm2volts(params["readout_power"])
                flags |= ModifiedFlags.PARAMETERS_MODIFIED
                flags |= ModifiedFlags.PROGRAM_MODIFIED
                flags |= ModifiedFlags.CONFIG_MODIFIED
        else:
            if "readout_amplitude" in params and self.readout_pulse.amplitude != params["readout_amplitude"]:
                self.readout_pulse.amplitude = params["readout_amplitude"]
                flags |= ModifiedFlags.PARAMETERS_MODIFIED
                flags |= ModifiedFlags.PROGRAM_MODIFIED
                flags |= ModifiedFlags.CONFIG_MODIFIED

        return flags
from qua_dashboards.core.base_updatable_component import ModifiedFlags
from qua_dashboards.utils.dash_utils import create_input_field
from dash import html
from qualang_tools.units.units import unit
from qua_dashboards.video_mode.inner_loop_actions.inner_loop_action import (
    InnerLoopAction,
)
from qua_dashboards.utils.qua_types import QuaVariableFloat
from quam_builder.architecture.quantum_dots.voltage_sequence import GateSet

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
        self.selected_readout_channels = []
    @property
    def selected_readout_names(self) -> list[str]:
        return [ch.name for ch in getattr(self, "selected_readout_channels", [])]
    def _pulse_for(self, ch):
        return ch.operations["readout"]

    def __call__(
        self, x: QuaVariableFloat, y: QuaVariableFloat
    ) -> Tuple[QuaVariableFloat, QuaVariableFloat]:
        # Map sweep values to named channels via axis names
        if y is None: 
            levels = {self.x_axis_name: x}
        else:
            levels = {self.x_axis_name: x, self.y_axis_name: y}

        duration = self.readout_pulse.length
        if self.pre_measurement_delay > 0:
            duration += self.pre_measurement_delay

        self.voltage_sequence.step_to_voltages(voltages=levels, duration=duration)

        if self.pre_measurement_delay > 0:
            self.readout_pulse.channel.wait(self.pre_measurement_delay)
        qua.align()
        
        result = []
        for channel in self.selected_readout_channels:
            I, Q = channel.measure("readout")
            result.extend([I, Q])

        qua.align()
        self.voltage_sequence.ramp_to_zero()
        qua.wait(2000)

        return result

    def initial_action(self):
        # Create VoltageSequence within QUA program context
        self.voltage_sequence = self.gate_set.new_sequence(
            track_integrated_voltage=self.track_integrated_voltage
        )
        # Initialize all channels to zero
        self.voltage_sequence.step_to_voltages({}, duration=16)

    def final_action(self):
        # Use GateSet's built-in ramp to zero
        self.voltage_sequence.ramp_to_zero()

    def build_readout_controls(self, channels = None):
        channels = getattr(self, "selected_readout_channels", []) or [self.readout_pulse.channel]
        rows = []
        for ch in channels: 
            pulse = self._pulse_for(ch)
            name = ch.name

            additional_components = [
                create_input_field(
                    id=self._get_id(f"{name}-readout_frequency"),
                    label=f"{name} frequency",
                    value=pulse.channel.intermediate_frequency,
                    input_style={"width": "200px"},
                    units="Hz",
                ),
                create_input_field(
                    id=self._get_id(f"{name}-readout_duration"),
                    label=f"{name} duration",
                    value=pulse.length,
                    units="ns",
                    step=10,
                ),
            ]
            if self.use_dBm:
                additional_components.append(
                    create_input_field(
                        id=self._get_id(f"{name}-readout_power"),
                        label=f"{name} power",
                        value=unit.volts2dBm(pulse.amplitude),
                        units="dBm",
                    ),
                )
            else:
                additional_components.append(
                    create_input_field(
                        id=self._get_id(f"{name}-readout_amplitude"),
                        label=f"{name} amplitude",
                        value=pulse.amplitude,
                        units="V",
                    ),
                )
            rows.append(html.Div(additional_components, id = f"{self.component_id}-ro-params-{name}"))
        return rows

    def get_dash_components(self, include_subcomponents: bool = True) -> List[html.Div]:
        components = super().get_dash_components(include_subcomponents)

        components.append(html.Div(
            id=f"{self.component_id}-readout-params-container",
            children=self.build_readout_controls(),
        ))
        return components

    def update_parameters(self, parameters: Dict[str, Dict[str, Any]]) -> ModifiedFlags:
        """Update the data acquirer's attributes based on the input values."""
        if self.component_id not in parameters:
            return ModifiedFlags.NONE

        params = parameters[self.component_id]

        flags = ModifiedFlags.NONE

        channels = getattr(self, "selected_readout_channels", []) or [self.readout_pulse.channel]
        for ch in channels: 
            pulse = self._pulse_for(ch)
            name = ch.name

            f_key = f"{name}-readout_frequency"
            d_key = f"{name}-readout_duration"
            p_key = f"{name}-readout_power"
            v_key = f"{name}-readout_amplitude"

            freq = params.get(f_key, params.get("readout_frequency"))
            dur  = params.get(d_key,  params.get("readout_duration"))
            amp_dbm = params.get(p_key, params.get("readout_power"))
            amp_v = params.get(v_key, params.get("readout_amplitude"))

            if freq is not None and (
                pulse.channel.intermediate_frequency
                != freq
            ):
                pulse.channel.intermediate_frequency = freq
                flags |= ModifiedFlags.PARAMETERS_MODIFIED
                flags |= ModifiedFlags.PROGRAM_MODIFIED
                flags |= ModifiedFlags.CONFIG_MODIFIED

            if dur is not None and pulse.length != dur:
                pulse.length = dur
                flags |= ModifiedFlags.PARAMETERS_MODIFIED
                flags |= ModifiedFlags.PROGRAM_MODIFIED
                flags |= ModifiedFlags.CONFIG_MODIFIED

            if self.use_dBm:
                if amp_dbm is not None and unit.volts2dBm(pulse.amplitude) != amp_dbm:
                    pulse.amplitude = unit.dBm2volts(amp_dbm)
                    flags |= ModifiedFlags.PARAMETERS_MODIFIED
                    flags |= ModifiedFlags.PROGRAM_MODIFIED
                    flags |= ModifiedFlags.CONFIG_MODIFIED
            else:
                if amp_v is not None and pulse.amplitude != amp_v:
                    pulse.amplitude = amp_v
                    flags |= ModifiedFlags.PARAMETERS_MODIFIED
                    flags |= ModifiedFlags.PROGRAM_MODIFIED
                    flags |= ModifiedFlags.CONFIG_MODIFIED

        return flags
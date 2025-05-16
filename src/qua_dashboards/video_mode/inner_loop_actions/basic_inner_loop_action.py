from qua_dashboards.core.base_updatable_component import ModifiedFlags
from qua_dashboards.utils.dash_utils import create_input_field
from dash import html
from qm.qua.lib import Cast, Math
from qualang_tools.units.units import unit
from qua_dashboards.video_mode.inner_loop_actions.inner_loop_action import (
    InnerLoopAction,
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
)


from typing import Any, Dict, List, Tuple


class BasicInnerLoopAction(InnerLoopAction):
    """Inner loop action for the video mode: set voltages and measure using QUAM objects.

    Args:
        x_element: The QUAM Channel object along the x-axis.
        y_element: The QUAM Channel object along the y-axis.
        readout_pulse: The QUAM Pulse object to measure.
        pre_measurement_delay: The optional delay before the measurement.
        ramp_rate: The ramp rate for voltage changes (optional).
        use_dBm: Whether to use dBm for amplitude (optional).
    """

    def __init__(
        self,
        x_element,
        y_element,
        readout_pulse,
        pre_measurement_delay: float = 0.0,
        ramp_rate: float = 0.0,
        use_dBm=False,
    ):
        super().__init__()
        self.x_elem = x_element
        self.y_elem = y_element
        self.readout_pulse = readout_pulse
        self.pre_measurement_delay = pre_measurement_delay
        self.ramp_rate = ramp_rate
        self.use_dBm = use_dBm

        self._last_x_voltage = None
        self._last_y_voltage = None
        self.reached_voltage = None

    def perform_ramp(self, element, previous_voltage, new_voltage):
        ramp_cycles_ns_V = declare(int, int(1e9 / self.ramp_rate / 4))
        qua_ramp = declare(fixed, self.ramp_rate / 1e9)
        dV = declare(fixed)
        duration = declare(int)
        self.reached_voltage = declare(fixed)
        assign(dV, new_voltage - previous_voltage)
        assign(duration, Math.abs(Cast.mul_int_by_fixed(ramp_cycles_ns_V, dV)))

        with if_(duration > 4):
            with if_(dV > 0):
                assign(
                    self.reached_voltage,
                    previous_voltage + Cast.mul_fixed_by_int(qua_ramp, duration << 2),
                )
                play(ramp(self.ramp_rate / 1e9), element.name, duration=duration)
            with else_():
                assign(
                    self.reached_voltage,
                    previous_voltage - Cast.mul_fixed_by_int(qua_ramp, duration << 2),
                )
                play(ramp(-self.ramp_rate / 1e9), element.name, duration=duration)
        with else_():
            ramp_rate = dV * (1 / 16e-9)
            play(ramp(ramp_rate), element.name, duration=4)
            # element.play("step", amplitude_scale=dV << 2)
            assign(self.reached_voltage, new_voltage)

    def set_dc_offsets(self, x: QuaVariableFloat, y: QuaVariableFloat):
        if self.ramp_rate > 0:
            if getattr(self.x_elem, "sticky", None) is None:
                raise RuntimeError("Ramp rate is not supported for non-sticky elements")
            if getattr(self.y_elem, "sticky", None) is None:
                raise RuntimeError("Ramp rate is not supported for non-sticky elements")

            self.perform_ramp(self.x_elem, self._last_x_voltage, x)
            assign(self._last_x_voltage, self.reached_voltage)
            self.perform_ramp(self.y_elem, self._last_y_voltage, y)
            assign(self._last_y_voltage, self.reached_voltage)
        else:
            self.x_elem.set_dc_offset(x)
            self.y_elem.set_dc_offset(y)

            assign(self._last_x_voltage, x)
            assign(self._last_y_voltage, y)

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

            ramp_to_zero(self.x_elem.name)
            ramp_to_zero(self.y_elem.name)
            assign(self._last_x_voltage, 0.0)
            assign(self._last_y_voltage, 0.0)
        else:
            self.set_dc_offsets(0, 0)
        align()

    def get_dash_components(self, include_subcomponents: bool = True) -> List[html.Div]:
        components = super().get_dash_components(include_subcomponents)

        additional_components = [
            create_input_field(
                id={"type": self.component_id, "index": "readout_frequency"},
                label="Readout frequency",
                value=self.readout_pulse.channel.intermediate_frequency,
                units="Hz",
                step=20e3,
            ),
            create_input_field(
                id={"type": self.component_id, "index": "readout_duration"},
                label="Readout duration",
                value=self.readout_pulse.length,
                units="ns",
                input_style={"width": "200px"},
                step=10,
            ),
        ]

        if self.use_dBm:
            additional_components.append(
                create_input_field(
                    id={"type": self.component_id, "index": "readout_power"},
                    label="Readout power",
                    value=unit.volts2dBm(self.readout_pulse.amplitude),
                    units="dBm",
                ),
            )
        else:
            additional_components.append(
                create_input_field(
                    id={"type": self.component_id, "index": "readout_amplitude"},
                    label="Readout amplitude",
                    value=self.readout_pulse.amplitude,
                    units="V",
                ),
            )

        components.append(html.Div(additional_components))

        return components

    def update_parameters(self, parameters: Dict[str, Dict[str, Any]]) -> ModifiedFlags:
        """Update the data acquirer's attributes based on the input values."""
        try:
            params = parameters[self.component_id]
        except KeyError:
            print(f"Inner loop action parameters: {list(parameters.keys())}")
            raise

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

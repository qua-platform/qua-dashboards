from qua_dashboards.core.base_updatable_component import ModifiedFlags
from qua_dashboards.utils.dash_utils import create_input_field
import dash_bootstrap_components as dbc
from dash import html
from qualang_tools.units.units import unit
from qua_dashboards.video_mode.inner_loop_actions.inner_loop_action import (
    InnerLoopAction,
)
from qua_dashboards.utils.qua_types import QuaVariableFloat
from quam_builder.architecture.quantum_dots.components import GateSet

from qm import qua


from typing import Any, Dict, List, Tuple, Optional


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
        x_axis_name: str,
        y_axis_name: str,
        pre_measurement_delay: int = 0,
        track_integrated_voltage: bool = False,
        use_dBm=False,
    ):
        super().__init__()
        self.gate_set = gate_set
        self.x_axis_name = x_axis_name
        self.y_axis_name = y_axis_name
        self.pre_measurement_delay = pre_measurement_delay
        self.track_integrated_voltage = track_integrated_voltage
        self.use_dBm = use_dBm
        self.voltage_sequence = None
        self.ramp_duration = 16
        self.selected_readout_channels = []
        self.readout_pulse_mapping = {}
        self.x_mode = "Voltage"
        self.y_mode = "Voltage"

    def _pulse_for(self, ch):
        if ch.name not in self.readout_pulse_mapping.keys():
            raise ValueError("Channel not in registered readout pulses")
        else:
            return self.readout_pulse_mapping[ch.name]
        

    @staticmethod
    def element_finder(name: str) -> Tuple[str, str]:
        if name is None: 
            return None
        if "frequency" in name.lower():
            return name.replace("_frequency", "")
        if "drive" in name.lower():
            return name.replace("_drive", "")
        return name

    @staticmethod
    def _voltage_contrib(axis_name: Optional[str], mode: str, val: QuaVariableFloat):
        if axis_name is None or mode != "Voltage": 
            return {}
        return {axis_name: val}
    
    def _freq_contrib(self, axis_name: Optional[str], mode: str, val: QuaVariableFloat):
        elem = self.element_finder(axis_name)
        if axis_name is None or elem is None or mode != "Frequency":
            return None
        return (elem, val)
    
    def _amp_contrib(self, axis_name: Optional[str], mode: str, val: QuaVariableFloat, dbm:bool = False):
        elem = self.element_finder(axis_name)
        if axis_name is None or elem is None or mode != "Drive":
            return {}
        amp = None
        for ch in self.selected_readout_channels:
            if ch.name == elem: 
                amp = self._pulse_for(ch).amplitude
                break 
        if amp is None: 
            return {}
        
        qua.assign(self.helper_var, float(min(max(amp, 0.0), 1.999)))
        qua.assign(self.scale_var, val / self.helper_var)

        return {elem: self.scale_var}


    def __call__(
                    self, x: QuaVariableFloat, y: QuaVariableFloat
    ) -> Tuple[QuaVariableFloat, QuaVariableFloat]:

        v_x = self._voltage_contrib(self.x_axis_name, self.x_mode, (x>>12)<<12)
        v_y = self._voltage_contrib(self.y_axis_name, self.y_mode, (y>>12)<<12) if y is not None else {}

        f_x = self._freq_contrib(self.x_axis_name, self.x_mode, x)
        f_y = self._freq_contrib(self.y_axis_name, self.y_mode, y) if y is not None else None

        a_x = self._amp_contrib(self.x_axis_name, self.x_mode, x)
        a_y = self._amp_contrib(self.y_axis_name, self.y_mode, y) if y is not None else {}

        levels: Dict[str, QuaVariableFloat] = {**v_x, **v_y}
        freq_updates = [u for u in (f_x, f_y) if u is not None]
        amp_scale: Dict[str, QuaVariableFloat] = {**a_x, **a_y}
 
        if levels: 
            last_voltages: Dict[str, QuaVariableFloat] = {}
            if self.x_mode == "Voltage": 
                last_voltages[self.x_axis_name] = self.last_v_x
            if self.y_mode == "Voltage" and y is not None: 
                last_voltages[self.y_axis_name] = self.last_v_y
            last_resolved_voltages = self.gate_set.resolve_voltages(last_voltages)
            resolved_voltages = self.gate_set.resolve_voltages(levels)

            ramp_time = qua.fixed(1 / self.ramp_duration / 4)
            for ch_string in self.gate_set.channels.keys():
                qua.assign(self.loop_current, resolved_voltages[ch_string])
                qua.assign(self.loop_current, (self.loop_current>>12)<<12)
                qua.assign(self.loop_past, last_resolved_voltages[ch_string])
                qua.assign(self.loop_past, (self.loop_past>>12)<<12)
                applied_voltage = (
                    self.loop_current - self.loop_past
                )
                qua.assign(self.slope, applied_voltage)
                qua.assign(self.slope, (self.slope))
                qua.assign(self.slope, self.slope * ramp_time)
                qua.play(qua.ramp(self.slope), ch_string, duration=self.ramp_duration)

            if self.x_mode == "Voltage": 
                qua.assign(self.last_v_x, (x>>12)<<12)
            if self.y_mode == "Voltage" and y is not None: 
                qua.assign(self.last_v_y, (y>>12)<<12)

        qua.assign(self.last_v_x, x)
        qua.assign(self.last_v_x, (self.last_v_x >> 12) << 12)
        if y is not None: 
            qua.assign(self.last_v_y, y)
            qua.assign(self.last_v_y, (self.last_v_y >> 12) << 12)

        qua.align()
        duration = max(
                    self._pulse_for(op).length for op in self.selected_readout_channels
                )
        if self.pre_measurement_delay > 0:
            duration += self.pre_measurement_delay
            qua.wait(duration)

        qua.align()

        for elem, freq in freq_updates:
            qua.update_frequency(elem, freq)
        
        result = []
        for channel in self.selected_readout_channels:
            elem = channel.name
            scale = 1
            if elem in amp_scale:
                scale = amp_scale[elem]
            I, Q = channel.measure(self._pulse_for(channel).id, amplitude_scale = scale)
            result.extend([I, Q])
        qua.align()

        for channel in self.selected_readout_channels:
            qua.ramp_to_zero(channel.name, duration=self.ramp_duration)
        qua.align()
        qua.wait(2000)

        return result
    
    def initial_action(self):
        # Create VoltageSequence within QUA program context
        self.voltage_sequence = self.gate_set.new_sequence(
            track_integrated_voltage=self.track_integrated_voltage
        )

        self.slope = qua.declare(qua.fixed)
        self.last_v_x = qua.declare(qua.fixed, value = 0)
        self.last_v_y = qua.declare(qua.fixed, value = 0)
        self.helper_var = qua.declare(qua.fixed)
        self.scale_var = qua.declare(qua.fixed, value = 1)
        self.loop_current= qua.declare(qua.fixed)
        self.loop_past = qua.declare(qua.fixed)

        # Initialize all channels to zero
        self.voltage_sequence.ramp_to_zero()

    def final_action(self):
        # Use GateSet's built-in ramp to zero
        self.voltage_sequence.ramp_to_zero()

    def build_readout_controls(self, channels=None):
        """
        Build the controls per selected readout channel
        """
        channels = self.selected_readout_channels
        rows = []
        for ch in channels:
            pulse = self._pulse_for(ch)
            name = ch.name

            additional_components = [
                create_input_field(
                    id=self._get_id(f"{name}-readout_frequency"),
                    label="Frequency",
                    value=pulse.channel.intermediate_frequency,
                    input_style={"width": "150px"},
                    units="Hz",
                ),
                create_input_field(
                    id=self._get_id(f"{name}-readout_duration"),
                    label="Duration",
                    value=pulse.length,
                    units="ns",
                    step=10,
                ),
            ]
            if self.use_dBm:
                additional_components.append(
                    create_input_field(
                        id=self._get_id(f"{name}-readout_power"),
                        label="Power",
                        value=unit.volts2dBm(pulse.amplitude),
                        units="dBm",
                    ),
                )
            else:
                additional_components.append(
                    create_input_field(
                        id=self._get_id(f"{name}-readout_amplitude"),
                        label="Amplitude",
                        value=pulse.amplitude,
                        units="V",
                    ),
                )

            rows.append(
                dbc.Card(
                    [
                        dbc.CardHeader(html.H6(f"{name} Parameters")),
                        dbc.CardBody(
                            html.Div(
                                additional_components,
                                id=f"{self.component_id}-ro-params-{name}",
                            ),
                            className="text-light",
                        ),
                    ],
                    color="dark",
                    inverse=True,
                    className="h-100 tab-card-dark",
                    style={
                        "outline": "2px solid #fff",
                    },
                ),
            )
        return rows

    def get_dash_components(self, include_subcomponents: bool = True) -> List[html.Div]:
        components = super().get_dash_components(include_subcomponents)

        components.append(
            html.Div(
                id=f"{self.component_id}-readout-params-container",
                children=self.build_readout_controls(),
            )
        )
        return components

    def update_parameters(self, parameters: Dict[str, Dict[str, Any]]) -> ModifiedFlags:
        """Update the data acquirer's attributes based on the input values."""
        if self.component_id not in parameters:
            return ModifiedFlags.NONE

        params = parameters[self.component_id]

        flags = ModifiedFlags.NONE

        channels = self.selected_readout_channels
        for ch in channels:
            pulse = self._pulse_for(ch)
            name = ch.name

            f_key = f"{name}-readout_frequency"
            d_key = f"{name}-readout_duration"
            p_key = f"{name}-readout_power"
            v_key = f"{name}-readout_amplitude"

            freq = params.get(f_key, params.get("readout_frequency"))
            dur = params.get(d_key, params.get("readout_duration"))
            amp_dbm = params.get(p_key, params.get("readout_power"))
            amp_v = params.get(v_key, params.get("readout_amplitude"))

            if freq is not None and (pulse.channel.intermediate_frequency != freq):
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

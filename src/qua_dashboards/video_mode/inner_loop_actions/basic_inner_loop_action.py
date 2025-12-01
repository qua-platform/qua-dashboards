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
from qua_dashboards.video_mode.sweep_axis import BaseSweepAxis, VoltageSweepAxis

from qm import qua

from typing import Any, Dict, List, Tuple, Optional


class BasicInnerLoopAction(InnerLoopAction):
    """Inner loop action for the video mode: set voltages and measure using GateSet.

    Args:
        gate_set: The GateSet object containing voltage channels.
        x_axis: X SweepAxis object.
        y_axis: Y SweepAxis object.
        pre_measurement_delay: The optional delay before the measurement in ns..
        track_integrated_voltage: Whether to track integrated voltage (optional).
        use_dBm: Whether to use dBm for amplitude (optional).
    """

    def __init__(
        self,
        gate_set: GateSet,
        x_axis: BaseSweepAxis,
        y_axis: BaseSweepAxis,
        pre_measurement_delay: int = 0,
        track_integrated_voltage: bool = True,
        use_dBm=False,
        apply_compensation: bool = True
    ):
        super().__init__()
        self.gate_set = gate_set
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.pre_measurement_delay = pre_measurement_delay
        self.track_integrated_voltage = track_integrated_voltage
        self.use_dBm = use_dBm
        self.voltage_sequence = None
        self.ramp_duration = 16
        self.selected_readout_channels = []
        self.readout_pulse_mapping = {}
        self.x_mode = "Voltage"
        self.y_mode = "Voltage"
        self.apply_compensation = apply_compensation

    def _pulse_for(self, ch):
        if ch.name not in self.readout_pulse_mapping.keys():
            raise ValueError("Channel not in registered readout pulses")
        else:
            return self.readout_pulse_mapping[ch.name]
        
    @staticmethod
    def pre_loop_action(inner_loop_self) -> None: 
        """
        Called before the X and Y axes have gone to the sweep coordinates, to perform a QUA snippet.
        """
        pass
        
    @staticmethod
    def loop_action(inner_loop_self) -> None: 
        """
        Called after the X and Y axes have gone to the sweep coordinates, to perform a QUA snippet.
        """
        pass

    def __call__(
        self, x: QuaVariableFloat, y: QuaVariableFloat
    ) -> List[QuaVariableFloat]:
    
        # Initial action
        self.pre_loop_action(self)

        # Apply functions
        # For FrequencySweepAxis, applies update_frequency and returns empty dict.
        # For VoltageSweepAxis, supplies a dict component to be fed into voltage_sequence
        # For AmplitudeSweepAxis, calculates amplitude scale and returns the scale as a dict component {element: scale}.
        # Add functionalities to existing BaseSweepAxis objects in their respective apply commands
        # For new SweepAxis objects (e.g. QubitSweepAxis), simply have an apply command that returns an empty dict.

        x_apply = self.x_axis.apply(x)
        y_apply = self.y_axis.apply(y) if (self.y_axis and y is not None) else None

        voltage_coordinates = {
            **x_apply.get("voltage", {}),
            **(y_apply.get("voltage", {}) or {}),
        } if y_apply is not None else {**x_apply.get("voltage", {})}
        self.voltage_sequence.step_to_voltages(voltage_coordinates, duration = 1000)

        self.loop_action(self)

        amplitude_scales = {
            **x_apply.get("amplitude_scales", {}),
            **(y_apply.get("amplitude_scales", {}) or {}),
        } if y_apply is not None else {**x_apply.get("amplitude_scales", {})}

        qua.align()
        duration = max(
            self._pulse_for(op).length for op in self.selected_readout_channels
        )
        if self.pre_measurement_delay > 0:
            duration += self.pre_measurement_delay
            qua.wait(duration // 4)
        qua.align()
        result = []
        for channel in self.selected_readout_channels:
            elem = channel.name
            scale = 1
            if elem in amplitude_scales:
                scale = amplitude_scales.get(elem, 1)
            I, Q = channel.measure(self._pulse_for(channel).id, amplitude_scale=scale)
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
            track_integrated_voltage=self.track_integrated_voltage, enforce_qua_calcs=True
        )
        self.x_axis.declare_vars()
        self.y_axis.declare_vars()

        # Initialize all channels to zero
        self.voltage_sequence.ramp_to_zero()

    def final_action(self):
        # Apply conditional compensation pulse
        if self.apply_compensation: 
            self.voltage_sequence.apply_compensation_pulse()

        # Use GateSet's built-in ramp to zero, in-case it is not handled by the compensation pulse
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
                    step=1,
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

            if dur not in (None, ""):
                dur = int(dur)
                if dur % 4 != 0:
                    message = (
                        f"{name}: readout duration must be multiple of 4 (got {dur} ns)"
                    )
                    raise ValueError(message)
                if pulse.length != dur:
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

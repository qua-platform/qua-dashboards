"""
Generic physical voltage-control component.

Physical voltages are controlled through QCoDeS-like parameters.

If a physical gate also exists in a VirtualDCSet, the voltage is set
through the VirtualDCSet so that its internal voltage cache remains
synchronized.

QDAC-only gates are controlled directly through their QCoDeS parameters.
"""

import logging
from typing import Mapping, Optional

import dash
from dash import ALL, MATCH, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from qua_dashboards.core import BaseComponent, ParameterProtocol

from .physical_voltage_control_row import (
    PhysicalVoltageControlRow,
    format_voltage_fixed,
)


logger = logging.getLogger(__name__)


class PhysicalVoltageControlComponent(BaseComponent):
    def __init__(
        self,
        component_id: str,
        voltage_parameters: Mapping[str, ParameterProtocol],
        dc_set=None,
        step_size: float = 1e-3,
        update_interval_ms: int = 1000,
        channel_labels: Optional[Mapping[str, str]] = None,
    ):
        super().__init__(component_id=component_id)

        self.voltage_parameters = dict(voltage_parameters)
        self.dc_set = dc_set
        self.step_size = step_size
        self.update_interval_ms = update_interval_ms
        self.channel_labels = dict(channel_labels or {})

        # For videomode
        self.voltage_parameters_by_name = self.voltage_parameters

        # Keep all internal voltage values in volts.
        self.display_unit = "V"

        # Dashboard panel size
        self.layout_columns = 12
        self.layout_rows = max(4, len(self.voltage_parameters))

        self.rows = {
            name: PhysicalVoltageControlRow(
                component_id=self.component_id,
                param=param,
                label=name,
                channel_label=self.channel_labels.get(name),
                display_unit=self.display_unit,
            )
            for name, param in self.voltage_parameters.items()
        }

    def _to_display_voltage(self, value: float) -> float:
        if self.display_unit == "mV":
            return value * 1e3
        return value

    def _from_display_voltage(self, value: float) -> float:
        if self.display_unit == "mV":
            return value * 1e-3
        return value

    def _display_precision(self) -> int:
        return 3 if self.display_unit == "mV" else 6

    def get_layout(self):
        return dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Physical Voltage Control", className="card-title"),

                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Label(
                                    "Step size",
                                    style={"marginBottom": "0"},
                                ),
                                width="auto",
                            ),
                            dbc.Col(
                                dbc.Input(
                                    id=self._get_id("step-size"),
                                    type="number",
                                    value=self._to_display_voltage(self.step_size),
                                    step=1e-4,
                                ),
                                width=3,
                            ),
                            dbc.Col(
                                dcc.Dropdown(
                                    id=self._get_id("display-unit"),
                                    options=[
                                        {"label": "V", "value": "V"},
                                        {"label": "mV", "value": "mV"},
                                    ],
                                    value=self.display_unit,
                                    clearable=False,
                                    searchable=False,
                                    style={
                                        "width": "80px",
                                        "color": "black",
                                    },
                                ),
                                width="auto",
                            ),
                        ],
                        align="center",
                        className="mb-3",
                    ),

                    html.Hr(),

                    html.Div(
                        [
                            self.rows[name].get_layout()
                            for name in self.voltage_parameters
                        ]
                    ),

                    dcc.Interval(
                        id=self._get_id("update-interval"),
                        interval=self.update_interval_ms,
                        n_intervals=0,
                    ),
                ]
            ),
            color="dark",
            inverse=True,
        )

    def register_callbacks(self, app: dash.Dash) -> None:
        value_type = f"{self.component_id}-value"
        unit_label_type = f"{self.component_id}-unit-label"
        increment_type = f"{self.component_id}-increment"
        decrement_type = f"{self.component_id}-decrement"

        @app.callback(
            Output(self._get_id("step-size"), "value", allow_duplicate=True),
            Output(
                {"type": value_type, "index": ALL},
                "value",
                allow_duplicate=True,
            ),
            Output(
                {"type": unit_label_type, "index": ALL},
                "children",
            ),
            Input(self._get_id("display-unit"), "value"),
            State(self._get_id("step-size"), "value"),
            State({"type": value_type, "index": ALL}, "id"),
            prevent_initial_call=True,
        )
        def change_display_unit(new_unit, current_step_display, value_ids):
            if new_unit not in ("V", "mV"):
                raise PreventUpdate

            old_unit = self.display_unit

            try:
                step_value = float(current_step_display)
            except (TypeError, ValueError):
                raise PreventUpdate

            if new_unit == "mV":
                # V -> mV
                step_in_volts = step_value
            else:
                # mV -> V
                step_in_volts = step_value * 1e-3

            self.display_unit = new_unit
            new_step_display = self._to_display_voltage(step_in_volts)
            precision = self._display_precision()

            values = []
            for value_id in value_ids:
                gate_name = value_id["index"]
                param = self.voltage_parameters[gate_name]

                try:
                    value = param.get_latest()

                    if value is None:
                        value = param.get()

                    display_value = self._to_display_voltage(float(value))
                    values.append(
                        format_voltage_fixed(display_value, precision=precision)
                    )

                except Exception:
                    logger.exception("Failed to read voltage for %s", gate_name)
                    values.append(dash.no_update)

            unit_labels = [new_unit for _ in value_ids]

            return new_step_display, values, unit_labels

        @app.callback(
            Output(
                {"type": value_type, "index": MATCH},
                "value",
                allow_duplicate=True,
            ),
            Input({"type": increment_type, "index": MATCH}, "n_clicks"),
            Input({"type": decrement_type, "index": MATCH}, "n_clicks"),
            State(self._get_id("step-size"), "value"),
            State(self._get_id("display-unit"), "value"),
            prevent_initial_call=True,
        )
        def change_voltage(
            increment_clicks,
            decrement_clicks,
            step_size,
            display_unit,
        ):
            if step_size is None:
                raise PreventUpdate

            triggered = dash.ctx.triggered_id

            if not isinstance(triggered, dict):
                raise PreventUpdate

            gate_name = triggered.get("index")
            trigger_type = triggered.get("type")

            if gate_name not in self.voltage_parameters:
                raise PreventUpdate

            self.display_unit = display_unit
            step = self._from_display_voltage(float(step_size))

            if trigger_type == increment_type:
                if not increment_clicks:
                    raise PreventUpdate
                delta = step

            elif trigger_type == decrement_type:
                if not decrement_clicks:
                    raise PreventUpdate
                delta = -step

            else:
                raise PreventUpdate

            param = self.voltage_parameters[gate_name]

            current = param.get_latest()

            if current is None:
                current = param.get()

            current = float(current)
            target = current + delta

            logger.info("Setting %s: %.6f V -> %.6f V", gate_name, current, target)

            # If this physical gate belongs to the VirtualDCSet,
            # let VDC perform the update so its internal cache stays synchronized.
            try:
                if self.dc_set is not None and gate_name in self.dc_set.channels:
                    self.dc_set.set_voltages(
                        {gate_name: target},
                        requery=False,
                        resync=False,
                    )
                else:
                    param.set(target)

            except Exception:
                logger.exception(
                    "Failed to set physical gate %s from %.6f V to %.6f V",
                    gate_name,
                    current,
                    target,
                )
                raise PreventUpdate

            updated = param.get_latest()

            if updated is None:
                updated = target

            display_value = self._to_display_voltage(float(updated))

            return format_voltage_fixed(
                display_value,
                precision=self._display_precision(),
            )

        @app.callback(
            Output(
                {"type": value_type, "index": ALL},
                "value",
                allow_duplicate=True,
            ),
            Input(self._get_id("update-interval"), "n_intervals"),
            State({"type": value_type, "index": ALL}, "id"),
            State(self._get_id("display-unit"), "value"),
            prevent_initial_call=True,
        )
        def refresh_voltages(_n_intervals, value_ids, display_unit):
            if not value_ids:
                return []

            self.display_unit = display_unit
            precision = self._display_precision()
            values = []

            for value_id in value_ids:
                gate_name = value_id["index"]
                param = self.voltage_parameters[gate_name]

                try:
                    value = param.get_latest()

                    if value is None:
                        value = param.get()

                    display_value = self._to_display_voltage(float(value))
                    values.append(
                        format_voltage_fixed(display_value, precision=precision)
                    )

                except Exception:
                    logger.exception("Failed to read voltage for %s", gate_name)
                    values.append(dash.no_update)

            return values

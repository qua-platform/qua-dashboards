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

        # Dashboard panel size
        self.layout_columns = 12
        self.layout_rows = max(4, len(self.voltage_parameters))

        self.rows = {
            name: PhysicalVoltageControlRow(
                component_id=self.component_id,
                param=param,
                label=name,
                channel_label=self.channel_labels.get(name),
            )
            for name, param in self.voltage_parameters.items()
        }

    def get_layout(self):
        return dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Physical Voltage Control", className="card-title"),

                    dbc.Row(
                        [
                            dbc.Col(dbc.Label("Step size", style={"marginBottom": "0"}), width="auto"),
                            dbc.Col(
                                dbc.Input(
                                    id=self._get_id("step-size"),
                                    type="number",
                                    value=self.step_size,
                                    step=1e-4,
                                ),
                                width=3,
                            ),
                            dbc.Col(dbc.Label("V", style={"marginBottom": "0"}), width="auto"),
                        ],
                        align="center",
                        className="mb-3",
                    ),

                    html.Hr(),

                    html.Div([self.rows[name].get_layout() for name in self.voltage_parameters]),

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
        increment_type = f"{self.component_id}-increment"
        decrement_type = f"{self.component_id}-decrement"

        @app.callback(
            Output({"type": value_type, "index": MATCH}, "value", allow_duplicate=True),
            Input({"type": increment_type, "index": MATCH}, "n_clicks"),
            Input({"type": decrement_type, "index": MATCH}, "n_clicks"),
            State(self._get_id("step-size"), "value"),
            prevent_initial_call=True,
        )
        def change_voltage(increment_clicks, decrement_clicks, step_size):
            if step_size is None:
                raise PreventUpdate

            triggered = dash.ctx.triggered_id

            if not isinstance(triggered, dict):
                raise PreventUpdate

            gate_name = triggered.get("index")
            trigger_type = triggered.get("type")

            if gate_name not in self.voltage_parameters:
                raise PreventUpdate

            step = float(step_size)

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

            return format_voltage_fixed(updated)

        @app.callback(
            Output({"type": value_type, "index": ALL}, "value", allow_duplicate=True),
            Input(self._get_id("update-interval"), "n_intervals"),
            State({"type": value_type, "index": ALL}, "id"),
            prevent_initial_call=True,
        )
        def refresh_voltages(_n_intervals, value_ids):
            if not value_ids:
                return []

            values = []

            for value_id in value_ids:
                gate_name = value_id["index"]
                param = self.voltage_parameters[gate_name]

                try:
                    value = param.get_latest()

                    if value is None:
                        value = param.get()

                    values.append(format_voltage_fixed(float(value)))

                except Exception:
                    logger.exception("Failed to read voltage for %s", gate_name)
                    values.append(dash.no_update)

            return values
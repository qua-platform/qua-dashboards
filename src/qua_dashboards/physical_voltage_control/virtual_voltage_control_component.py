"""
Relative virtual-gate voltage control.

Available virtual-gate names are obtained dynamically from the VirtualGateSet.

Users select which virtual gates they want to control from a dropdown.
The selected control rows remain stable and are rebuilt only when the
dropdown selection changes.

Voltage changes are applied through VirtualDCSet.
"""

import logging

import dash
from dash import ALL, MATCH, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from qua_dashboards.core import BaseComponent
from .physical_voltage_control_row import format_voltage_fixed
from .virtual_voltage_control_row import VirtualVoltageControlRow


logger = logging.getLogger(__name__)


class VirtualVoltageControlComponent(BaseComponent):
    def __init__(
        self,
        component_id: str,
        gate_set,
        dc_set,
        step_size: float = 1e-3,
        update_interval_ms: int = 2000,
        preselected_gates=None,
    ):
        super().__init__(component_id=component_id)

        self.gate_set = gate_set
        self.dc_set = dc_set
        self.step_size = step_size
        self._offsets: dict[str, float] = {}

        self.display_unit = "mV"
        
        self.update_interval_ms = update_interval_ms

        available = set(self.virtual_gate_names)

        self.preselected_gates = [
            name
            for name in (preselected_gates or [])
            if name in available
        ]

    def _to_display_voltage(self, value: float) -> float:
        if self.display_unit == "mV":
            return value * 1e3

        return value


    def _from_display_voltage(self, value: float) -> float:
        if self.display_unit == "mV":
            return value * 1e-3

        return value


    @property
    def virtual_gate_names(self):
        """
        Return all virtual gates currently available in the VirtualGateSet.
        """

        physical_names = set(self.gate_set.channels.keys())

        return sorted(
            name
            for name in self.gate_set.valid_channel_names
            if name not in physical_names
        )

    def _build_gate_options(self):
        """
        Build dropdown options from the current VirtualGateSet.
        """

        return [
            {
                "label": name,
                "value": name,
            }
            for name in self.virtual_gate_names
        ]

    def _build_rows(self, selected_names):
        """
        Build control rows only for gates selected by the user.
        """

        if not selected_names:
            return []

        available = set(self.virtual_gate_names)

        rows = []

        for name in selected_names:
            if name not in available:
                continue

            self._offsets.setdefault(name, 0.0)

            row = VirtualVoltageControlRow(
                component_id=self.component_id,
                name=name,
                offset=self._offsets[name],
                display_unit=self.display_unit,
            )

            rows.append(row.get_layout())

        return rows

    def get_layout(self):
        return dbc.Card(
            dbc.CardBody(
                [
                    html.H5(
                        "Virtual Gate Control",
                        className="card-title",
                    ),

                    dbc.Label(
                        "Virtual gates",
                        style={"marginBottom": "4px"},
                    ),

                    dcc.Dropdown(
                        id=self._get_id("virtual-gate-selector"),
                        options=self._build_gate_options(),
                        value=self.preselected_gates,
                        multi=True,
                        searchable=True,
                        clearable=True,
                        placeholder="Search or select virtual gates...",
                        className="virtual-gate-dropdown",
                        style={
                            "color": "black",
                        },
                    ),

                    html.Br(),

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
                                    id=self._get_id("virtual-step-size"),
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

                            dbc.Col(
                                dbc.Button(
                                    "Set reference",
                                    id=self._get_id("set-reference"),
                                    n_clicks=0,
                                    size="sm",
                                    outline=True,
                                    color="secondary",
                                ),
                                width="auto",
                            ),
                        ],
                        align="center",
                        className="mb-3",
                    ),

                    html.Hr(),

                    html.Div(
                        id=self._get_id("virtual-rows-container"),
                        children=self._build_rows(self.preselected_gates),
                    ),

                    dcc.Interval(
                        id=self._get_id("virtual-gate-list-refresh"),
                        interval=self.update_interval_ms,
                        n_intervals=0,
                    ),

                    dcc.Store(
                        id=self._get_id("virtual-action-store"),
                        data=0,
                    ),
                ]
            ),
            color="dark",
            inverse=True,
        )

    def register_callbacks(self, app: dash.Dash) -> None:
        offset_type = f"{self.component_id}-virtual-offset"
        increment_type = f"{self.component_id}-virtual-increment"
        decrement_type = f"{self.component_id}-virtual-decrement"

        # ------------------------------------------------------------
        # Periodically refresh only the dropdown options.
        #
        # This allows newly-created virtual gates to appear in the
        # dropdown without rebuilding the active control rows.
        # ------------------------------------------------------------
        @app.callback(
            Output(
                self._get_id("virtual-step-size"),
                "value",
                allow_duplicate=True,
            ),
            Output(
                {
                    "type": offset_type,
                    "index": ALL,
                },
                "value",
                allow_duplicate=True,
            ),
            Output(
                {
                    "type": f"{self.component_id}-virtual-unit-label",
                    "index": ALL,
                },
                "children",
            ),
            Input(
                self._get_id("display-unit"),
                "value",
            ),
            State(
                self._get_id("virtual-step-size"),
                "value",
            ),
            State(
                {
                    "type": offset_type,
                    "index": ALL,
                },
                "id",
            ),
            prevent_initial_call=True,
        )
        def change_display_unit(
            new_unit,
            current_step_display,
            offset_input_ids,
        ):
            if new_unit not in ("V", "mV"):
                raise PreventUpdate

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

            offset_values = []

            for input_id in offset_input_ids:
                name = input_id["index"]
                value_in_volts = self._offsets.get(name, 0.0)

                display_value = self._to_display_voltage(value_in_volts)

                precision = 3 if new_unit == "mV" else 6

                offset_values.append(
                    format_voltage_fixed(
                        display_value,
                        precision=precision,
                    )
                )

            unit_labels = [
                new_unit
                for _ in offset_input_ids
            ]

            return (
                new_step_display,
                offset_values,
                unit_labels,
            )
        @app.callback(
            Output(
                {
                    "type": offset_type,
                    "index": MATCH,
                },
                "value",
                allow_duplicate=True,
            ),
            Input(
                {
                    "type": offset_type,
                    "index": MATCH,
                },
                "n_submit",
            ),
            State(
                {
                    "type": offset_type,
                    "index": MATCH,
                },
                "value",
            ),
            State(
                {
                    "type": offset_type,
                    "index": MATCH,
                },
                "id",
            ),
            State(
                self._get_id("display-unit"),
                "value",
            ),
            prevent_initial_call=True,
        )
        def set_virtual_offset(
            n_submit,
            submitted_value,
            input_id,
            display_unit,
        ):
            if not n_submit or submitted_value is None:
                raise PreventUpdate

            gate_name = input_id["index"]

            if gate_name not in self.virtual_gate_names:
                raise PreventUpdate

            try:
                submitted_display_value = float(submitted_value)

                self.display_unit = display_unit

                new_offset = self._from_display_voltage(
                    submitted_display_value
                )

            except (TypeError, ValueError):
                raise PreventUpdate

            old_offset = self._offsets.get(gate_name, 0.0)

            delta = new_offset - old_offset

            try:
                self.dc_set.increment_voltages(
                    {gate_name: delta},
                    requery=False,
                    resync=False,
                )
            except Exception:
                logger.exception(
                    "Failed to change virtual gate %s offset from %.6f V to %.6f V",
                    gate_name,
                    old_offset,
                    new_offset,
                )
                raise PreventUpdate

            self._offsets[gate_name] = new_offset

            logger.info(
                "Virtual gate %s offset: %.6f V -> %.6f V",
                gate_name,
                old_offset,
                new_offset,
            )

            display_value = self._to_display_voltage(new_offset)

            precision = 3 if display_unit == "mV" else 6

            return format_voltage_fixed(
                display_value,
                precision=precision,
            )
        
        @app.callback(
            Output(
                self._get_id("virtual-gate-selector"),
                "options",
            ),
            Input(
                self._get_id("virtual-gate-list-refresh"),
                "n_intervals",
            ),
        )
        def refresh_virtual_gate_options(_n_intervals):
            return self._build_gate_options()

        # ------------------------------------------------------------
        # Rebuild control rows only when the user changes the selected
        # gates in the dropdown.
        # ------------------------------------------------------------

        @app.callback(
            Output(
                self._get_id("virtual-rows-container"),
                "children",
            ),
            Input(
                self._get_id("virtual-gate-selector"),
                "value",
            ),
        )
        def update_selected_rows(selected_names):
            return self._build_rows(selected_names)

        # ------------------------------------------------------------
        # Virtual gate +/- control
        # ------------------------------------------------------------

        @app.callback(
            Output(
                self._get_id("virtual-action-store"),
                "data",
            ),
            Output(
                {
                    "type": offset_type,
                    "index": ALL,
                },
                "value",
                allow_duplicate=True,
            ),
            Input(
                {
                    "type": increment_type,
                    "index": ALL,
                },
                "n_clicks",
            ),
            Input(
                {
                    "type": decrement_type,
                    "index": ALL,
                },
                "n_clicks",
            ),
            State(
                self._get_id("virtual-step-size"),
                "value",
            ),
            State(
                self._get_id("display-unit"),
                "value",
            ),
            State(
                self._get_id("virtual-action-store"),
                "data",
            ),
            State(
                {
                    "type": offset_type,
                    "index": ALL,
                },
                "id",
            ),
            prevent_initial_call=True,
        )
        def change_virtual_voltage(
            increment_clicks,
            decrement_clicks,
            step_size,
            display_unit,
            action_counter,
            offset_input_ids,
        ):
            if step_size is None:
                raise PreventUpdate

            triggered = dash.ctx.triggered_id

            if not isinstance(triggered, dict):
                raise PreventUpdate

            gate_name = triggered.get("index")
            trigger_type = triggered.get("type")

            if gate_name not in self.virtual_gate_names:
                raise PreventUpdate

            self.display_unit = display_unit

            step = self._from_display_voltage(
                float(step_size)
            )

            if trigger_type == increment_type:
                if not any(increment_clicks):
                    raise PreventUpdate

                delta = step

            elif trigger_type == decrement_type:
                if not any(decrement_clicks):
                    raise PreventUpdate

                delta = -step

            else:
                raise PreventUpdate

            current_offset = self._offsets.get(gate_name, 0.0)

            try:
                self.dc_set.increment_voltages(
                    {gate_name: delta},
                    requery=False,
                    resync=False,
                )
            except Exception:
                logger.exception(
                    "Failed to increment virtual gate %s by %.6f V",
                    gate_name,
                    delta,
                )
                raise PreventUpdate

            new_offset = current_offset + delta
            self._offsets[gate_name] = new_offset

            logger.info(
                "Virtual gate %s offset: %.6f V -> %.6f V",
                gate_name,
                current_offset,
                new_offset,
            )
            
            offset_values = []

            for input_id in offset_input_ids:
                name = input_id["index"]
                value = self._offsets.get(name, 0.0)

                display_value = self._to_display_voltage(value)

                offset_values.append(
                    format_voltage_fixed(display_value, precision=3)
                )

            return (
                int(action_counter or 0) + 1,
                offset_values,
            )

        @app.callback(
            Output(
                {
                    "type": offset_type,
                    "index": ALL,
                },
                "value",
                allow_duplicate=True,
            ),
            Input(
                self._get_id("set-reference"),
                "n_clicks",
            ),
            State(
                {
                    "type": offset_type,
                    "index": ALL,
                },
                "id",
            ),
            State(
                self._get_id("display-unit"),
                "value",
            ),
            prevent_initial_call=True,
        )
        def set_reference(n_clicks, offset_input_ids, display_unit):
            if not n_clicks:
                    raise PreventUpdate

            # Reset all stored virtual offsets.
            # This does NOT change any physical voltages.
            for name in self._offsets:
                self._offsets[name] = 0.0

            self.display_unit = display_unit

            precision = 3 if display_unit == "mV" else 6

            return [
                format_voltage_fixed(
                    0.0,
                    precision=precision,
                )
                for _ in offset_input_ids
            ]
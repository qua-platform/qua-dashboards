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
from dash import ALL, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from qua_dashboards.core import BaseComponent

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
        self.update_interval_ms = update_interval_ms

        available = set(self.virtual_gate_names)

        self.preselected_gates = [
            name
            for name in (preselected_gates or [])
            if name in available
        ]

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

        return [
            VirtualVoltageControlRow(
                component_id=self.component_id,
                name=name,
            ).get_layout()
            for name in selected_names
            if name in available
        ]

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
                                    value=self.step_size,
                                    step=1e-4,
                                ),
                                width=3,
                            ),

                            dbc.Col(
                                dbc.Label(
                                    "V",
                                    style={"marginBottom": "0"},
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
                self._get_id("virtual-action-store"),
                "data",
            ),
            prevent_initial_call=True,
        )
        def change_virtual_voltage(
            increment_clicks,
            decrement_clicks,
            step_size,
            action_counter,
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

            step = float(step_size)

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

            current = float(
                self.dc_set.get_voltage(
                    gate_name,
                    requery=False,
                )
            )

            target = current + delta

            logger.info(
                "Virtual gate %s: %.6f V -> %.6f V",
                gate_name,
                current,
                target,
            )

            try:
                self.dc_set.set_voltages(
                    {gate_name: target},
                    requery=False,
                    resync=False,
                )

            except Exception:
                logger.exception(
                    "Failed to set virtual gate %s from %.6f V to %.6f V",
                    gate_name,
                    current,
                    target,
                )
                raise PreventUpdate

            return int(action_counter or 0) + 1
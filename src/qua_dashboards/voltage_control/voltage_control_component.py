"""
Dash component for displaying and controlling a series of voltage parameters.
"""

import logging
from typing import Dict, Sequence

import dash
from dash import (
    dcc,
    html,
    Input,
    Output,
    State
)

from qua_dashboards.core import BaseComponent, ParameterProtocol

from .voltage_control_row import VoltageControlRow, format_voltage

logger = logging.getLogger(__name__)

DEFAULT_INPUT_CLASS_NAME = ""  # Input fields will always have this class
COMPONENT_MAX_WIDTH = "450px"


class VoltageControlComponent(BaseComponent):
    """
    A Dash component to display and set voltages from multiple sources.
    Uses VoltageControlRow to manage individual parameter rows.
    Simplified: No edit highlighting, no auto-defocus on enter.
    """

    def __init__(
        self,
        component_id: str,
        voltage_parameters: Sequence[ParameterProtocol],
        update_interval_ms: int = 1000,
        layout_columns: int = 3,
    ):
        super().__init__(component_id=component_id)
        self.voltage_parameters = voltage_parameters
        self.update_interval_ms = update_interval_ms

        self._initial_values_loaded = False  # To ensure first update populates values

        self._row_components: Dict[str, VoltageControlRow] = {}
        input_id_type = self._get_id_type_str("input")
        for param in self.voltage_parameters:
            row = VoltageControlRow(input_id_type=input_id_type, param=param)
            self._row_components[param.name] = row

        self.layout_columns = layout_columns

    def _get_id_type_str(self, element_name: str) -> str:
        return f"comp-{self.component_id}-{element_name}"

    def get_layout(self) -> html.Div:
        # Layout generation remains the same structurally
        if not self._row_components:
            return html.Div(
                "No voltage parameters configured.",
                style={"maxWidth": COMPONENT_MAX_WIDTH},
            )
        return html.Div(
            [
                dcc.Store(id = self._get_id("offsets-refresh-trigger"), data = 0),
                dcc.Interval(
                    id=self._get_id("update-interval"),
                    interval=self.update_interval_ms,
                    n_intervals=0,
                ),
                *[row.get_layout() for row in self._row_components.values()],
                html.Div([
                    html.Button("Apply", id = self._get_id("apply"), className="btn btn-primary"),
                ], 
                style={"maxWidth": COMPONENT_MAX_WIDTH, "margin": "0 auto 8px auto"},),
            ],
            style={"maxWidth": COMPONENT_MAX_WIDTH, "margin": "0 auto"},
        )

    def register_callbacks(self, app: dash.Dash) -> None:
        if not self.voltage_parameters:
            return

        output_list_for_periodic_update = []
        input_id_type_str = self._get_id_type_str("input")
        param_input_ids = [{"type": input_id_type_str, "index": param.name} for param in self.voltage_parameters]
        for param in self.voltage_parameters:
            param_input_id = {"type": input_id_type_str, "index": param.name}
            output_list_for_periodic_update.append(
                Output(param_input_id, "value", allow_duplicate=True)
            )
            output_list_for_periodic_update.append(
                Output(
                    param_input_id, "className", allow_duplicate=True
                )  # Keep for structure if row outputs it
            )

        if not output_list_for_periodic_update:
            return

        apply_id = self._get_id("apply")
        bump_store = self._get_id("offsets-refresh-trigger")
        
        @app.callback(
            Output(bump_store, "data"),
            Input(apply_id, "n_clicks"),                            # reuse same outputs
            *[State(pid, "value") for pid in param_input_ids],
            prevent_initial_call=True,
        )
        def on_apply(n_clicks, *texts):
            if not n_clicks:
                raise dash.exceptions.PreventUpdate

            ok = True
            for param, text in zip(self.voltage_parameters, texts):
                try:
                    val = float(str(text).replace("V", "").strip())
                    param.set(val)
                except Exception as e:
                    ok = False
                    logger.warning(f"APPLY parse/set failed for {param.name}: {e}")

            return n_clicks

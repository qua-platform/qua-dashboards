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
)

from qua_dashboards.core import BaseComponent, ParameterProtocol
from qua_dashboards.utils import CallbackParameter

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
        callback_on_param_change = None
        
    ):
        super().__init__(component_id=component_id)
        self.voltage_parameters = voltage_parameters
        #wrap parameters to trigger callbacks on change.
        if callback_on_param_change is not None:
            self.voltage_parameters = [CallbackParameter(param, callback_on_param_change) for param in self.voltage_parameters]
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
                dcc.Interval(
                    id=self._get_id("update-interval"),
                    interval=self.update_interval_ms,
                    n_intervals=0,
                ),
                *[row.get_layout() for row in self._row_components.values()],
            ],
            style={"maxWidth": COMPONENT_MAX_WIDTH, "margin": "0 auto"},
        )

    def register_callbacks(self, app: dash.Dash) -> None:
        if not self.voltage_parameters:
            return

        output_list_for_periodic_update = []
        input_id_type_str = self._get_id_type_str("input")
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

        @app.callback(
            output_list_for_periodic_update,
            Input(self._get_id("update-interval"), "n_intervals"),
            prevent_initial_call="initial_duplicate",
        )
        def periodic_update(_n_intervals: int):
            outputs_tuple_elements = []
            first_load = not self._initial_values_loaded

            for param in self.voltage_parameters:
                param_name = param.name
                control_row = self._row_components[param_name]
                current_input_field_value = control_row.current_input_text

                try:
                    live_value = param.get_latest()
                    text_to_display = format_voltage(live_value)
                except Exception as e:
                    logger.error(f"Err get_latest for {param_name}: {e}", exc_info=True)
                    text_to_display = format_voltage(live_value)

                if not first_load and current_input_field_value is not None:
                    # User has typed something, don't overwrite from periodic update immediately
                    # Let blur/submit handle it. But ensure formatting is eventually correct.
                    # The row's blur callback will restore to last_known_value (formatted).
                    # The row's submit callback will set to submitted_value (formatted).
                    outputs_tuple_elements.append(dash.no_update)
                else:
                    outputs_tuple_elements.append(text_to_display)

                outputs_tuple_elements.append(
                    DEFAULT_INPUT_CLASS_NAME
                )  # Always default class

            if first_load and self.voltage_parameters:
                self._initial_values_loaded = True
            return tuple(outputs_tuple_elements)

        for row_component in self._row_components.values():
            row_component.register_callbacks(app)

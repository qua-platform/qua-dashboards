"""
Dash component for displaying and controlling a series of voltage parameters.
"""

import logging
from typing import List, Dict, Any, Sequence

import dash
from dash import (
    dcc,
    html,
    Input,
    Output,
)  # MATCH removed as it's in row

from qua_dashboards.core import BaseComponent
from .voltage_parameter_protocol import VoltageParameterProtocol
from .voltage_control_row import VoltageControlRow  # Import the new row class

logger = logging.getLogger(__name__)

EDITING_CLASS_NAME = "voltage-input-editing"
DEFAULT_INPUT_CLASS_NAME = ""
COMPONENT_MAX_WIDTH = "500px"  # Keep overall component width


class VoltageControlComponent(BaseComponent):
    """
    A Dash component to display and set voltages from multiple sources.
    Uses VoltageControlRow to manage individual parameter rows.
    """

    def __init__(
        self,
        component_id: str,
        voltage_parameters: Sequence[VoltageParameterProtocol],
        update_interval_ms: int = 1000,
    ):
        super().__init__(component_id=component_id)
        self.voltage_parameters = voltage_parameters  # Keep for periodic update
        self.update_interval_ms = update_interval_ms

        # Centralized state
        self._internal_states: Dict[str, Dict[str, Any]] = {
            param.name: {
                "last_known_value": None,
                "is_editing": False,
                "current_input_text": None,
            }
            for param in self.voltage_parameters
        }
        self._initial_values_loaded = False

        # Create row components
        self._row_components: List[VoltageControlRow] = []
        input_id_type = self._get_id_type_str("input")  # Base type for inputs
        for param in self.voltage_parameters:
            row = VoltageControlRow(
                input_id_type=input_id_type,
                param=param,
                update_parent_state_callback=self._update_internal_state_field,
                trigger_set_value_callback=self._trigger_parameter_set,
                get_parent_state_callback=self._get_internal_state_field,
            )
            self._row_components.append(row)

    # --- Helper methods for row components to interact with parent state ---
    def _update_internal_state_field(
        self, param_name: str, field_key: str, new_value: Any
    ) -> None:
        if (
            param_name in self._internal_states
            and field_key in self._internal_states[param_name]
        ):
            self._internal_states[param_name][field_key] = new_value
        else:
            logger.warning(
                f"Attempted to update invalid state: {param_name}.{field_key}"
            )

    def _get_internal_state_field(self, param_name: str, field_key: str) -> Any:
        return self._internal_states.get(param_name, {}).get(field_key)

    def _trigger_parameter_set(self, param_name: str, value_to_set: float) -> bool:
        param_obj = next(
            (p for p in self.voltage_parameters if p.name == param_name), None
        )
        if not param_obj:
            logger.error(f"Parameter {param_name} not found for set operation.")
            return False
        try:
            param_obj.set(value_to_set)
            logger.info(f"Successfully set {param_name} to {value_to_set}")
            # Optimistic update in parent's state after successful hardware set
            self._internal_states[param_name]["last_known_value"] = value_to_set
            self._internal_states[param_name]["is_editing"] = False
            self._internal_states[param_name]["current_input_text"] = None
            return True
        except Exception as e:
            logger.error(
                f"Error calling set for {param_name} with {value_to_set}: {e}",
                exc_info=True,
            )
            # Optionally revert optimistic update or signal error to UI
            return False

    # --- End Helper methods ---

    def _get_id_type_str(self, element_name: str) -> str:
        return f"comp-{self.component_id}-{element_name}"

    def get_layout(self) -> html.Div:
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
                # Get layout from each row component
                *[row.get_layout() for row in self._row_components],
            ],
            style={"maxWidth": COMPONENT_MAX_WIDTH, "margin": "0 auto"},
        )

    def register_callbacks(self, app: dash.Dash) -> None:
        if not self.voltage_parameters:
            logger.warning(f"VCC ({self.component_id}): No params, skipping cbacks.")
            return

        # --- Periodic Update Callback (remains in parent) ---
        # It needs to output to ALL inputs, so pattern matching on outputs is complex here
        # Simpler to construct the list of Outputs directly
        output_list_for_periodic_update = []
        input_id_type_str = self._get_id_type_str("input")
        for param in self.voltage_parameters:
            # Construct the ID dict for each input
            param_input_id = {"type": input_id_type_str, "index": param.name}
            output_list_for_periodic_update.append(
                Output(param_input_id, "value", allow_duplicate=True)
            )
            output_list_for_periodic_update.append(
                Output(param_input_id, "className", allow_duplicate=True)
            )

        if (
            not output_list_for_periodic_update
        ):  # Should be redundant due to earlier check
            return

        @app.callback(
            output_list_for_periodic_update,
            Input(self._get_id("update-interval"), "n_intervals"),
            prevent_initial_call="initial_duplicate",
        )
        def periodic_update(_n_intervals: int):
            # (Logic is largely the same as before, using self._internal_states)
            outputs_tuple_elements = []
            first_load = not self._initial_values_loaded

            for param in self.voltage_parameters:
                param_name = param.name
                state = self._internal_states[param_name]
                try:
                    live_value = param.get_latest()
                    state["last_known_value"] = live_value  # Update central knowledge
                    text_to_display = str(live_value)
                    class_to_display = DEFAULT_INPUT_CLASS_NAME
                except Exception as e:
                    logger.error(f"Err get_latest for {param_name}: {e}", exc_info=True)
                    text_to_display = (
                        str(state["last_known_value"])
                        if state["last_known_value"] is not None
                        else "Error"
                    )
                    class_to_display = EDITING_CLASS_NAME  # Show error

                if state["is_editing"]:
                    # User is typing, periodic update should not change the input text
                    outputs_tuple_elements.append(dash.no_update)
                    # But, it should ensure style reflects if text is different from new live_value
                    is_different_from_live = False
                    try:
                        if state["current_input_text"] is not None:
                            is_different_from_live = (
                                abs(float(state["current_input_text"]) - live_value)
                                > 1e-9
                            )
                    except ValueError:
                        is_different_from_live = True
                    outputs_tuple_elements.append(
                        EDITING_CLASS_NAME
                        if is_different_from_live
                        else DEFAULT_INPUT_CLASS_NAME
                    )
                else:
                    outputs_tuple_elements.append(text_to_display)
                    outputs_tuple_elements.append(class_to_display)

            if first_load and self.voltage_parameters:
                self._initial_values_loaded = True
            return tuple(outputs_tuple_elements)

        # --- Register callbacks for each row component ---
        for row_component in self._row_components:
            row_component.register_callbacks(app)

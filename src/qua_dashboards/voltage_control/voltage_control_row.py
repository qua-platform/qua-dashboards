"""
Manages the layout and callbacks for a single voltage control row
within the VoltageControlComponent.
"""

import logging
from typing import Dict, Any, Callable, Optional
import json  # For clientside callback ID

import dash
from dash import dcc, html, Input, Output, State, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from .voltage_parameter_protocol import VoltageParameterProtocol

logger = logging.getLogger(__name__)

EDITING_CLASS_NAME = "voltage-input-editing"
DEFAULT_INPUT_CLASS_NAME = ""
LABEL_WIDTH = "200px"
INPUT_WIDTH = "150px"
UNITS_WIDTH = "80px"
DEFAULT_PRECISION = 6  # For voltage display


def _format_voltage_display(
    value: Optional[float], precision: int = DEFAULT_PRECISION
) -> str:
    """Formats a float value for display, handling None and errors."""
    if value is None:
        return ""  # Display as blank if None
    try:
        # Ensure it's a float before formatting to handle potential string inputs from state
        return f"{float(value):.{precision}f}"
    except (ValueError, TypeError):
        logger.warning(f"Could not format value '{value}' as float.")
        return str(value)  # Fallback, or could be "Invalid"


class VoltageControlRow:
    """
    Represents and manages a single row in the VoltageControlComponent.
    Handles its own layout and registers its specific interactive callbacks.
    """

    def __init__(
        self,
        input_id_type: str,
        param: VoltageParameterProtocol,
        update_parent_state_callback: Callable[[str, str, Any], None],
        trigger_set_value_callback: Callable[[str, float], bool],
        get_parent_state_callback: Callable[[str, str], Any],
        parent_component_id: str,  # For dummy output ID uniqueness
    ):
        self.input_id_type = input_id_type
        self.param = param
        self._update_parent_state = update_parent_state_callback
        self._trigger_set_value = trigger_set_value_callback
        self._get_parent_state = get_parent_state_callback
        self.parent_component_id = (
            parent_component_id  # To make dummy output IDs unique
        )

        self.input_id = {"type": self.input_id_type, "index": self.param.name}
        # Dummy div for clientside callback output (to avoid "Inputs ... have already been used" error)
        self.dummy_output_id_for_blur = {
            "type": f"comp-{self.parent_component_id}-blur-dummy-output",
            "index": self.param.name,
        }

    def get_layout(self) -> dbc.Row:
        """Generates the Dash layout for this single voltage control row."""
        return dbc.Row(
            [
                dbc.Col(
                    dbc.Label(self.param.label, style={"whiteSpace": "nowrap"}),
                    style={"width": LABEL_WIDTH, "flex": f"0 0 {LABEL_WIDTH}"},
                ),
                dbc.Col(
                    dbc.Input(
                        id=self.input_id,
                        type="text",
                        value=_format_voltage_display(
                            None
                        ),  # Initially blank, populated by parent
                        n_submit=0,
                        n_blur=0,
                        debounce=False,
                        className=DEFAULT_INPUT_CLASS_NAME,
                        style={"width": "100%"},
                    ),
                    style={"width": INPUT_WIDTH, "flex": f"0 0 {INPUT_WIDTH}"},
                ),
                dbc.Col(
                    dbc.Label(self.param.units, style={"whiteSpace": "nowrap"}),
                    style={
                        "width": UNITS_WIDTH,
                        "flex": f"0 0 {UNITS_WIDTH}",
                        "textAlign": "left",
                    },
                ),
                # Hidden div for clientside callback to target
                html.Div(id=self.dummy_output_id_for_blur, style={"display": "none"}),
            ],
            align="center",
            className="mb-2 g-3",
        )

    def register_callbacks(self, app: dash.Dash) -> None:
        """Registers callbacks specific to this row's input field."""

        # --- Server-side Callbacks ---
        @app.callback(
            Output(self.input_id, "className", allow_duplicate=True),
            Input(self.input_id, "value"),
            prevent_initial_call=True,
        )
        def handle_input_text_change(text_value: Optional[str]):
            if text_value is None:
                text_value = ""

            self._update_parent_state(self.param.name, "current_input_text", text_value)
            self._update_parent_state(self.param.name, "is_editing", True)

            last_known_value = self._get_parent_state(
                self.param.name, "last_known_value"
            )
            is_different = False
            if last_known_value is not None:
                try:
                    # Compare based on unformatted potential float value of input
                    is_different = (
                        abs(float(text_value) - float(last_known_value)) > 1e-9
                    )  # Precision guard for float comparison
                except ValueError:
                    is_different = True
            else:
                is_different = bool(text_value.strip())

            return EDITING_CLASS_NAME if is_different else DEFAULT_INPUT_CLASS_NAME

        @app.callback(
            Output(self.input_id, "value", allow_duplicate=True),
            Output(self.input_id, "className", allow_duplicate=True),
            Input(self.input_id, "n_submit"),
            State(self.input_id, "value"),
            prevent_initial_call=True,
        )
        def handle_input_submit(_n_submit: int, submitted_text_value: Optional[str]):
            if not _n_submit or submitted_text_value is None:
                raise PreventUpdate

            try:
                float_value = float(submitted_text_value)
                set_successful = self._trigger_set_value(self.param.name, float_value)

                if set_successful:
                    # Parent state updated by _trigger_set_value
                    # Optimistically display the formatted submitted value
                    return _format_voltage_display(
                        float_value
                    ), DEFAULT_INPUT_CLASS_NAME
                else:
                    # If set failed, keep current text and editing style
                    return dash.no_update, EDITING_CLASS_NAME
            except ValueError:
                logger.warning(
                    f"Invalid float: '{submitted_text_value}' for {self.param.name}"
                )
                return dash.no_update, EDITING_CLASS_NAME
            except Exception as e:
                logger.error(
                    f"Error during set for {self.param.name}: {e}", exc_info=True
                )
                return dash.no_update, EDITING_CLASS_NAME

        @app.callback(
            Output(self.input_id, "value", allow_duplicate=True),
            Output(self.input_id, "className", allow_duplicate=True),
            Input(self.input_id, "n_blur"),
            prevent_initial_call=True,
        )
        def handle_input_blur(_n_blur: int):
            if not _n_blur:
                raise PreventUpdate

            if self._get_parent_state(self.param.name, "is_editing"):
                logger.debug(f"Blur on {self.param.name} while editing, reverting.")
                self._update_parent_state(self.param.name, "is_editing", False)
                self._update_parent_state(self.param.name, "current_input_text", None)

                last_known = self._get_parent_state(self.param.name, "last_known_value")
                return _format_voltage_display(last_known), DEFAULT_INPUT_CLASS_NAME
            else:
                logger.debug(f"Blur on {self.param.name}, not editing. No change.")
                raise PreventUpdate

        # --- Clientside Callback for Blurring on Enter ---
        # The input_id needs to be correctly stringified for use in getElementById
        # Dash's pattern-matching IDs are JSON strings: {"index":"name","type":"type_str"}
        # The order of keys in the JSON string matters for getElementById.
        # Dash typically serializes with 'index' then 'type' if both present.
        # It's safer to get the actual ID string if possible, or construct it carefully.
        # For simplicity, we assume 'index' then 'type' if both are present.
        # If only 'type' and 'index', it's `{"index": "...", "type": "..."}`

        # Create the ID string as Dash would for the DOM
        # Note: dash.development.base_component.generate_id might be useful if accessible
        # but it's simpler to construct based on known pattern.
        # A dict with "type" and "index" is stringified with keys sorted alphabetically by default by json.dumps
        clientside_id_str = json.dumps(self.input_id, sort_keys=True)

        app.clientside_callback(
            f"""
            function(n_submit) {{
                if (n_submit && n_submit > 0) {{
                    // We need a brief delay to allow the server-side callback for n_submit
                    // to potentially update the value before blurring.
                    // Otherwise, blur might happen on the old value if user types fast.
                    // This is a common challenge with coordinating server/client actions.
                    // A small timeout is a pragmatic approach.
                    setTimeout(function() {{
                        const el = document.getElementById({clientside_id_str});
                        if (el) {{
                            el.blur();
                        }}
                    }}, 50); // 50ms delay, adjust if needed
                }}
                return dash_clientside.no_update; // Or an empty string for a dummy children prop
            }}
            """,
            Output(
                self.dummy_output_id_for_blur, "children"
            ),  # Output to the dummy div
            Input(self.input_id, "n_submit"),
            prevent_initial_call=True,
        )

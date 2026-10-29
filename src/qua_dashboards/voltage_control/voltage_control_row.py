"""
Manages the layout and callbacks for a single voltage control row
within the VoltageControlComponent.
"""

import logging
from typing import List, Optional

import dash
from dash import Input, Output, State, ClientsideFunction, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from qua_dashboards.core import ParameterProtocol

logger = logging.getLogger(__name__)

DEFAULT_INPUT_CLASS_NAME = ""  # Standard class for inputs
LABEL_WIDTH = "140px"
INPUT_WIDTH = "120px"
UNITS_WIDTH = "45px"
LAST_VALUE_WIDTH = "110px"
DEFAULT_VOLTAGE_PRECISION = 6


def format_voltage(
    value: Optional[float], precision: int = DEFAULT_VOLTAGE_PRECISION
) -> str:
    """
    Formats a float voltage for display.

    Rounds to the specified precision and removes unnecessary trailing zeros.
    Handles None by returning an empty string.
    """
    if value is None:
        return ""
    try:
        # Round to ensure correct number of decimal places if it's close
        # Add a small epsilon for floating point inaccuracies before rounding
        # for numbers like x.999999...
        rounded_value = round(float(value) + 1e-12, precision)

        # Convert to string. If it became an integer, str() handles it.
        s = str(rounded_value)

        if "." in s:
            integer_part, decimal_part = s.split(".")
            # Ensure decimal part is no longer than precision and strip trailing zeros
            decimal_part = decimal_part[:precision].rstrip("0")
            if not decimal_part:  # All zeros were stripped, or it was .0, .00 etc.
                return integer_part
            return f"{integer_part}.{decimal_part}"
        return s  # It's an integer
    except (ValueError, TypeError):
        logger.warning(f"Could not format value '{value}' as float for display.")
        return str(value)  # Fallback


class VoltageControlRow:
    """
    Represents and manages a single row in the VoltageControlComponent.
    """

    def __init__(
        self,
        input_id_type: str,
        param: ParameterProtocol,
    ):
        self.input_id_type = input_id_type
        self.param = param
        self.current_input_text = format_voltage(param.get_latest())
        self.input_id = {"type": self.input_id_type, "index": self.param.name}
        self.last_committed_text = self.current_input_text

        # Previously applied values, most recent last. Undo walks back to the value at instantiation.
        self.current_value = param.get_latest()
        self._undo_stack: List[float] = []

        self.last_value_id = {
            "type": f"{self.input_id_type}-last-value",
            "index": self.param.name,
        }
        self.undo_button_id = {
            "type": f"{self.input_id_type}-undo",
            "index": self.param.name,
        }

        # ID for the dummy div that triggers the blur action
        self.blur_trigger_id = {
            "type": f"{self.input_id_type}-blur-trigger",
            "index": self.param.name,
        }

    @property
    def last_value(self) -> Optional[float]:
        return self._undo_stack[-1] if self._undo_stack else None

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def _set(self, value: float) -> str:
        self.param.set(value)
        self.current_value = value
        self.current_input_text = format_voltage(value)
        self.last_committed_text = self.current_input_text
        return self.current_input_text

    def apply_value(self, value: float) -> str:
        """
        Sets the parameter to value, pushing the previously held value onto the undo stack.

        Returns the formatted text to display in the input.
        """
        previous_value = self.current_value
        text = self._set(value)
        if value != previous_value:
            self._undo_stack.append(previous_value)
        return text

    def undo(self) -> str:
        """
        Reapplies the most recent value on the undo stack, removing it from the stack.

        Returns the formatted text to display in the input.
        """
        if not self._undo_stack:
            return self.current_input_text
        value = self._undo_stack[-1]
        text = self._set(value)
        self._undo_stack.pop()
        return text

    def get_last_value_text(self) -> str:
        if self.last_value is None:
            return "Last: –"
        return f"Last: {format_voltage(self.last_value)}"

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
                        value=format_voltage(self.param.get_latest()),
                        n_submit=0,
                        n_blur=0,
                        debounce=False,
                        className=DEFAULT_INPUT_CLASS_NAME,
                        style={"width": "100%"},
                    ),
                    style={
                        "width": INPUT_WIDTH,
                        "flex": f"0 0 {INPUT_WIDTH}",
                        "paddingRight": "0px",
                    },
                ),
                dbc.Col(
                    dbc.Label(
                        children=self.param.unit, style={"whiteSpace": "nowrap"}
                    ),
                    style={
                        "width": UNITS_WIDTH,
                        "flex": f"0 0 {UNITS_WIDTH}",
                        "textAlign": "left",
                    },
                ),
                dbc.Col(
                    html.Small(
                        self.get_last_value_text(),
                        id=self.last_value_id,  # type: ignore
                        style={"whiteSpace": "nowrap"},
                    ),
                    style={
                        "width": LAST_VALUE_WIDTH,
                        "flex": f"0 0 {LAST_VALUE_WIDTH}",
                    },
                ),
                dbc.Col(
                    dbc.Button(
                        "Undo",
                        id=self.undo_button_id,
                        n_clicks=0,
                        disabled=not self.can_undo,
                        size="sm",
                        color="secondary",
                    ),
                    width="auto",
                ),
                # Hidden Div to trigger clientside callback for blur
                html.Div(id=self.blur_trigger_id, style={"display": "none"}),  # type: ignore
            ],
            align="center",
            className="mb-2 g-3 flex-nowrap",
        )

    def register_callbacks(self, app: dash.Dash) -> None:
        """Registers callbacks specific to this row's input field."""

        @app.callback(
            Output(self.input_id, "className", allow_duplicate=True),
            Input(self.input_id, "value"),
            prevent_initial_call=True,
        )
        def handle_input_text_change(text_value: Optional[str]):
            if text_value is None:
                text_value = ""
            self.current_input_text = text_value
            return DEFAULT_INPUT_CLASS_NAME

        @app.callback(
            Output(self.input_id, "value", allow_duplicate=True),
            Output(
                self.blur_trigger_id, "children", allow_duplicate=True
            ),  # Output to trigger blur
            Input(self.input_id, "n_submit"),
            State(self.input_id, "value"),
            prevent_initial_call=True,
        )
        def handle_input_submit(_n_submit: int, submitted_text_value: Optional[str]):
            if not _n_submit or submitted_text_value is None:
                raise PreventUpdate
            try:
                float_value = float(submitted_text_value)
                # Return formatted value and trigger for blur
                return self.apply_value(float_value), _n_submit
            except ValueError:
                logger.warning(
                    f"Invalid float: '{submitted_text_value}' for {self.param.name}"
                )
                # Keep invalid text, no blur trigger update
                return dash.no_update, dash.no_update
            except Exception as e:
                logger.error(
                    f"Error during set for {self.param.name}: {e}", exc_info=True
                )
                return dash.no_update, dash.no_update

        @app.callback(
            Output(self.input_id, "value", allow_duplicate=True),
            Input(self.input_id, "n_blur"),
            prevent_initial_call=True,
        )
        def handle_input_blur(_n_blur: int):
            if not _n_blur:
                raise PreventUpdate
            self.current_input_text = format_voltage(self.param.get_latest())
            self.last_committed_text = self.current_input_text
            return self.current_input_text

        # # Clientside callback to blur the input
        # app.clientside_callback(
        #     r"""
        #     function(children, inputId) {
        #         // This emulates the old custom.blurInput behaviour
        #         try {
        #             if (!inputId) {
        #                 return null;
        #             }
        #             // Dash uses JSON.stringify(id_dict) as the DOM id
        #             var idStr = JSON.stringify(inputId);
        #             var el = document.getElementById(idStr);
        #             if (el && typeof el.blur === 'function') {
        #                 el.blur();
        #             }
        #         } catch (e) {
        #             console.error("blurInput clientside error:", e);
        #         }
        #         // Dummy return value for the Store output
        #         return null;
        #     }
        #     """,
        #     Output(self.blur_trigger_id, "data-last-blurred", allow_duplicate=True),
        #     Input(self.blur_trigger_id, "children"),
        #     State(self.input_id, "id"),
        #     prevent_initial_call=True,
        # )


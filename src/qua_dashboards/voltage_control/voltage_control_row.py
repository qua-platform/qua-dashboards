"""
Manages the layout and callbacks for a single voltage control row
within the VoltageControlComponent.
"""

import logging
from typing import Optional

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

        # ID for the dummy div that triggers the blur action
        self.blur_trigger_id = {
            "type": f"{self.input_id_type}-blur-trigger",
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
                self.param.set(float_value)
                self.current_input_text = format_voltage(float_value)
                # Return formatted value and trigger for blur
                return self.current_input_text, _n_submit
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
            return self.current_input_text

        # Clientside callback to blur the input
        app.clientside_callback(
            ClientsideFunction(namespace="custom", function_name="blurInput"),
            Output(
                self.blur_trigger_id, "data-last-blurred", allow_duplicate=True
            ),  # Dummy output
            Input(self.blur_trigger_id, "children"),
            State(self.input_id, "id"),  # Pass the dict ID of the input
            prevent_initial_call=True,
        )

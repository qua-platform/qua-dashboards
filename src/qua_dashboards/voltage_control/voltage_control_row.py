"""
Manages the layout and callbacks for a single voltage control row
within the VoltageControlComponent.
"""

import logging
from typing import Optional

import dash
from dash import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from .voltage_parameter_protocol import VoltageParameterProtocol

logger = logging.getLogger(__name__)

DEFAULT_INPUT_CLASS_NAME = ""  # Standard class for inputs
LABEL_WIDTH = "200px"
INPUT_WIDTH = "150px"
UNITS_WIDTH = "80px"
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
        # Add a small epsilon for floating point inaccuracies before rounding for numbers like x.999999...
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
        param: VoltageParameterProtocol,
        # parent_component_id argument removed
    ):
        self.input_id_type = input_id_type
        self.param = param
        self.current_input_text = format_voltage(param.get_latest())
        self.input_id = {"type": self.input_id_type, "index": self.param.name}
        # dummy_blur_output_id removed

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
                        value=format_voltage(
                            self.param.get_latest()
                        ),  # Initially blank
                        n_submit=0,
                        n_blur=0,
                        debounce=False,  # Keep for responsiveness if needed, or set to True
                        className=DEFAULT_INPUT_CLASS_NAME,  # Always default
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
                # html.Div for dummy_blur_output_id removed
            ],
            align="center",
            className="mb-2 g-3",
        )

    def register_callbacks(self, app: dash.Dash) -> None:
        """Registers callbacks specific to this row's input field."""

        # Callback for when user types - only to update parent's current_input_text
        @app.callback(
            Output(
                self.input_id, "className", allow_duplicate=True
            ),  # Still output className to satisfy Dash if it's also an output elsewhere
            Input(self.input_id, "value"),
            prevent_initial_call=True,
        )
        def handle_input_text_change(text_value: Optional[str]):
            if text_value is None:
                text_value = ""
            self.current_input_text = text_value
            return DEFAULT_INPUT_CLASS_NAME

        # Callback for when user presses Enter (n_submit)
        @app.callback(
            Output(self.input_id, "value", allow_duplicate=True),
            # className output removed, periodic_update will handle it.
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
                return self.current_input_text

            except ValueError:
                logger.warning(
                    f"Invalid float: '{submitted_text_value}' for {self.param.name}"
                )
                return dash.no_update  # Keep invalid text for user to see/correct
            except Exception as e:
                logger.error(
                    f"Error during set for {self.param.name}: {e}", exc_info=True
                )
                return dash.no_update

        # Callback for when an input field loses focus (n_blur)
        @app.callback(
            Output(self.input_id, "value", allow_duplicate=True),
            # className output removed
            Input(self.input_id, "n_blur"),
            prevent_initial_call=True,
        )
        def handle_input_blur(_n_blur: int):
            if not _n_blur:
                raise PreventUpdate

            self.current_input_text = format_voltage(self.param.get_latest())
            return self.current_input_text

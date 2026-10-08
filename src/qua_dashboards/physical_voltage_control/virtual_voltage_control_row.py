"""
Layout for one relative virtual-gate control row.
"""

from dash import html
import dash_bootstrap_components as dbc

from .physical_voltage_control_row import format_voltage_fixed


class VirtualVoltageControlRow:
    """
    Example:

    vP1                 0.000000 V   -   +
    """

    def __init__(
        self,
        component_id: str,
        name: str,
        offset: float = 0.0,
        display_unit: str = "mV",

    ):
        self.component_id = component_id
        self.name = name
        self.offset = offset
        self.display_unit = display_unit

        self.increment_id = {
            "type": f"{self.component_id}-virtual-increment",
            "index": self.name,
        }

        self.decrement_id = {
            "type": f"{self.component_id}-virtual-decrement",
            "index": self.name,
        }

        self.offset_input_id = {
            "type": f"{self.component_id}-virtual-offset",
            "index": self.name,
        }

        self.unit_label_id = {
            "type": f"{self.component_id}-virtual-unit-label",
            "index": self.name,
        }

    def get_layout(self):

        display_offset = (
            self.offset * 1e3
            if self.display_unit == "mV"
            else self.offset
        )
        return dbc.Row(
            [
                # Virtual gate name
                dbc.Col(
                    dbc.Label(
                        self.name,
                        style={
                            "whiteSpace": "nowrap",
                            "marginBottom": "0",
                        },
                    ),
                    width=3,
                ),

                # Empty spacer matching the QDAC channel column
                dbc.Col(
                    html.Div(
                        "",
                        style={
                            "width": "28px",
                        },
                    ),
                    width="auto",
                ),

                # Relative virtual-gate offset
                dbc.Col(
                    dbc.Input(
                        id=self.offset_input_id,
                        type="text",
                        value=format_voltage_fixed(display_offset, 3),
                        n_submit=0,
                        debounce=False,
                        style={
                            "width": "110px",
                            "textAlign": "center",
                        },
                    ),
                    width="auto",
                ),

                # Unit
                dbc.Col(
                    dbc.Label(
                        self.display_unit,
                        id=self.unit_label_id,
                        style={
                            "whiteSpace": "nowrap",
                            "marginBottom": "0",
                        },
                    ),
                    width="auto",
                ),

                # Minus button
                dbc.Col(
                    dbc.Button(
                        "−",
                        id=self.decrement_id,
                        n_clicks=0,
                        size="sm",
                        color="secondary",
                        outline=True,
                        style={
                            "width": "36px",
                            "height": "32px",
                        },
                    ),
                    width="auto",
                ),

                # Plus button
                dbc.Col(
                    dbc.Button(
                        "+",
                        id=self.increment_id,
                        n_clicks=0,
                        size="sm",
                        color="secondary",
                        outline=True,
                        style={
                            "width": "36px",
                            "height": "32px",
                        },
                    ),
                    width="auto",
                ),
            ],
            align="center",
            className="mb-2 g-2 flex-nowrap",
        )
"""
Layout for one physical voltage-control row.
"""

from typing import Optional

import dash_bootstrap_components as dbc


DEFAULT_VOLTAGE_PRECISION = 6


def format_voltage_fixed(
    value: Optional[float],
    precision: int = DEFAULT_VOLTAGE_PRECISION,
) -> str:
    """Format voltage using a fixed number of decimal places."""

    if value is None:
        return ""

    try:
        return f"{float(value):.{precision}f}"

    except (TypeError, ValueError):
        return str(value)


class PhysicalVoltageControlRow:
    """
    Display one physical voltage parameter.

    Example
    -------
    P1_gate    ch01    0.100000 V    -  +
    """

    def __init__(
        self,
        component_id: str,
        param,
        label: Optional[str] = None,
        channel_label: Optional[str] = None,
        display_unit: str = "V",
    ):
        self.component_id = component_id
        self.param = param

        self.label = label or param.name
        self.channel_label = channel_label
        self.display_unit = display_unit

        self.value_id = {
            "type": f"{component_id}-value",
            "index": self.label,
        }

        self.unit_label_id = {
            "type": f"{component_id}-unit-label",
            "index": self.label,
        }

        self.decrement_id = {
            "type": f"{component_id}-decrement",
            "index": self.label,
        }

        self.increment_id = {
            "type": f"{component_id}-increment",
            "index": self.label,
        }

    def get_layout(self) -> dbc.Row:
        """Return the Dash layout for this row."""

        value = self.param.get_latest()

        if value is None:
            try:
                value = self.param.get()

            except Exception:
                value = None

        if value is None:
            display_value = None
        elif self.display_unit == "mV":
            display_value = float(value) * 1e3
        else:
            display_value = float(value)

        precision = 3 if self.display_unit == "mV" else 6

        return dbc.Row(
            [
                # Logical gate name
                dbc.Col(
                    dbc.Label(
                        self.label,
                        style={
                            "whiteSpace": "nowrap",
                            "marginBottom": "0",
                        },
                    ),
                    width=3,
                ),

                # QDAC channel
                dbc.Col(
                    dbc.Label(
                        self.channel_label or "",
                        style={
                            "color": "#A0A0A0",
                            "fontSize": "0.80rem",
                            "whiteSpace": "nowrap",
                            "marginBottom": "0",
                        },
                    ),
                    width="auto",
                ),

                # Current absolute voltage
                dbc.Col(
                    dbc.Input(
                        id=self.value_id,
                        type="text",
                        value=format_voltage_fixed(display_value, precision=precision),
                        readonly=True,
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
                        "-",
                        id=self.decrement_id,
                        n_clicks=0,
                        size="sm",
                        outline=True,
                        color="secondary",
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
                        outline=True,
                        color="secondary",
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

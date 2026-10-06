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
    ):
        self.component_id = component_id
        self.param = param

        self.label = label or param.name
        self.channel_label = channel_label

        self.value_id = {
            "type": f"{component_id}-value",
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
                        className="text-muted",
                        style={
                            "fontSize": "0.85rem",
                            "whiteSpace": "nowrap",
                            "marginBottom": "0",
                        },
                    ),
                    width=1,
                ),

                # Current absolute voltage
                dbc.Col(
                    dbc.Input(
                        id=self.value_id,
                        type="text",
                        value=format_voltage_fixed(value),
                        readonly=True,
                    ),
                    width=4,
                ),

                # Unit
                dbc.Col(
                    dbc.Label(
                        "V",
                        style={
                            "whiteSpace": "nowrap",
                            "marginBottom": "0",
                        },
                    ),
                    width=1,
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
                    width=1,
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
                    width=1,
                ),
            ],
            align="center",
            className="mb-2 g-2 flex-nowrap",
        )
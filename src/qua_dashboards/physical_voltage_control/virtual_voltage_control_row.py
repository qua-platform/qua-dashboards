"""
Layout for one relative virtual-gate control row.
"""

import dash_bootstrap_components as dbc


class VirtualVoltageControlRow:
    """
    Example:

    vP1        -   +
    """

    def __init__(self, component_id: str, name: str):
        self.component_id = component_id
        self.name = name

        self.decrement_id = {
            "type": f"{component_id}-virtual-decrement",
            "index": name,
        }

        self.increment_id = {
            "type": f"{component_id}-virtual-increment",
            "index": name,
        }

    def get_layout(self):
        return dbc.Row(
            [
                dbc.Col(
                    dbc.Label(
                        self.name,
                        style={
                            "whiteSpace": "nowrap",
                            "marginBottom": "0",
                        },
                    ),
                    width=8,
                ),

                dbc.Col(
                    dbc.Button(
                        "−",
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
                    width=2,
                ),

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
                    width=2,
                ),
            ],
            align="center",
            className="mb-2 g-2 flex-nowrap",
        )

from dash import dcc, html
import dash_bootstrap_components as dbc


def get_gate_selection_ui(gateset):
    channel_names = list(gateset.channels.keys())
    for layer in gateset.layers:
        channel_names = channel_names + layer.source_gates
    return dbc.Card([
        dbc.CardHeader("Sweep Channel Selection"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("X Channel"),
                    dcc.Dropdown(
                        id="sweep-x-channel",
                        options=[{"label": name, "value": name} for name in channel_names],
                        value=channel_names[0]  
                    ),
                ]),
                dbc.Col([
                    dbc.Label("Y Channel"),
                    dcc.Dropdown(
                        id="sweep-y-channel",
                        options=[{"label": name, "value": name} for name in channel_names],
                        value=channel_names[1] if len(channel_names) > 1 else channel_names[0]
                    ),
                ]),
            ]),
            dbc.Button("Set Sweep Channels", id="set-sweep-channels", color="primary"),
        ]),
    ], className="mb-3")
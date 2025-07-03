import html
from dash import dcc
import dash_bootstrap_components as dbc

# def get_gate_selection_ui(gateset):
#     options = [{"label": k, "value": k} for k in gateset.channels.keys()]
#     return html.Div([
#         html.Label("Select X sweep gate"),
#         dcc.Dropdown(
#             id="gate-sweep-x",
#             options=options,
#             value=options[0]["value"],
#             clearable=False,
#         ),
#         html.Label("Select Y sweep gate"),
#         dcc.Dropdown(
#             id="gate-sweep-y",
#             options=options,
#             value=options[1]["value"] if len(options) > 1 else options[0]["value"],
#             clearable=False,
#         ),
#         dcc.Store(id="gate-sweep-selection"),
#     ], style={"marginBottom": 20})

def get_gate_selection_ui(gateset):
    # List of all possible gate/channel names
    channel_names = list(gateset.channels.keys())
    return dbc.Card([
        dbc.CardHeader("Sweep Channel Selection"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("X Channel"),
                    dcc.Dropdown(
                        id="sweep-x-channel",
                        options=[{"label": name, "value": name} for name in channel_names],
                        value=channel_names[0]  # Default
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
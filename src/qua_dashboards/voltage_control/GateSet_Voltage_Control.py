import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from qua_dashboards.core import BaseUpdatableComponent

class GateSetControl(BaseUpdatableComponent):
    def __init__(self, gateset, component_id="Gate-Set QDAC Offset Controller"):
        super().__init__(component_id=component_id)
        self.gateset = gateset

    def get_layout(self):
        # build one input per channel
        inputs = []
        current = self.gateset.get_voltages()
        for ch in self.gateset.channels:
            inputs.append(
                dbc.Row([
                    dbc.Label(ch, width="auto"),
                    dbc.Col(
                        dcc.Input(
                            id=self._get_id(f"voltage-{ch}"),
                            type="number",
                            value=current[ch],
                            min=-5, max=5, step=0.001,
                        ),
                        width=4
                    ),
                ], className="mb-2")
            )

        # Apply button + hidden Store
        return html.Div([
            dbc.Card([
                dbc.CardHeader("Gate Voltages"),
                dbc.CardBody(inputs + [
                    dbc.Button("Apply", id=self._get_id("apply"), color="primary"),
                ])
            ]),
            dcc.Store(id=self._get_id("store"))
        ])

    def register_callbacks(self, app: dash.Dash):
        # on Apply click, read all inputs and call apply_voltages
        states = [
            State(self._get_id(f"voltage-{ch}"), "value")
            for ch in self.gateset.channels
        ]
        @app.callback(
            Output(self._get_id("store"), "data"), 
            Input(self._get_id("apply"), "n_clicks"),
            *states,
            prevent_initial_call=True,
        )
        def _on_apply(n_clicks, *vals):
            # map channel names → values
            volt_dict = {
                ch: v for ch, v in zip(self.gateset.channels, vals)
            }
            self.gateset.apply_voltages(volt_dict, allow_extra_entries=False)
            return dash.no_update
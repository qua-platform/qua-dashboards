import dash_bootstrap_components as dbc
import dash
from dash import html, dcc, Input, Output, State, ALL, no_update
from dash.exceptions import PreventUpdate
import numpy as np

class VirtualLayerAdder:
    """
    A component allowing the addition of new virtualisation layers using VirtualGateSet. Renders an NxN editable matrix (default identity) with editable column names. 
    An apply button calls gateset.add_layer(..) internally. 
    """

    def __init__(self, gateset, component_id: str):
        """
        gateset: your VirtualGateSet instance
        component_id: a unique string to namespace this editor
        """
        self.gateset = gateset
        self.component_id = component_id
        self.apply_button_id = f"{self.component_id}-apply-button"
        self.layout_columns = 9
        self.layout_rows    = 4 

    def _matrix_builder(self, source_gates: list[str], target_gates: list[str]):
        """
        Returns a list of dbc.Row children for an MxK matrix editor
        """
        K = len(source_gates)
        M = len(target_gates)

        header = [ dbc.Col(html.B("→ / ↓"), width = 2) ] + [
            dbc.Col(dcc.Input(
                id = {"type":"vg-source-column", "index":self.component_id, "col": j}, 
                value = source_gates[j], style = {"width":"100%", "textAlign":"center"}
            ), width = 2) for j in range(K)
        ]

        rows = []
        for i, row_name in enumerate(target_gates):
            cells = [dbc.Col(html.B(row_name), width = 2)] 
            for j in range(K):
                default_value = 1.0 if (i==j) else 0.0
                cells.append(
                    dbc.Col(dcc.Input(
                        id = {"type":"vg-layer-cell", 
                              "index":self.component_id,
                              "row":i,"col":j},
                              type = "number",
                              value = default_value, 
                              style = {"width":"100%", "textAlign":"center"}
                              
                    ), width = 2)
                )
            rows.append(dbc.Row(cells, className="mb-1"))
        return [dbc.Row(header, className="mb-2"), *rows]

    def get_layout(self):
        if self.gateset.layers:
            assigned = {target for layer in self.gateset.layers for target in layer.target_gates}
            full_list = [name for name in self.gateset.valid_channel_names if name not in assigned]
        else:
            full_list = list(self.gateset.channels.keys())
        
        dropdown_options = [{"label":n,"value":n} for n in sorted(set(full_list))]

        return dbc.Card([
                    dbc.CardHeader("Virtual Gate-Set Layer Adder"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col(html.Label("Target gates"), width=2),
                            dbc.Col(dcc.Dropdown(
                                    id=f"{self.component_id}-tgt-dd",
                                    options=dropdown_options, multi=True, value=[], placeholder = "Pick Target Gates!"),
                                width=4),
                        ], className="mb-3"),

                        html.Div(id=f"{self.component_id}-matrix-container"),


                        dbc.Button("Apply", id=self.apply_button_id, color="primary", className="mt-2"),
                        html.Div(
                            id={"type": "LAYER_OUTPUT", "index": self.component_id},
                            style={"display": "none"},
                        ),
                        dcc.Store(id={"type": "LAYER_REFRESH", "index": self.component_id}, data=0)
                    ])
                    ], className="mb-4")

    def register_callbacks(self, app):
        @app.callback(
            Output(f"{self.component_id}-matrix-container", "children"),
            Input(f"{self.component_id}-tgt-dd", "value"),
        )
        def _rebuild_matrix(target_list):
            if not target_list:
                return html.Div("Pick target")
            return self._matrix_builder(
                source_gates = [""] * len(target_list),
                target_gates=target_list
            )
        @app.callback(
            Output({"type":"LAYER_OUTPUT","index":self.component_id}, "children", allow_duplicate=True),
            Output({"type":"LAYER_REFRESH","index":self.component_id}, "data", allow_duplicate=True),
            Output("vg-layer-refresh-trigger", "data", allow_duplicate=True),
            Input(self.apply_button_id, "n_clicks"),
            State(f"{self.component_id}-tgt-dd", "value"),
            State({"type":"vg-source-column", "index":self.component_id, "col": ALL}, "value"),
            State({"type":"vg-layer-cell",   "index":self.component_id, "row": ALL, "col": ALL}, "value"),
            State({"type":"LAYER_REFRESH",    "index":self.component_id}, "data"),
            State("vg-layer-refresh-trigger","data"),
            prevent_initial_call=True
        )
        def _apply(nc, target_list, source_names, flat_cells, refresh, glob_ref):
            if nc is None or not target_list:
                raise PreventUpdate

            M = len(target_list)
            K = len(source_names)
            if len(flat_cells) != M*K:
                app.logger.error(f"Wrong shape: Expected {M*K} cells, got {len(flat_cells)}")
                return no_update, no_update, no_update

            mat = [flat_cells[i*K:(i+1)*K] for i in range(M)]
            try:
                self.gateset.add_layer(
                    source_gates=source_names,
                    target_gates=target_list,
                    matrix=mat
                )
                app.logger.info(f"Added layer {source_names} -> {target_list}")
            except Exception as e:
                app.logger.error(f"add_layer failed: {e}")
                return f"Error: {e}", no_update, no_update

            return "Layer added!", (refresh or 0)+1, (glob_ref or 0)+1
        
        @app.callback(
            Output(f"{self.component_id}-tgt-dd", "options"),
            Output(f"{self.component_id}-tgt-dd", "value"),
            Input("vg-layer-refresh-trigger", "data"),
            prevent_initial_call=True,
        )
        def _refresh_target_dropdown(_):
            if self.gateset.layers:
                assigned = {t for layer in self.gateset.layers for t in layer.target_gates}
                full_list = [n for n in self.gateset.valid_channel_names if n not in assigned]
            else:
                full_list = list(self.gateset.channels.keys())
            options = [{"label": n, "value": n} for n in sorted(set(full_list))]
            return options, []
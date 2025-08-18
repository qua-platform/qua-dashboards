import dash_bootstrap_components as dbc
import dash
from dash import html, dcc, Input, Output, State, ALL, no_update
from dash.exceptions import PreventUpdate
import numpy as np
class VirtualLayerAdder:
    """
    A Dash-style editor for adding a new virtualisation layer to a VirtualQdacGateSet.
    Renders an NxN editable matrix (default = identity), with editable row and column names,
    and an “Apply” button that calls gateset.add_layer(...) under the hood.
    """

    def __init__(self, gateset, component_id: str):
        """
        gateset: your VirtualQdacGateSet instance
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
            # Input(f"{self.component_id}-n-src",  "value"),
        )
        def _rebuild_matrix(target_list):
            if not target_list:
                return html.Div("Pick target")
            return self._matrix_builder(
                source_gates = [""] * len(target_list),
                target_gates=target_list
            )
        @app.callback(
            Output({"type":"LAYER_OUTPUT","index":self.component_id}, "children"),
            Output({"type":"LAYER_REFRESH","index":self.component_id}, "data"),
            Output("vg-layer-refresh-trigger", "data"),
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
                return no_update

            mat = [flat_cells[i*K:(i+1)*K] for i in range(M)]
            try:
                self.gateset.add_layer(
                    source_gates=source_names,
                    target_gates=target_list,
                    matrix=mat
                )
                app.logger.info(f"Added layer {source_names}→{target_list}")
            except Exception as e:
                app.logger.error(f"add_layer failed: {e}")

            return no_update, (refresh or 0)+1, (glob_ref or 0)+1

from qua_dashboards.core import BaseComponent
class VirtualLayerManager(BaseComponent):
    def __init__(self,gateset,component_id = "VG_manager"):
        super().__init__(component_id = component_id)
        self.gateset = gateset
        self.component_id = component_id
    
    def _render_matrix_editor(self, layer_idx):
        layer = self.gateset.layers[layer_idx]
        N = len(layer.source_gates)
        header = [dbc.Col("", width=2)] + [
            dbc.Col(html.B(name), width=2) for name in layer.source_gates
        ]
        rows = []
        for i, row_name in enumerate(layer.target_gates):
            row = [dbc.Col(html.B(row_name), width=2)]
            for j in range(N):
                row.append(
                    dbc.Col(
                        dcc.Input(
                            id={"type": "edit-matrix-cell", "layer": layer_idx, "row": i, "col": j},
                            type="number",
                            value=layer.matrix[i][j],
                        ),
                        width=2,
                    )
                )
            rows.append(dbc.Row(row, className="mb-1"))
        return html.Div([dbc.Row(header, className="mb-2"), *rows])


    def get_layout(self):
        layers = [layer for layer in self.gateset.layers]

        num_of_layers = len(layers)
        options = [{"label":f"Layer {i+1}", "value":i} for i in range(num_of_layers)]
        default = 0 if num_of_layers > 0 else None
        return dbc.Card([
            dbc.CardHeader("Edit Existing Virtual Gate Layers"),
            dbc.CardBody([
                dcc.Dropdown(
                    id = f"{self.component_id}-layer-dropdown",
                    options = options,
                    value = default,
                    clearable = False
                ),
                html.Div(id = f"{self.component_id}-layer-matrix-editor"),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button(
                                "Apply Changes",
                                id=f"{self.component_id}-apply-edit",
                                color="primary",
                                className="mt-2 w-100",
                            )
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Reset to Identity",
                                id=f"{self.component_id}-identity-reset",
                                color="secondary",
                                className="mt-2 w-100",
                            )
                        ),
                    ],
                    className="g-2",
                ),                
                html.Div(id=f"{self.component_id}-edit-output", style={"display": "none"})
            ])
        ])


    def register_callbacks(self, app):
        @app.callback(
            Output(f"{self.component_id}-layer-matrix-editor", "children"), 
            Input(f"{self.component_id}-layer-dropdown", "value")
        )

        def show_selected_layer(layer_idx):
            if layer_idx is None: 
                return "No Layer Selected"
            
            return self._render_matrix_editor(layer_idx)


        @app.callback(
            Output(f"{self.component_id}-edit-output", "children"),
            Input(f"{self.component_id}-apply-edit", "n_clicks"),
            State(f"{self.component_id}-layer-dropdown", "value"), 
            State({'type': 'edit-matrix-cell', 'layer': ALL, 'row': ALL, 'col': ALL}, 'value'),
            prevent_initial_call=True
        )
        def apply_layer_changes(n_clicks, layer_idx, matrix_flat):
            if not n_clicks:
                raise dash.exceptions.PreventUpdate
            
            layer = self.gateset.layers[layer_idx]
            length = len(layer.source_gates)
            M = np.reshape(np.array(matrix_flat), (length, length)).tolist()
            layer.matrix = M
            return "Updated!"

        @app.callback(
            Output({'type': 'edit-matrix-cell', 'layer': ALL, 'row': ALL, 'col': ALL}, 'value'),
            Input(f"{self.component_id}-identity-reset", 'n_clicks'),
            State(f"{self.component_id}-layer-dropdown", 'value'),
            prevent_initial_call=True
        )
        def _reset_to_identity(n_clicks, layer_idx):
            if not n_clicks or layer_idx is None:
                raise PreventUpdate
            layer = self.gateset.layers[layer_idx]
            N = len(layer.source_gates)

            flat_id = []
            for i in range(N):
                for j in range(N):
                    flat_id.append(1.0 if i == j else 0.0)

            return flat_id
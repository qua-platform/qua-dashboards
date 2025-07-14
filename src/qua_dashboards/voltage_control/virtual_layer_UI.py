import dash_bootstrap_components as dbc
import dash
from dash import html, dcc, Input, Output, State, ALL, no_update
from dash.exceptions import PreventUpdate
import numpy as np
class VirtualLayerEditor:
    """
    A Dash‐style editor for adding a new virtualisation layer to a VirtualQdacGateSet.
    Renders an N×N editable matrix (default = identity), with editable column names,
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

    def get_layout(self):
        if self.gateset.layers:
            target_gates = self.gateset.layers[-1].source_gates
        else:
            target_gates = list(self.gateset.channels.keys())
        N = len(target_gates)

        header_cells = [dbc.Col(html.B("→ / ↓"), width=2)]
        for col_idx, col_name in enumerate(target_gates):
            header_cells.append(
                dbc.Col(
                    dcc.Input(
                        id={"type": "vg-layer-col", "index": self.component_id, "col": col_idx},
                        value=col_name,
                        style={
                            "width": "100%",
                            "fontWeight": "bold",
                            "color": "black",
                            "textAlign": "center",
                        },
                    ),
                    width=2,
                )
            )
        header_row = dbc.Row(header_cells, className="mb-2")

        data_rows = []
        for row_idx, row_name in enumerate(target_gates):
            row_cells = [
                dbc.Col(html.B(row_name), width=2),
            ]
            for col_idx in range(N):
                default = 1.0 if row_idx == col_idx else 0.0
                row_cells.append(
                    dbc.Col(
                        dcc.Input(
                            id={
                                "type": "vg-layer-cell",
                                "index": self.component_id,
                                "row": row_idx,
                                "col": col_idx,
                            },
                            type="number",
                            value=default,
                            style={
                                "width": "100%",
                                "textAlign": "center",
                                "color": "black",
                            },
                        ),
                        width=2,
                    )
                )
            data_rows.append(dbc.Row(row_cells, className="mb-1"))

        return dbc.Card(
            [
                dbc.CardHeader("Virtual Gate‐Set Layer Adder"),
                dbc.CardBody(
                    [
                        header_row,
                        *data_rows,
                        dbc.Button(
                            "Apply",
                            id=self.apply_button_id,
                            color="primary",
                            className="mt-2",
                        ),
                        html.Div(
                            id={"type": "LAYER_OUTPUT", "index": self.component_id},
                            style={"display": "none"},
                        ),
                        dcc.Store(id={"type": "LAYER_REFRESH", "index": self.component_id}, data=0)
                    ]
                ),
            ],
            className="mb-4",
        )

    def register_callbacks(self, app):
        @app.callback(
            Output({"type": "LAYER_OUTPUT", "index": self.component_id}, "children"),
            Output({"type": "LAYER_REFRESH", "index": self.component_id}, "data"),
            Output("vg-layer-refresh-trigger", "data"),
            Input(self.apply_button_id, "n_clicks"),
            State({"type": "vg-layer-col", "index": self.component_id, "col": ALL}, "value"),
            State(
                {"type": "vg-layer-cell", "index": self.component_id, "row": ALL, "col": ALL},
                "value",
            ),
            State({"type": "LAYER_REFRESH", "index": self.component_id}, "data"),
            State("vg-layer-refresh-trigger", "data"),
            prevent_initial_call=True,
        )
        def _apply_new_layer(n_clicks, col_names, cell_matrix, refresh_val, global_refresh):
            if not n_clicks:
                raise PreventUpdate

            if self.gateset.layers:
                target_gates = list(self.gateset.layers[-1].source_gates)
            else:
                target_gates = list(self.gateset.channels.keys())
            N = len(target_gates)

            if len(cell_matrix) != N * N:
                app.logger.error(
                    f"[vg-layer] Expected {N} cols & {N*N} cells, got "
                    f"{len(col_names)} cols & {len(cell_matrix)} cells"
                )
                return no_update
            matrix = []
            for i in range(N):
                row = [float(cell_matrix[i*N + j]) for j in range(N)]
                matrix.append(row)
            try:
                self.gateset.add_layer(
                    source_gates=col_names,
                    target_gates=target_gates,
                    matrix=matrix,
                )
                app.logger.info(f"[vg-layer] Added layer: {col_names} → {target_gates}")
            except Exception as e:
                app.logger.error(f"[vg-layer] Failed to add layer: {e}")

            return no_update, (refresh_val or 0) + 1, (global_refresh or 0) + 1

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
        
        # @app.callback(
        #     Output(f"{self.component_id}-layer-matrix-editor", "children", allow_duplicate=True),
        #     Input(f"{self.component_id}-identity-reset", "n_clicks"),
        #     State(f"{self.component_id}-layer-dropdown", "value"),
        #     prevent_initial_call=True,
        # )
        # def _reset_to_identity(n_clicks, layer_idx):
        #     if not n_clicks:
        #         raise PreventUpdate
        #     layer = self.gateset.layers[layer_idx]
        #     N = len(layer.source_gates)
        #     layer.matrix = np.eye(N).tolist()
        #     return self._render_matrix_editor(layer_idx)
  

        @app.callback(
            # one Output per cell input, matching the same ALL selector you used for apply
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

            # build a flat list of identity entries:
            flat_id = []
            for i in range(N):
                for j in range(N):
                    flat_id.append(1.0 if i == j else 0.0)

            # return that list as the new .value for each matching input
            return flat_id
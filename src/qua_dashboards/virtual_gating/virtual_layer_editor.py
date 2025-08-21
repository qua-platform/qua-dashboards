import dash_bootstrap_components as dbc
import dash
from dash import html, dcc, Input, Output, State, ALL, no_update
from dash.exceptions import PreventUpdate
import numpy as np
from qua_dashboards.core import BaseComponent
from .virtual_layer_adder import VirtualLayerAdder


class VirtualLayerEditor(BaseComponent):
    """
    A dash component allowing the editing of virtual layers to work with VirtualGateSet. Renders the matrix as an NxN editable table, complete with a reset_to_identity button. 
    An apply button changes the VirtualGateSet.layer matrix. 
    """


    def __init__(self,gateset,component_id = "VG_manager"):
        """
        Args:
            gateset: The VirtualGateSet instance
            component_id: Dash Component ID; also the name of the component title
        """
        super().__init__(component_id = component_id)
        self.gateset = gateset
        self.component_id = component_id
        self._adder = VirtualLayerAdder(self.gateset, component_id = f"{component_id}-adder")
    
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

        editor_card = dbc.Card([
            dbc.CardHeader("VirtualGateSet Layer Editor"),
            dbc.CardBody([
                dcc.Dropdown(
                    id = f"{self.component_id}-layer-dropdown",
                    options = options,
                    value = default,
                    clearable = False
                ),
                html.Div(id = f"{self.component_id}-layer-matrix-editor"),
                dcc.Interval(
                    id=f"{self.component_id}-init",
                    interval=50, n_intervals=0, max_intervals=1
                ),
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

        return html.Div([
            self._adder.get_layout(), 
            editor_card, 
        ])


    def register_callbacks(self, app):
        self._adder.register_callbacks(app)

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
            Output(f"{self.component_id}-layer-matrix-editor", "children", allow_duplicate=True),
            Input(f"{self.component_id}-identity-reset", 'n_clicks'),
            State(f"{self.component_id}-layer-dropdown", 'value'),
            prevent_initial_call=True
        )
        def reset_to_identity_and_refresh(n_clicks, layer_idx):
            if not n_clicks or layer_idx is None:
                raise PreventUpdate
            
            layer = self.gateset.layers[layer_idx]
            N = len(layer.source_gates)
            identity = [[1.0 if i == j else 0.0 for j in range(N)] for i in range(N)]
            layer.matrix = identity
            
            return self._render_matrix_editor(layer_idx)

        @app.callback(
            Output(f"{self.component_id}-layer-dropdown", "options"),
            Output(f"{self.component_id}-layer-dropdown", "value"),
            Input("vg-layer-refresh-trigger", "data"),
            Input(f"{self.component_id}-init", "n_intervals"),
            prevent_initial_call=True,
        )
        def _refresh_dropdown(_refresh, _init):
            num = len(self.gateset.layers)
            options = [{"label": f"Layer {i+1}", "value":i} for i in range(num)]

            val = num - 1 if num >0 else None
            return options, val
        
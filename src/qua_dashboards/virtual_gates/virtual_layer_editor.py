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


    def __init__(self,gateset,component_id = "VG_manager", dc_set = None):
        """
        Args:
            gateset: The VirtualGateSet instance
            component_id: Dash Component ID; also the name of the component title
        """
        super().__init__(component_id = component_id)
        self.gateset = gateset
        self.dc_set = dc_set
        self.component_id = component_id
        self.layer_names = [
            {"label": f"Layer {i+1}", "value": i}
            for i, _ in enumerate(self.gateset.layers)
        ]
        self._adder = VirtualLayerAdder(self.gateset, component_id = f"{component_id}-adder", layer_names=self.layer_names, dc_set=dc_set)

    @property
    def active_target(self): 
        """Returns gateset or dc_set based on current selection. Default is gateset"""
        return getattr(self, "_current_target_set", self.gateset)
    
    def _render_matrix_editor(self, layer_idx, use_dc = False):
        target_set = self.dc_set if (use_dc and self.dc_set) else self.gateset 
        if layer_idx >= len(target_set.layers): 
            return html.Div("Layer not available for this target")
        layer = target_set.layers[layer_idx]
        num_sources = len(layer.source_gates)
        num_targets = len(layer.target_gates)
        col_width = 12 // (num_sources + 1)
        header = [dbc.Col("", width=col_width)] + [
            dbc.Col(html.B(name), width=col_width) for name in layer.source_gates
        ]
        rows = []
        for i, row_name in enumerate(layer.target_gates):
            row = [dbc.Col(html.B(row_name), width=col_width)]
            for j in range(num_sources):
                row.append(
                    dbc.Col(
                        dcc.Input(
                            id={"type": "edit-matrix-cell", "layer": layer_idx, "row": i, "col": j},
                            type="number",
                            value=layer.matrix[i][j],
                            style={"width": "auto"}
                        ),
                        width=col_width,
                    )
                )
            rows.append(dbc.Row(row, className="mb-1"))
        return html.Div([dbc.Row(header, className="mb-2"), *rows])


    def get_layout(self):
        layers = [layer for layer in self.gateset.layers]
        num_of_layers = len(layers)
        options = self.layer_names
        default = 0 if num_of_layers > 0 else None

        opx_dc_toggle_switch = dbc.Row([
            dbc.Col(html.Label("OPX", className="me-2"), width="auto"),
            dbc.Col(
                dbc.Switch(
                    id=f"{self.component_id}-target-switch",
                    value=False,  # False = OPX, True = DC
                    className="mt-1",
                ),
                width="auto",
            ),
            dbc.Col(html.Label("DC"), width="auto"),
            dbc.Col(
                dbc.Checkbox(
                    id=f"{self.component_id}-edit-both",
                    label="Edit both",
                    value=False,
                    className="ms-4",
                ),
                width="auto",
            ),
        ], className="mb-3 align-items-center") if self.dc_set is not None else html.Div()

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
            opx_dc_toggle_switch,
            editor_card, 
        ])


    def register_callbacks(self, app):
        self._adder.register_callbacks(app)

        @app.callback(
            Output(f"{self.component_id}-layer-matrix-editor", "children"),
            Output(f"{self.component_id}-layer-dropdown", "options"),
            Output(f"{self.component_id}-layer-dropdown", "value"),
            Input(f"{self.component_id}-layer-dropdown", "value"),
            Input(f"{self.component_id}-target-switch", "value") if self.dc_set else Input(f"{self.component_id}-init", "n_intervals"),
            Input("vg-layer-refresh-trigger", "data"),
            Input(f"{self.component_id}-init", "n_intervals"),
        )
        def show_selected_layer(layer_idx, use_dc, _refresh, _init):
            if self.dc_set is None:
                use_dc = False
            
            target_set = self.dc_set if (use_dc and self.dc_set) else self.gateset
            options = [{"label": f"Layer {i+1}", "value": i} for i in range(len(target_set.layers))]

            if not options:
                return "No Layers", [], None
            if layer_idx is None or layer_idx >= len(target_set.layers):
                layer_idx = 0

            return self._render_matrix_editor(layer_idx, use_dc), options, layer_idx


        @app.callback(
            Output(f"{self.component_id}-edit-output", "children"),
            Output("vg-layer-refresh-trigger", "data", allow_duplicate=True),
            Input(f"{self.component_id}-apply-edit", "n_clicks"),
            State(f"{self.component_id}-layer-dropdown", "value"), 
            State(f"{self.component_id}-target-switch", "value") if self.dc_set else State(f"{self.component_id}-layer-dropdown", "value"),
            State(f"{self.component_id}-edit-both", "value") if self.dc_set else State(f"{self.component_id}-layer-dropdown", "value"),
            State({'type': 'edit-matrix-cell', 'layer': ALL, 'row': ALL, 'col': ALL}, 'value'),
            prevent_initial_call=True
        )
        def apply_layer_changes(n_clicks, layer_idx, use_dc, edit_both, matrix_flat):
            if not n_clicks:
                raise dash.exceptions.PreventUpdate
            if self.dc_set is None:
                edit_both = False
                use_dc = False

            target_set = self.dc_set if (use_dc and self.dc_set) else self.gateset
            layer = target_set.layers[layer_idx]
            num_sources = len(layer.source_gates)
            num_targets = len(layer.target_gates)
            M = np.reshape(np.array(matrix_flat), (num_sources, num_targets)).tolist()
            if edit_both:
                if layer_idx < len(self.gateset.layers):
                    self.gateset.layers[layer_idx].matrix = [row[:] for row in M]
                if layer_idx < len(self.dc_set.layers):
                    self.dc_set.layers[layer_idx].matrix = [row[:] for row in M]
            else:
                target_set.layers[layer_idx].matrix = M
            return "Updated!", dash.no_update if False else 1
        
        @app.callback(
            Output(f"{self.component_id}-layer-matrix-editor", "children", allow_duplicate=True),
            Output("vg-layer-refresh-trigger", "data", allow_duplicate=True),
            Input(f"{self.component_id}-identity-reset", 'n_clicks'),
            State(f"{self.component_id}-layer-dropdown", 'value'),
            State(f"{self.component_id}-target-switch", "value") if self.dc_set else State(f"{self.component_id}-layer-dropdown", "value"),
            State(f"{self.component_id}-edit-both", "value") if self.dc_set else State(f"{self.component_id}-layer-dropdown", "value"),
            State("vg-layer-refresh-trigger", "data"),
            prevent_initial_call=True
        )
        def reset_to_identity_and_refresh(n_clicks, layer_idx, use_dc, edit_both, vg_val):
            if not n_clicks or layer_idx is None:
                raise PreventUpdate
            
            if self.dc_set is None:
                edit_both = False
                use_dc = False
            
            target_set = self.dc_set if (use_dc and self.dc_set) else self.gateset
            layer = target_set.layers[layer_idx]
            num_sources = len(layer.source_gates)
            num_targets = len(layer.target_gates)
            identity = [[1.0 if i == j else 0.0 for j in range(num_targets)] for i in range(num_sources)]
            if edit_both:
                if layer_idx < len(self.gateset.layers):
                    self.gateset.layers[layer_idx].matrix = [row[:] for row in identity]
                if layer_idx < len(self.dc_set.layers):
                    self.dc_set.layers[layer_idx].matrix = [row[:] for row in identity]
            else:
                target_set.layers[layer_idx].matrix = identity
            
            return self._render_matrix_editor(layer_idx), (vg_val or 0) + 1

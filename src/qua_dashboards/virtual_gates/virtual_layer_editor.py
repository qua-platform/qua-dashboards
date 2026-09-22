import math

import dash_bootstrap_components as dbc
import dash
from dash import html, dcc, Input, Output, State, ALL, no_update
from dash.exceptions import PreventUpdate
import numpy as np
from qua_dashboards.core import BaseComponent
from .virtual_layer_adder import VirtualLayerAdder

# Input column width. 1rem is the dashboard root font size.
_MATRIX_COL_REM = 6


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
        self.layout_rows = self._estimate_layout_rows()

    def _estimate_layout_rows(self) -> int:
        """Grid rows tall enough for the largest layer, plus the adder and editor chrome."""
        counts = []
        for gateset in (self.gateset, self.dc_set):
            layers = getattr(gateset, "layers", None) or []
            counts.extend(len(getattr(layer, "source_gates", [])) for layer in layers)
        n_rows = max(counts, default=0)
        row_px = 2 * 16 + 4
        matrix_px = 9 * 16 + n_rows * row_px
        needed = matrix_px + 520
        # item height = 100*h + 10*(h-1); the tile content is capped at 95%.
        return max(10, math.ceil((needed + 30) / 104.5))

    @property
    def active_target(self): 
        """Returns gateset or dc_set based on current selection. Default is gateset"""
        return getattr(self, "_current_target_set", self.gateset)
    
    def _render_matrix_editor(self, layer_idx, use_dc = False):
        target_set = self.dc_set if (use_dc and self.dc_set) else self.gateset 
        if layer_idx >= len(target_set.layers): 
            return html.Div("Layer not available for this target")
        layer = target_set.layers[layer_idx]
        col = f"{_MATRIX_COL_REM}rem"
        n_cols = len(layer.target_gates)
        header_label = {
            "writingMode": "vertical-rl",
            "transform": "rotate(180deg)",
            "whiteSpace": "nowrap",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
            "maxHeight": "9rem",
            "fontSize": "0.75rem",
            "lineHeight": "1.1",
            "fontWeight": "600",
        }
        cells = [
            html.Div(
                html.Span(name, title=name, style=header_label),
                style={
                    "width": col,
                    "minWidth": "0",
                    "display": "flex",
                    "justifyContent": "center",
                    "alignItems": "flex-end",
                    "alignSelf": "end",
                    "overflow": "hidden",
                },
            )
            for name in layer.target_gates
        ]
        for i, row_name in enumerate(layer.source_gates):
            cells.append(
                html.Div(
                    row_name,
                    style={
                        "whiteSpace": "nowrap",
                        "textAlign": "right",
                        "paddingRight": "8px",
                        "fontWeight": "600",
                    },
                )
            )
            for j in range(n_cols):
                cells.append(
                    dcc.Input(
                        id={
                            "type": "edit-matrix-cell",
                            "layer": layer_idx,
                            "row": i,
                            "col": j,
                        },
                        type="number",
                        value=layer.matrix[i][j],
                        className="vg-matrix-input",
                        style={
                            "width": col,
                            "minWidth": "0",
                            "boxSizing": "border-box",
                            "textAlign": "center",
                            "padding": "0 4px",
                            "display": "block",
                            "fontSize": "0.85rem",
                        },
                    )
                )
        return html.Div(
            html.Div(
                [html.Div(), *cells],
                style={
                    "display": "grid",
                    "gridTemplateColumns": f"max-content repeat({n_cols}, {col})",
                    "columnGap": "4px",
                    "rowGap": "4px",
                    "alignItems": "center",
                    "justifyItems": "center",
                    "width": "max-content",
                },
            ),
            style={"overflow": "auto"},
        )


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

        return html.Div(
            [
                self._adder.get_layout(),
                opx_dc_toggle_switch,
                editor_card,
            ],
            className="vg-layer-editor",
        )


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
            State({'type': 'edit-matrix-cell', 'layer': ALL, 'row': ALL, 'col': ALL}, 'id'),
            State("vg-layer-refresh-trigger", "data"),
            prevent_initial_call=True
        )
        def apply_layer_changes(n_clicks, layer_idx, use_dc, edit_both, matrix_flat, cell_ids, vg_val):
            if not n_clicks:
                raise dash.exceptions.PreventUpdate
            if self.dc_set is None:
                edit_both = False
                use_dc = False

            target_set = self.dc_set if (use_dc and self.dc_set) else self.gateset
            layer = target_set.layers[layer_idx]
            num_sources = len(layer.source_gates)
            num_targets = len(layer.target_gates)
            coord_vals = {(cid["row"], cid["col"]): v for cid, v in zip(cell_ids, matrix_flat)}
            M = [
                [float(coord_vals.get((i, j), 0.0) or 0.0) for j in range(num_targets)]
                for i in range(num_sources)
            ]
            if edit_both:
                if layer_idx < len(self.gateset.layers):
                    self.gateset.layers[layer_idx].matrix = [row[:] for row in M]
                if self.dc_set is not None and layer_idx < len(self.dc_set.layers):
                    self.dc_set.layers[layer_idx].matrix = [row[:] for row in M]
            else:
                target_set.layers[layer_idx].matrix = M
            return "Updated!", (vg_val or 0) + 1
        
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
            
            return self._render_matrix_editor(layer_idx, use_dc), (vg_val or 0) + 1

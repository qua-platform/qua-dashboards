import dash_bootstrap_components as dbc
import dash
from dash import html, dcc, Input, Output, State, ALL, no_update
from dash.exceptions import PreventUpdate
import numpy as np
from qua_dashboards.core import BaseComponent
from .virtual_layer_adder import VirtualLayerAdder
from .matrix_convention import to_apply_matrix, to_quam_matrix


class VirtualLayerEditor(BaseComponent):
    """
    A dash component allowing the editing of virtual layers to work with VirtualGateSet. Renders the matrix as an NxN editable table.
    An apply button changes the VirtualGateSet.layer matrix. 
    """


    def __init__(self,gateset,component_id = "VG_manager", dc_set = None, save_path=None):
        """
        Args:
            gateset: The VirtualGateSet instance
            component_id: Dash Component ID; also the name of the component title
        """
        super().__init__(component_id = component_id)
        self.gateset = gateset
        self.dc_set = dc_set
        self.save_path = save_path
        self.component_id = component_id
        # self.layer_names = [
        #     {"label": f"Layer {i+1}", "value": i}
        #     for i, _ in enumerate(self.gateset.layers)
        # ]
        self.layer_names = [
            {"label": (lyr.id or f"Layer {i+1}"), "value": i}
            for i, lyr in enumerate(self.gateset.layers)
        ]
        self._adder = VirtualLayerAdder(self.gateset, component_id = f"{component_id}-adder", layer_names=self.layer_names, dc_set=dc_set)

    @property
    def active_target(self): 
        """Returns gateset or dc_set based on current selection. Default is gateset"""
        return getattr(self, "_current_target_set", self.gateset)
    
    def _render_matrix_editor(self, layer_idx, use_dc=False):
        target_set = self.dc_set if (use_dc and self.dc_set) else self.gateset

        if layer_idx is None:
            return html.Div("No layer selected")

        if not 0 <= layer_idx < len(target_set.layers):
            return html.Div("Layer not available for this target")

        layer = target_set.layers[layer_idx]

        matrix_shown = to_apply_matrix(layer.matrix)

        header = html.Tr(
            [
                html.Th(""),
                *[
                    html.Th(
                        name,
                    )
                    for name in layer.target_gates
                ],
            ]
        )

        rows = []

        for i, source_name in enumerate(layer.source_gates):
            cells = [
                html.Th(
                    source_name,
                    style={
                        "padding": "4px 8px",
                        "textAlign": "right",
                        "whiteSpace": "nowrap",
                    },
                )
            ]

            for j in range(len(layer.target_gates)):
                cell_class = (
                    "virtual-matrix-cell-diagonal"
                    if i == j
                    else "virtual-matrix-cell"
                )
                cells.append(
                    html.Td(
                        dcc.Input(
                            id={
                                "type": "edit-matrix-cell",
                                "layer": layer_idx,
                                "row": i,
                                "col": j,
                            },
                            type="number",
                            value=matrix_shown[i][j],
                            debounce=False,
                            n_submit=0,
                            className="virtual-matrix-input",
                        ),
                        className=cell_class,
                    ),
                
                )

            rows.append(html.Tr(cells))

        return html.Table(
            [
                html.Thead(header),
                html.Tbody(rows),
            ],
            className="virtual-matrix-table",
        )


    def get_layout(self):
        layers = [layer for layer in self.gateset.layers]
        num_of_layers = len(layers)
        options = self.layer_names
        default = "0" if num_of_layers > 0 else None
        
        # Tsung-Lin
        opx_dc_toggle_switch = html.Div([
            dbc.Switch(id=f"{self.component_id}-target-switch", value=False),
            dbc.Checkbox(id=f"{self.component_id}-edit-both", value=True),
        ], style={"display": "none"}) if self.dc_set is not None else html.Div()
        # opx_dc_toggle_switch = dbc.Row([
        #     dbc.Col(html.Label("OPX", className="me-2"), width="auto"),
        #     dbc.Col(
        #         dbc.Switch(
        #             id=f"{self.component_id}-target-switch",
        #             value=False,  # False = OPX, True = DC
        #             className="mt-1",
        #         ),
        #         width="auto",
        #     ),
        #     dbc.Col(html.Label("DC"), width="auto"),
        #     dbc.Col(
        #         dbc.Checkbox(
        #             id=f"{self.component_id}-edit-both",
        #             label="Edit both",
        #             value=False,
        #             className="ms-4",
        #         ),
        #         width="auto",
        #     ),
        # ], className="mb-3 align-items-center") if self.dc_set is not None else html.Div()

        editor_card = dbc.Card([
            dbc.CardHeader("VirtualGateSet Layer Editor"),
            dbc.CardBody([
                dcc.Tabs(
                    id=f"{self.component_id}-layer-tabs",
                    value=default,
                    children=[
                        dcc.Tab(
                            label=option["label"],
                            value=str(option["value"]),
                            className="virtual-layer-tab",
                            selected_className="virtual-layer-tab-selected",
                        )
                        for option in options
                    ],
                    className="virtual-layer-tabs",
                ),
                html.Div(id = f"{self.component_id}-layer-matrix-editor"),
                dcc.Interval(
                    id=f"{self.component_id}-init",
                    interval=50, n_intervals=0, max_intervals=1
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    dbc.Button(
                                        "Apply Changes",
                                        id=f"{self.component_id}-apply-edit",
                                        color="primary",
                                        className="me-2 mt-2",
                                    ),
                                    dbc.Button(
                                        "Save QuAM state",
                                        id=f"{self.component_id}-save-quam",
                                        color="success",
                                        className="mt-2",
                                    ),
                                ]
                            ),
                            width="auto",
                        ),
                        dbc.Col(),
                        dbc.Col(
                            dbc.Button(
                                "Delete layer",
                                id=f"{self.component_id}-delete-layer",
                                color="danger",
                                className="mt-2",
                            ),
                            width="auto",
                        ),
                    ],
                    align="center",
                    className="g-2",
                ),
                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Delete layer?")),
                        dbc.ModalBody(id=f"{self.component_id}-delete-confirm-message"),
                        dbc.ModalFooter(
                            [
                                dbc.Button(
                                    "Cancel",
                                    id=f"{self.component_id}-delete-cancel",
                                    color="secondary",
                                    outline=True,
                                    className="me-2",
                                ),
                                dbc.Button(
                                    "Delete layer",
                                    id=f"{self.component_id}-delete-confirm",
                                    color="danger",
                                ),
                            ]
                        ),
                    ],
                    id=f"{self.component_id}-delete-modal",
                    is_open=False,
                    centered=True,
                ),
                html.Div(id=f"{self.component_id}-edit-output", style={"display": "none"}),
                html.Div(id=f"{self.component_id}-save-status", className="mt-2")
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
            Output(f"{self.component_id}-layer-tabs", "children"),
            Output(f"{self.component_id}-layer-tabs", "value"),
            Input(f"{self.component_id}-layer-tabs", "value"),
            Input(f"{self.component_id}-target-switch", "value")
            if self.dc_set
            else Input(f"{self.component_id}-init", "n_intervals"),
            Input("vg-layer-refresh-trigger", "data"),
            Input(f"{self.component_id}-init", "n_intervals"),
        )
        def show_selected_layer(layer_idx, use_dc, _refresh, _init):
            if self.dc_set is None:
                use_dc = False

            target_set = self.dc_set if (use_dc and self.dc_set) else self.gateset

            tabs = [
                dcc.Tab(
                    label=lyr.id or f"Layer {i + 1}",
                    value=str(i),
                    className="virtual-layer-tab",
                    selected_className="virtual-layer-tab-selected",
                )
                for i, lyr in enumerate(target_set.layers)
            ]

            if not tabs:
                return "No Layers", [], None

            if layer_idx is None:
                layer_idx = 0
            else:
                try:
                    layer_idx = int(layer_idx)
                except (TypeError, ValueError):
                    layer_idx = 0

            if not 0 <= layer_idx < len(target_set.layers):
                layer_idx = 0

            return self._render_matrix_editor(layer_idx, use_dc), tabs, str(layer_idx)

        @app.callback(
            Output(f"{self.component_id}-edit-output", "children"),
            Output(f"{self.component_id}-save-status", "children", allow_duplicate=True),
            Output("vg-layer-refresh-trigger", "data", allow_duplicate=True),
            Input(f"{self.component_id}-apply-edit", "n_clicks"),
            Input({"type": "edit-matrix-cell", "layer": ALL, "row": ALL, "col": ALL}, "n_submit"),
            State(f"{self.component_id}-layer-tabs", "value"),
            State(f"{self.component_id}-target-switch", "value") if self.dc_set else State(f"{self.component_id}-layer-tabs", "value"),
            State(f"{self.component_id}-edit-both", "value") if self.dc_set else State(f"{self.component_id}-layer-tabs", "value"),
            State({'type': 'edit-matrix-cell', 'layer': ALL, 'row': ALL, 'col': ALL}, 'value'),
            State({'type': 'edit-matrix-cell', 'layer': ALL, 'row': ALL, 'col': ALL}, 'id'),
            State("vg-layer-refresh-trigger", "data"),
            prevent_initial_call=True
        )
        def apply_layer_changes(n_clicks, n_submit, layer_idx, use_dc, edit_both, matrix_flat, cell_ids, vg_val):
            if not n_clicks and not any(n_submit or []):
                raise PreventUpdate

            if layer_idx is None:
                raise PreventUpdate

            try:
                layer_idx = int(layer_idx)
            except (TypeError, ValueError):
                raise PreventUpdate

            if self.dc_set is None:
                edit_both = True
                use_dc = False

            target_set = self.dc_set if (use_dc and self.dc_set) else self.gateset

            if not 0 <= layer_idx < len(target_set.layers):
                raise PreventUpdate

            layer = target_set.layers[layer_idx]

            num_sources = len(layer.source_gates)
            num_targets = len(layer.target_gates)

            coord_vals = {
                (cid["row"], cid["col"]): v
                for cid, v in zip(cell_ids, matrix_flat)
            }

            M = [
                [
                    float(coord_vals.get((i, j), 0.0) or 0.0)
                    for j in range(num_targets)
                ]
                for i in range(num_sources)
            ]

            M = to_quam_matrix(M)

            if edit_both:
                if layer_idx < len(self.gateset.layers):
                    self.gateset.layers[layer_idx].matrix = [row[:] for row in M]

                if self.dc_set is not None and layer_idx < len(self.dc_set.layers):
                    self.dc_set.layers[layer_idx].matrix = [row[:] for row in M]

            else:
                target_set.layers[layer_idx].matrix = M

            return "Updated!", "", (vg_val or 0) + 1
        
        @app.callback(
            Output(f"{self.component_id}-delete-modal", "is_open"),
            Output(f"{self.component_id}-delete-confirm-message", "children"),
            Input(f"{self.component_id}-delete-layer", "n_clicks"),
            Input(f"{self.component_id}-delete-cancel", "n_clicks"),
            State(f"{self.component_id}-delete-modal", "is_open"),
            State(f"{self.component_id}-layer-tabs", "value"),
            prevent_initial_call=True,
        )
        def toggle_delete_modal(delete_clicks, cancel_clicks, is_open, layer_idx):
            triggered = dash.ctx.triggered_id

            if triggered == f"{self.component_id}-delete-cancel":
                return False, no_update

            if triggered != f"{self.component_id}-delete-layer" or not delete_clicks:
                raise PreventUpdate

            if layer_idx is None:
                raise PreventUpdate

            try:
                layer_idx = int(layer_idx)
            except (TypeError, ValueError):
                raise PreventUpdate

            if not 0 <= layer_idx < len(self.gateset.layers):
                raise PreventUpdate

            layer = self.gateset.layers[layer_idx]
            layer_name = layer.id or f"Layer {layer_idx + 1}"

            message = html.Div(
                [
                    html.P(f'Are you sure you want to delete layer "{layer_name}"?'),
                    html.P(
                        "This removes the layer from the current QuAM state. "
                        "Save the QuAM state afterwards if you want to keep the deletion.",
                        className="mb-0",
                    ),
                ]
            )

            return True, message

        @app.callback(
            Output(f"{self.component_id}-save-status", "children", allow_duplicate=True),
            Output("vg-layer-refresh-trigger", "data", allow_duplicate=True),
            Output(f"{self.component_id}-delete-modal", "is_open", allow_duplicate=True),
            Input(f"{self.component_id}-delete-confirm", "n_clicks"),
            State(f"{self.component_id}-layer-tabs", "value"),
            State("vg-layer-refresh-trigger", "data"),
            prevent_initial_call=True,
        )
        def delete_layer(n_clicks, layer_idx, vg_val):
            if not n_clicks or layer_idx is None:
                raise PreventUpdate

            try:
                layer_idx = int(layer_idx)
            except (TypeError, ValueError):
                raise PreventUpdate

            if layer_idx != len(self.gateset.layers) - 1:
                return (
                    dbc.Alert("Only the top layer can be deleted.", color="warning"),
                    no_update,
                    False,
                )

            self.gateset.layers.pop(layer_idx)

            if self.dc_set is not None and layer_idx < len(self.dc_set.layers):
                self.dc_set.layers.pop(layer_idx)

            self.gateset._calculate_influence_map()

            return (
                dbc.Alert(
                    "Layer deleted (Save QuAM state to keep it).",
                    color="success",
                ),
                (vg_val or 0) + 1,
                False,
            )

        @app.callback(
            Output(f"{self.component_id}-save-status", "children"),
            Input(f"{self.component_id}-save-quam", "n_clicks"),
            prevent_initial_call=True,
        )
        def save_quam_state(n_clicks):
            if not n_clicks:
                raise PreventUpdate
            if not self.save_path:
                return dbc.Alert("save_path is not set.", color="danger", dismissable=True)
            try:
                machine = self.gateset.get_root()
                machine.save(self.save_path)
                return dbc.Alert(f"Saved to {self.save_path}", color="success", dismissable=True)
            except Exception as e:
                return dbc.Alert(f"Save failed: {e}", color="danger", dismissable=True)

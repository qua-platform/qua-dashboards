import dash_bootstrap_components as dbc
import dash
from dash import html, dcc, Input, Output, State, ALL, no_update
from dash.exceptions import PreventUpdate
import numpy as np
from .matrix_convention import to_quam_matrix

class VirtualLayerAdder:
    """
    A component allowing the addition of new virtualisation layers using VirtualGateSet. Renders an NxN editable matrix (default identity) with editable column names. 
    An apply button calls gateset.add_layer(..) internally. 
    """

    def __init__(self, gateset, component_id: str, layer_names: list, dc_set = None):
        """
        gateset: your VirtualGateSet instance
        component_id: a unique string to namespace this editor
        """
        self.gateset = gateset
        self.dc_set = dc_set
        self.component_id = component_id
        self.apply_button_id = f"{self.component_id}-apply-button"
        self.layout_columns = 9
        self.layout_rows    = 4 
        self.layer_names = layer_names

    def _matrix_builder(self, source_gates: list[str], target_gates: list[str]):
        header = html.Tr(
            [
                html.Th(""),
                *[
                    html.Th(
                        name,
                        style={
                            "padding": "4px 8px",
                            "textAlign": "center",
                            "whiteSpace": "nowrap",
                        },
                    )
                    for name in target_gates
                ],
            ]
        )

        rows = []

        for i, source_name in enumerate(source_gates):
            cells = [
                html.Th(
                    dcc.Input(
                        id={
                            "type": "vg-source-column",
                            "index": self.component_id,
                            "col": i,
                        },
                        value=source_name,
                        placeholder="new gate", # source gate name
                        className="virtual-matrix-source-input",
                    ),
                    style={"padding": "0"},
                )
            ]

            for j in range(len(target_gates)):
                default_value = 1.0 if i == j else 0.0

                class_name = (
                    "virtual-matrix-input virtual-matrix-input-diagonal"
                    if i == j
                    else "virtual-matrix-input"
                )

                cells.append(
                    html.Td(
                        dcc.Input(
                            id={
                                "type": "vg-layer-cell",
                                "index": self.component_id,
                                "row": i,
                                "col": j,
                            },
                            type="number",
                            value=default_value,
                            className=class_name,
                        ),
                        style={"padding": "0", "margin": "0"},
                    )
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
        # if self.gateset.layers:
        #     assigned = {target for layer in self.gateset.layers for target in layer.target_gates}
        #     full_list = [name for name in self.gateset.valid_channel_names if name not in assigned]
        # else:
        #     full_list = list(self.gateset.channels.keys())
        full_list = list(self.gateset.valid_channel_names)
        
        dropdown_options = [{"label":n,"value":n} for n in sorted(set(full_list))]

        return dbc.Card([
                    dbc.CardHeader("VirtualGateSet Layer Adder"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col(html.Label("Target gates"), width=2),
                            dbc.Col(dcc.Dropdown(
                                    id=f"{self.component_id}-tgt-dd",
                                    options=dropdown_options, multi=True, value=[], placeholder = "Pick Target Gates!"),
                                width=4),
                            dbc.Col(html.Label("Layer name"), width=2),
                            dbc.Col(dcc.Input(
                                id=f"{self.component_id}-layer-name",
                                type="text",
                                placeholder="Enter layer name",
                                style={"width": "100%"},
                            ), width=4),
                        ], className="mb-3"),

                        html.Div(id=f"{self.component_id}-matrix-container"),
                        dcc.Interval(
                            id=f"{self.component_id}-init",
                            interval=50, n_intervals=0, max_intervals=1
                        ),
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
            Output(f"{self.component_id}-layer-name", "value", allow_duplicate=True),
            Input(self.apply_button_id, "n_clicks"),
            State(f"{self.component_id}-tgt-dd", "value"),
            State({"type":"vg-source-column", "index":self.component_id, "col": ALL}, "value"),
            State({"type":"vg-layer-cell",   "index":self.component_id, "row": ALL, "col": ALL}, "value"),
            State({"type":"LAYER_REFRESH",    "index":self.component_id}, "data"),
            State("vg-layer-refresh-trigger","data"),
            State(f"{self.component_id}-layer-name", "value"),
            State({"type":"vg-layer-cell", "index":self.component_id, "row":ALL, "col":ALL}, "id"),
            prevent_initial_call=True
        )
        def _apply(nc, target_list, source_names, flat_cells, refresh, glob_ref, layer_name, cell_ids):
            if nc is None or not target_list:
                raise PreventUpdate
            new_idx = len(self.gateset.layers)
            layer_name = layer_name if layer_name else f"Layer {new_idx + 1}"

            M = len(target_list)
            K = len(source_names)
            if len(flat_cells) != M * K:
                app.logger.error(f"Wrong shape: Expected {M*K} cells, got {len(flat_cells)}")
                return no_update, no_update, no_update, no_update

            coord_vals = { (cid["row"], cid["col"]): v for cid, v in zip(cell_ids, flat_cells) }
            apply_mat = [[ float(coord_vals.get((i, j), 0.0)) for j in range(K)] for i in range(M)]
            mat = to_quam_matrix(apply_mat)
            gateset_added = False

            try:
                self.gateset.add_layer(
                    source_gates=source_names,
                    target_gates=target_list,
                    matrix=mat,
                    layer_id=layer_name,
                )
                gateset_added = True

                if self.dc_set is not None:
                    self.dc_set.add_layer(
                        source_gates=source_names,
                        target_gates=target_list,
                        matrix=mat,
                        layer_id=layer_name,
                    )

                app.logger.info(f"Added layer {source_names} -> {target_list}")

            except Exception as e:
                # Roll back gateset if dc_set failed after gateset succeeded.
                if gateset_added and len(self.gateset.layers) > new_idx:
                    self.gateset.layers.pop()
                    self.gateset._calculate_influence_map()

                app.logger.exception("add_layer failed")

                return f"Error: {e}", no_update, no_update, no_update
            
            
            self.layer_names.append({"label": layer_name, "value": new_idx})
            return "Layer Added!", (refresh or 0)+1, (glob_ref or 0)+1, ""
        
        @app.callback(
            Output(f"{self.component_id}-tgt-dd", "options"),
            Output(f"{self.component_id}-tgt-dd", "value"),
            Input("vg-layer-refresh-trigger", "data"),
            Input(f"{self.component_id}-init", "n_intervals"),
            prevent_initial_call=True,
        )
        def _refresh_target_dropdown(_refresh, _init):
            # if self.gateset.layers:
            #     assigned = {t for layer in self.gateset.layers for t in layer.target_gates}
            #     full_list = [n for n in self.gateset.valid_channel_names if n not in assigned]
            # else:
            #     full_list = list(self.gateset.channels.keys())
            full_list = list(self.gateset.valid_channel_names)
            options = [{"label": n, "value": n} for n in sorted(set(full_list))]
            return options, []

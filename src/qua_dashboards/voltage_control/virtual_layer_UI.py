import dash_bootstrap_components as dbc
import dash
from dash import html, dcc, Input, Output, State, ALL, no_update
from dash.exceptions import PreventUpdate

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
                dbc.CardHeader("Virtual Gate‐Set Layer Editor"),
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
            Input(self.apply_button_id, "n_clicks"),
            State({"type": "vg-layer-col", "index": self.component_id, "col": ALL}, "value"),
            State(
                {"type": "vg-layer-cell", "index": self.component_id, "row": ALL, "col": ALL},
                "value",
            ),
            State({"type": "LAYER_REFRESH", "index": self.component_id}, "data"),   # ← ADD THIS!

            prevent_initial_call=True,
        )
        def _apply_new_layer(n_clicks, col_names, cell_matrix, refresh_val):
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

            return no_update, (refresh_val or 0) + 1

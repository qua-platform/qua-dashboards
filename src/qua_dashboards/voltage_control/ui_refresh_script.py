from dash import html, Output, Input, dcc

def ui_update(app, gate_set, virtual_layer_ui, virtual_layer_editor):
    app.layout.children.append(dcc.Store(id="vg-layer-refresh-trigger", data=0))
    app.layout.children.append(
        html.Div(id="VG_MANAGER_CONTAINER", children=virtual_layer_editor.get_layout()))
    app.layout.children.append(
        html.Div(id="VG_EDITOR_CONTAINER", children=virtual_layer_ui.get_layout()))
    @app.callback(
        Output("VG_EDITOR_CONTAINER", "children"),
        Input("vg-layer-refresh-trigger", "data"))
    def refresh_editor_layout(_):
        return virtual_layer_ui.get_layout()
    @app.callback(
        Output("VG_MANAGER_CONTAINER", "children"),
        Input("vg-layer-refresh-trigger", "data"),)
    def refresh_manager_layout(_):
        return virtual_layer_editor.get_layout()
    virtual_layer_ui.register_callbacks(app)
    virtual_layer_editor.register_callbacks(app)
    @app.callback(
        Output("sweep-x-channel", "options"),
        Output("sweep-y-channel", "options"),
        Input("vg-layer-refresh-trigger", "data"),
    )
    def _refresh_gate_options(_trigger):
        names = list(gate_set.channels.keys())
        for layer in gate_set.layers:
            names.extend(layer.source_gates)
        opts = [{"label": n, "value": n} for n in names]
        return opts, opts
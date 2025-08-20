from dash import html, Output, Input, dcc

def ui_update(app, virtual_layer_editor):
    app.layout.children = list(app.layout.children) + [
    dcc.Store(id="vg-layer-refresh-trigger", data=0)
    ]

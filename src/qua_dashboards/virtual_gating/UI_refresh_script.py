from dash import dcc, Input, Output

def ui_update(app, video_mode_component):
    app.layout.children = list(app.layout.children) + [
    dcc.Store(id="vg-layer-refresh-trigger", data=0)
    ]
    live_tab = next(tc for tc in video_mode_component.tab_controllers if type(tc).__name__ == "LiveViewTabController")
    controls_div_id = live_tab._get_id(live_tab._ACQUIRER_CONTROLS_DIV_ID_SUFFIX)
    acq = video_mode_component.data_acquirer
    @app.callback(
        Output(controls_div_id, "children", allow_duplicate=True),
        Input("vg-layer-refresh-trigger", "data"),
        prevent_initial_call=True,
    )
    def _refresh_controls(_n):
        acq.mark_virtual_layer_changed(affects_config=False)
        acq.ensure_axis()
        return acq.get_dash_components(include_subcomponents=True)

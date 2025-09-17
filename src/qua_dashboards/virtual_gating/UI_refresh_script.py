from dash import html, Output, Input, dcc

def ui_update(app, video_mode_component=None):
    app.layout.children = list(app.layout.children) + [
    dcc.Store(id="vg-layer-refresh-trigger", data=0)
    ]
    live_tab = next(tc for tc in video_mode_component.tab_controllers if type(tc).__name__ == "LiveViewTabController")
    acquirer_controls_div_id = live_tab._get_id(live_tab._ACQUIRER_CONTROLS_DIV_ID_SUFFIX)
    if not video_mode_component.data_acquirer or not acquirer_controls_div_id:
        return
    
    @app.callback(
        Output(acquirer_controls_div_id, "children", allow_duplicate=True),
        Input("vg-layer-refresh-trigger", "data"),
        prevent_initial_call=True,
    )
    def _refresh_controls(_n):
        video_mode_component.data_acquirer.ensure_axis()
        return video_mode_component.data_acquirer.get_dash_components(include_subcomponents=True)

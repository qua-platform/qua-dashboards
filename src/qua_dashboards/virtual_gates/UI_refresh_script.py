from dash import dcc, Input, Output

def ui_update(app, video_mode_component, voltage_control_component = None):
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

    if voltage_control_component is not None and voltage_control_component.dc_set is not None: 
        dropdown_id = voltage_control_component._get_id("dc-gate-selector")
        @app.callback(
            Output(dropdown_id, "options"), 
            Input("vg-layer-refresh-trigger", "data"), 
            prevent_initial_call = True
        )
        def _refresh_voltage_dropdown(_n): 
            return voltage_control_component._build_dropdown_options()
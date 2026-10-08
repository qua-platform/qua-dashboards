from dash import dcc, Input, Output
from dash.exceptions import PreventUpdate


def ui_update(app, video_mode_component, refresh_interval_ms=1000):
    acq = video_mode_component.data_acquirer

    live_tab = next(
        tc for tc in video_mode_component.tab_controllers
        if type(tc).__name__ == "LiveViewTabController"
    )

    controls_div_id = live_tab._get_id(
        live_tab._ACQUIRER_CONTROLS_DIV_ID_SUFFIX
    )

    known_gates = set(acq.gate_set.valid_channel_names)

    app.layout.children = list(app.layout.children) + [
        dcc.Interval(
            id="vg-axis-refresh-interval",
            interval=refresh_interval_ms,
            n_intervals=0,
        )
    ]

    @app.callback(
        Output(controls_div_id, "children", allow_duplicate=True),
        Input("vg-axis-refresh-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def _refresh_controls(_n):
        nonlocal known_gates

        current_gates = set(acq.gate_set.valid_channel_names)

        if current_gates == known_gates:
            raise PreventUpdate

        acq.mark_virtual_layer_changed(affects_config=False)
        acq.ensure_axis()

        known_gates = current_gates

        return acq.get_dash_components(
            include_subcomponents=True
        )
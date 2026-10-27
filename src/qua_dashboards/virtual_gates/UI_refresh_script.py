from dash import dcc, Input, Output
from dash.exceptions import PreventUpdate


def ui_update(app, video_mode_component, refresh_interval_ms=1000):
    acq = video_mode_component.data_acquirer

    live_tab = next(
        tc for tc in video_mode_component.tab_controllers
        if type(tc).__name__ == "LiveViewTabController"
    )

    x_dropdown_id = acq._get_id("gate-select-x")
    y_dropdown_id = acq._get_id("gate-select-y")

    known_gates = set(acq.gate_set.valid_channel_names)

    app.layout.children = list(app.layout.children) + [
        dcc.Interval(
            id="vg-axis-refresh-interval",
            interval=refresh_interval_ms,
            n_intervals=0,
        )
    ]

    @app.callback(
        Output(x_dropdown_id, "options"),
        Output(y_dropdown_id, "options"),
        Output(x_dropdown_id, "value"),
        Output(y_dropdown_id, "value"),
        Input("vg-axis-refresh-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def _refresh_axes(_n):
        nonlocal known_gates

        current_gates = set(acq.gate_set.valid_channel_names)

        if current_gates == known_gates:
            raise PreventUpdate

        print("Virtual gates changed:")
        print("  before:", known_gates)
        print("  after :", current_gates)

        acq.mark_virtual_layer_changed(
            affects_config=False
        )
        acq.ensure_axis()

        known_gates = current_gates

        x_options = acq._build_dropdown_options(
            acq._display_x_sweep_axes
        )
        y_options = acq._build_dropdown_options(
            acq._display_y_sweep_axes
        )

        print(
            "Voltage axes:",
            [ax.name for ax in acq.sweep_axes["Voltage"]],
        )

        return (
            x_options,
            y_options,
            acq.x_axis_name,
            acq.y_axis_name,
        )
import logging
from typing import Any
from qua_dashboards.utils.dash_utils import create_input_field
import dash_bootstrap_components as dbc
from dash import html, ALL, no_update, Dash, Input, Output, State, dcc, callback_context, MATCH
from qua_dashboards.core.base_updatable_component import ModifiedFlags


from qua_dashboards.video_mode.tab_controllers.base_tab_controller import (
    BaseTabController,
)
from qua_dashboards.video_mode.data_acquirers.base_data_acquirer import BaseDataAcquirer

logger = logging.getLogger(__name__)

__all__ = ["SettingsTabController"]


class SettingsTabController(BaseTabController):
    """
    Controls the 'Settings' tab in the Video Mode application.
    This tab allows the user to adjust the settings of the measurement, such as the readout power and frequency, or the readout parameter (I/Q/R/Phase)
    """

    _TAB_LABEL = "Settings"
    _TAB_VALUE = "settings-tab"
    _DUMMY_OUT_SUFFIX = "dummy-settings-updates"

    def __init__(
        self,
        data_acquirer: BaseDataAcquirer,
        component_id: str = "settings-tab",
        is_active: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initializes the Settings Tab.
        Args:
            component_id: A unique string identifier for this component instance.
            data_acquirer: The Data Acquirer instance that the measurement uses
            **kwargs: Additional keyword arguments passed to BaseComponent.
        """
        super().__init__(component_id=component_id, is_active=is_active, **kwargs)
        self._data_acquirer_instance = data_acquirer
        logger.info(
            f"Settings Tab '{self.component_id}' initialized with "
            f"Data Acquirer '{self._data_acquirer_instance}'."
        )
        self._inner_loop_functions = self._data_acquirer_instance.inner_functions_dict

    def get_layout(self):
        # ramp_duration_input = create_input_field(
        #     id={
        #         "type": "ramp_duration",
        #         "index": f"{self._data_acquirer_instance.component_id}::ramp_duration",
        #     },
        #     label="Ramp Duration",
        #     value=getattr(self._data_acquirer_instance.qua_inner_loop_action, "ramp_duration", 16),
        #     units="ns",
        #     step=4,
        # )
        pre_measurement_delay_input = create_input_field(
            id={
                "type": "pre_measurement_delay",
                "index": f"{self._data_acquirer_instance.component_id}::pre_measurement_delay",
            },
            label="Pre-Measurement Delay",
            value=getattr(self._data_acquirer_instance.qua_inner_loop_action, "pre_measurement_delay", 0),
            units="ns",
            step=4,
        )

        inner_controls = []
        try:
            inner_controls = (
                self._data_acquirer_instance.qua_inner_loop_action.get_dash_components(
                    include_subcomponents=True
                )
            )
        except: 
            pass
        readout_selector = dbc.Row(
            [
                dbc.Label(
                    "Readouts to acquire", width="auto", className="col-form-label"
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id=self._data_acquirer_instance._get_id("readouts"),
                        options=[
                            {"label": ch, "value": ch}
                            for ch in self._data_acquirer_instance.available_readout_channels.keys()
                        ],
                        value=[
                            ch.name
                            for ch in self._data_acquirer_instance.selected_readout_channels
                        ],
                        multi=True,
                        clearable=False,
                        style={"color": "black"},
                    ),
                    width=True,
                ),
            ],
            className="mb-2 align-items-center",
        )

        inner_loop_functions = ["Go to Point(s)"] + list(self._inner_loop_functions.keys())
        inner_function_selector = dbc.Row([            
            dbc.Label("Inner Loop Function", width = "auto", className = "col-form-label"), 
            dbc.Col(
                dcc.Dropdown(
                    id = self._data_acquirer_instance._get_id("inner-loop-function"), 
                    options = [
                        {"label": fn_name, "value": fn_name} for fn_name in inner_loop_functions
                    ], 
                    value = None, 
                    multi= False,
                    clearable = True, 
                    style = {"color": "black"}, 
                ), 
                width = True,
            ),
            ], 
            className = "mb-2 align-items-center"
        )
        point_sequence_section = self._build_point_sequence_section()
        inner_loop_section = html.Div(
            [
                html.Small("Inner Loop Action", className="text-light mb-2", style={"display": "block"}),
                inner_function_selector,
                point_sequence_section,
            ],
            style={
                "outline": "2px solid #fff",
                "borderRadius": "8px",
                "padding": "12px",
                "marginBottom": "12px",
            },
        )

        post_processing_fn_selector = dbc.Row([            
            dbc.Label("Post-Processing Function", width = "auto", className = "col-form-label"), 
            dbc.Col(
                dcc.Dropdown(
                    id = self._data_acquirer_instance._get_id("post-processing-function"), 
                    options = [
                        {"label": fn_name, "value": fn_name} for (fn_name) in self._data_acquirer_instance.post_processing_functions.keys()
                    ], 
                    value = "Raw_data", 
                    multi= False,
                    clearable = False, 
                    style = {"color": "black"}, 
                ), 
                width = True,
            ),
            ], 
            className = "mb-2 align-items-center"
        )

        scan_mode_selector = dbc.Row([            
            dbc.Label("2D Scan Mode", width = "auto", className = "col-form-label"), 
            dbc.Col(
                dcc.Dropdown(
                    id = self._data_acquirer_instance._get_id("scan-mode-selection"), 
                    options = [
                        {"label": scan_mode, "value": scan_mode} for (scan_mode) in self._data_acquirer_instance.scan_modes.keys()
                    ], 
                    value = self._data_acquirer_instance.current_scan_mode, 
                    multi= False,
                    clearable = False, 
                    style = {"color": "black"}, 
                ), 
                width = True,
            ),
            ], 
            className = "mb-2 align-items-center"
        )

        result_type_selector = dbc.Row(
            [
                dbc.Label("Result Type", width="auto", className="col-form-label"),
                dbc.Col(
                    dbc.Select(
                        id={
                            "type": "select",
                            "index": f"{self._data_acquirer_instance.component_id}::result-type",
                        },
                        options=[
                            {"label": rt, "value": rt}
                            for rt in self._data_acquirer_instance.result_types
                        ],
                        value=self._data_acquirer_instance.result_type,
                        style={"width": "150px"},
                    ),
                    width=True,
                ),
            ],
            className="mb-2 align-items-center",
        )
        return dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Settings", className="text-light"),
                    readout_selector,
                    result_type_selector,
                    post_processing_fn_selector,
                    scan_mode_selector,
                    #ramp_duration_input,
                    # xy_duration_input,
                    pre_measurement_delay_input,
                    inner_loop_section,
                    *inner_controls,
                    html.Div(
                        id=self._get_id(self._DUMMY_OUT_SUFFIX),
                        style={"display": "none"},
                    ),
                ]
            ),
            color="dark",
            inverse=True,
            className="tab-card-dark",
        )

    def on_tab_activated(self):
        logger.info(f"SettingsTabController '{self.component_id}' activated.")
        self._data_acquirer_instance.stop_acquisition()
        from qua_dashboards.video_mode.video_mode_component import VideoModeComponent

        super().on_tab_activated()
        return {
            VideoModeComponent._MAIN_STATUS_ALERT_ID_SUFFIX: html.Span(
                "STOPPED", style={"display": "none"}
            )
        }

    def on_tab_deactivated(self):
        logger.info(f"SettingsTabController '{self.component_id}' deactivated.")
        return super().on_tab_deactivated()

    def register_callbacks(self, app: Dash, **kwargs):
        acq = self._data_acquirer_instance
        dummy_out = self._get_id(self._DUMMY_OUT_SUFFIX)
        main_status_id = kwargs["orchestrator_stores"]["main-status-alert"]

        @app.callback(
            Output(main_status_id, "children", allow_duplicate=True),
            Output(dummy_out, "children", allow_duplicate = True),
            Input({"type": "comp-inner-loop", "index": ALL}, "value"),
            State({"type": "comp-inner-loop", "index": ALL}, "id"),
            Input({"type": "select", "index": ALL}, "value"),
            State({"type": "select", "index": ALL}, "id"),
            Input({"type": "ramp_duration", "index": ALL}, "value"),
            State({"type": "ramp_duration", "index": ALL}, "id"),
            Input({"type": "pre_measurement_delay", "index": ALL}, "value"),
            State({"type": "pre_measurement_delay", "index": ALL}, "id"),
            prevent_initial_call=True,
        )
        def _apply_settings(
            inner_vals, inner_ids, select_vals, select_ids, ramp_vals, ramp_ids, pre_meas_delay_vals, pre_meas_delay_ids
        ):
            """
            Collect changes from the Settings tab and forward them to the acquirer.
            - Inner loop controls use ids like {'type': 'comp-inner-loop', 'index': 'readout_duration'}
            - Result type select uses id like {'type': 'select', 'index': 'opx-data-acquirer::result-type'}
            """
            params_to_update = {}
            if inner_vals and inner_ids:
                for v, idd in zip(inner_vals, inner_ids):
                    if not isinstance(idd, dict):
                        continue
                    param = idd.get("index")
                    if param is None:
                        continue
                    params_to_update.setdefault(
                        acq.qua_inner_loop_action.component_id, {}
                    )[param] = v
            if select_vals and select_ids:
                idx = select_ids[0].get("index")
                comp_id, param = idx.split("::", 1)
                params_to_update.setdefault(comp_id, {})[param] = select_vals[0]

            if ramp_vals and ramp_ids:
                idx = ramp_ids[0].get("index")
                comp_id, param = idx.split("::", 1)
                if ramp_vals[0] is not None: 
                    ramp_vals[0] = int(ramp_vals[0])
                params_to_update.setdefault(comp_id, {})[param] = ramp_vals[0]
            if pre_meas_delay_vals and pre_meas_delay_ids:
                idx = pre_meas_delay_ids[0].get("index")
                comp_id, param = idx.split("::", 1)
                if pre_meas_delay_vals[0] is not None: 
                    pre_meas_delay_vals[0] = int(pre_meas_delay_vals[0])
                params_to_update.setdefault(comp_id, {})[param] = pre_meas_delay_vals[0]

            if not params_to_update:
                return no_update, no_update
            try: 
                acq.update_parameters(params_to_update)
                return "", no_update
            except Exception as e:  
                import dash_bootstrap_components as dbc
                return dbc.Alert(str(e), color="danger", dismissable=True), no_update
        
        @app.callback(
            Output(dummy_out, "children", allow_duplicate = True), 
            Input(self._data_acquirer_instance._get_id("post-processing-function"), "value"), 
            prevent_initial_call = True,
        ) 
        def _alter_post_processing_function(fn_name): 
            self._data_acquirer_instance.selected_function = self._data_acquirer_instance.post_processing_functions[fn_name]
            logger.info(f"Post Processing Function Changed to {fn_name}")
            return no_update
        
        @app.callback(
            Output(dummy_out, "children", allow_duplicate = True), 
            Input(self._data_acquirer_instance._get_id("scan-mode-selection"), "value"), 
            prevent_initial_call = True,
        ) 
        def _alter_scan_mode(name): 
            self._data_acquirer_instance.set_scan_mode(name)
            logger.info(f"Scan Mode Changed to {name}")
            return no_update
        
        @app.callback(
            Output(f"{self.component_id}-point-sequence-section", "style"), 
            Input(acq._get_id("inner-loop-function"), "value"), 
            prevent_initial_call=True,
        )
        def _toggle_point_section(fn_name): 
            if fn_name == "Go to Point(s)": 
                return {"display": "block"}
            elif fn_name in self._inner_loop_functions: 
                user_function = self._inner_loop_functions[fn_name]
                def loop_action(inner_loop_self): 
                    user_function()
                acq.qua_inner_loop_action.loop_action = loop_action
                acq._compilation_flags |= ModifiedFlags.PROGRAM_MODIFIED
                return {"display": "none"}
            
            else: 
                def loop_action(inner_loop_self): 
                    pass
                def pre_loop_action(inner_loop_self): 
                    pass
                acq.qua_inner_loop_action.loop_action = loop_action
                acq.qua_inner_loop_action.pre_loop_action = pre_loop_action
                acq._compilation_flags |= ModifiedFlags.PROGRAM_MODIFIED
                return {"display": "none"}
        
        @app.callback(
            Output(f"{self.component_id}-point-rows-container", "children"),
            Output(main_status_id, "children", allow_duplicate=True),
            Input(f"{self.component_id}-add-point-btn", "n_clicks"),
            Input({"type": "remove-point-btn", "index": ALL}, "n_clicks"),
            Input({"type": "point-timing", "index": ALL}, "value"),
            State({"type": "point-select", "index": ALL}, "value"),
            State({"type": "point-duration", "index": ALL}, "value"),
            State(f"{self.component_id}-xy-duration", "value"),
            State({"type": "point-ramp", "index": ALL}, "value"),
            State(f"{self.component_id}-xy-ramp", "value"),
            prevent_initial_call=True,
        )
        def _manage_point_rows(add_clicks, remove_clicks, timings, point_names, durations, xy_duration, point_ramps, xy_ramp):
            ctx = callback_context
            if not ctx.triggered:
                return no_update, no_update
            
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
            gate_set = acq.gate_set
            
            if not hasattr(gate_set, 'macros') or not gate_set.get_macros():
                if trigger_id == f"{self.component_id}-add-point-btn":
                    return no_update, dbc.Alert("No points defined in the GateSet. Identify some in the Annotation Tab first.", color="warning", dismissable=True, duration=4000)
                return no_update, no_update
            
            available_points = [{"label": name, "value": name} for name in gate_set.get_macros().keys()]
            rows_data = list(zip(timings, point_names, durations, point_ramps))
            
            if trigger_id == f"{self.component_id}-add-point-btn":
                rows_data.append(("pre", available_points[0]["value"] if available_points else None, 1000, 16))
            else:
                try:
                    trigger_dict = eval(trigger_id)
                    if trigger_dict.get("type") == "remove-point-btn":
                        remove_idx = trigger_dict["index"]
                        rows_data = [r for i, r in enumerate(rows_data) if i != remove_idx]
                except:
                    return no_update, no_update
            
            pre_rows = [(t, p, int(d), int(r)) for t, p, d, r in rows_data if t == "pre"]
            post_rows = [(t, p, int(d), int(r)) for t, p, d, r in rows_data if t == "post"]

            pre_points = [(p, int(d), int(r)) for t, p, d, r in rows_data if t == "pre" and p and d]
            post_points = [(p, int(d), int(r)) for t, p, d, r in rows_data if t == "post" and p and d]
            
            acq.qua_inner_loop_action.point_duration = int(xy_duration) if xy_duration else 0
            acq.qua_inner_loop_action.ramp_duration = int(xy_ramp) if xy_ramp else 16
            
            def pre_loop_action(inner_loop_self):
                for point, duration, ramp_duration in pre_points:
                    inner_loop_self.voltage_sequence.ramp_to_point(
                        point, duration=duration, ramp_duration=ramp_duration
                    )
            
            def loop_action(inner_loop_self):
                for point, duration, ramp_duration in post_points:
                    inner_loop_self.voltage_sequence.ramp_to_point(
                        point, duration=duration, ramp_duration=ramp_duration
                    )
            
            acq.qua_inner_loop_action.pre_loop_action = pre_loop_action
            acq.qua_inner_loop_action.loop_action = loop_action
            acq._compilation_flags |= ModifiedFlags.PROGRAM_MODIFIED
            logger.info(f"Point sequence updated from row manager - Pre: {pre_points}, Post: {post_points}")
            
            result = [self._build_point_header_row()]
            for i, (t, p, d, r) in enumerate(pre_rows):
                result.append(self._build_point_row(i, available_points, t, p, d, r))
            
            result.append(self._build_xy_row(xy_duration or 0, xy_ramp or 16))
            
            for i, (t, p, d, r) in enumerate(post_rows):
                result.append(self._build_point_row(len(pre_rows) + i, available_points, t, p, d, r))
            
            return result, no_update
        
        @app.callback(
            Output({"type": "point-duration", "index": MATCH}, "value"),
            Input({"type": "point-select", "index": MATCH}, "value"),
            prevent_initial_call=True,
        )
        def _update_duration_from_point(point_name):
            if not point_name:
                return no_update
            
            macros = acq.gate_set.get_macros()
            if point_name in macros:
                return macros[point_name].duration
            return no_update
                
        @app.callback(
            Output(dummy_out, "children", allow_duplicate=True), 
            Input({"type": "point-select", "index": ALL}, "value"),
            Input({"type": "point-duration", "index": ALL}, "value"), 
            Input({"type": "point-timing", "index": ALL}, "value"),
            Input(f"{self.component_id}-xy-duration", "value"),
            Input({"type": "point-ramp", "index": ALL}, "value"),
            Input(f"{self.component_id}-xy-ramp", "value"),
            prevent_initial_call = True,
        )
        def _update_point_sequence(point_names, durations, timings, xy_duration, point_ramps, xy_ramp): 
            pre_points = []
            post_points = []

            for point, duration, timing, ramp in list(zip(point_names, durations, timings, point_ramps)): 
                if point and duration: 
                    ramp_val = int(ramp or 16)
                    if timing == "pre": 
                        pre_points.append((point, int(duration), ramp_val))
                    else: 
                        post_points.append((point, int(duration), ramp_val))
            acq.qua_inner_loop_action.point_duration = int(xy_duration or 0)
            acq.qua_inner_loop_action.ramp_duration = int(xy_ramp or 16)
            def pre_loop_action(inner_loop_self): 
                for point, duration, ramp in pre_points: 
                    inner_loop_self.voltage_sequence.ramp_to_point(point, duration = duration, ramp_duration = ramp)
            def loop_action(inner_loop_self): 
                for point, duration, ramp in post_points: 
                    inner_loop_self.voltage_sequence.ramp_to_point(point, duration = duration, ramp_duration = ramp)

            acq.qua_inner_loop_action.loop_action = loop_action
            acq.qua_inner_loop_action.pre_loop_action = pre_loop_action
            logger.info(f"Point sequence updated - Pre: {pre_points}, Post: {post_points}")
            acq._compilation_flags |= ModifiedFlags.PROGRAM_MODIFIED
            return no_update
        

        @app.callback(
            Output(f"{self.component_id}-timeline-display", "children"),
            Input({"type": "point-select", "index": ALL}, "value"),
            Input({"type": "point-duration", "index": ALL}, "value"),
            Input({"type": "point-timing", "index": ALL}, "value"),
            Input(f"{self.component_id}-xy-duration", "value"),
            Input({"type": "pre_measurement_delay", "index": ALL}, "value"),
            Input({"type": "comp-inner-loop", "index": ALL}, "value"),
            State({"type": "comp-inner-loop", "index": ALL}, "id"),
            Input({"type": "point-ramp", "index": ALL}, "value"),
            Input(f"{self.component_id}-xy-ramp", "value"),
            prevent_initial_call=False,
        )
        def _update_timeline(
            point_names, point_durations, timings, 
            xy_dur, pre_meas_vals,
            inner_vals, inner_ids, 
            point_ramps, xy_ramp,
        ):
            inner_loop = acq.qua_inner_loop_action
            pre_measurement_delay = pre_meas_vals[0] if pre_meas_vals else 0
            readout_duration = max(
                inner_loop._pulse_for(ch).length 
                for ch in inner_loop.selected_readout_channels
            ) if inner_loop.selected_readout_channels else 0
            time_of_flight = max(ch.time_of_flight for ch in inner_loop.selected_readout_channels)

            pre_points = []
            post_points = []
            
            for point, duration, timing, ramp in zip(point_names, point_durations, timings, point_ramps):
                if point and duration:
                    ramp_val = int(ramp or 16)
                    if timing == "pre":
                        pre_points.append((point, int(duration), ramp_val))
                    else:
                        post_points.append((point, int(duration), ramp_val))
            
            return self._build_timeline_visuals(
                        pre_points=pre_points,
                        xy_duration=xy_dur or 0,
                        xy_ramp = int(xy_ramp or 16),
                        post_points=post_points,
                        pre_measurement_delay=pre_measurement_delay,
                        readout_duration=readout_duration,
                        time_of_flight=time_of_flight
                    )
    def _build_point_header_row(self) -> dbc.Row:
        return dbc.Row([
            dbc.Col(html.Small("Timing", className="text-light"), width=2),
            dbc.Col(html.Small("Point", className="text-light"), width=5),
            dbc.Col(html.Small("Ramp Dur (ns)", className="text-light"), width=2),
            dbc.Col(html.Small("Point Dur (ns)", className="text-light"), width=3),
        ], className="mb-1 g-1 align-items-center flex-nowrap pe-1")
        
    def _build_xy_row(self, xy_duration = 0, xy_ramp = 16): 
        return dbc.Row(
            [
                dbc.Col(html.Span(""), width=2),
                dbc.Col(html.Span("(x,y)", style={"fontWeight": "bold"}), width=5),
                dbc.Col(
                    dbc.Input(
                        id=f"{self.component_id}-xy-ramp",
                        type="number",
                        value=xy_ramp,
                        min=4,
                        step=4,
                        size="sm",
                        style={"width": "70px"},
                    ),
                    width=2,
                ),
                dbc.Col(
                    dbc.Input(
                        id=f"{self.component_id}-xy-duration",
                        type="number",
                        value=xy_duration,
                        min=0,
                        step=4,
                        size="sm",
                        style={"width": "80px"},
                    ),
                    width="auto", 
                    className="px-0 me-1"
                ),
            ],
            className="mb-1 g-1 align-items-center flex-nowrap pe-1",
            style={"padding": "0px"},
        )

    def _build_point_sequence_section(self): 
        """
        Build a collapsible section for point sequence configuration
        """

        inner_loop = self._data_acquirer_instance.qua_inner_loop_action
        xy_ramp = getattr(inner_loop, "ramp_duration", 16)
        pre_measurement_delay = getattr(inner_loop, "pre_measurement_delay", 0)
        readout_duration = max(
            inner_loop._pulse_for(ch).length 
            for ch in inner_loop.selected_readout_channels
        ) if inner_loop.selected_readout_channels else 0
        time_of_flight = max(ch.time_of_flight for ch in inner_loop.selected_readout_channels)

        xy_duration=int(getattr(inner_loop, "point_duration", 0) or 0)
        initial_timeline = self._build_timeline_visuals(
            pre_points=[],
            xy_duration=xy_duration,
            post_points=[],
            xy_ramp = xy_ramp,
            pre_measurement_delay=pre_measurement_delay,
            readout_duration=readout_duration,
            time_of_flight = time_of_flight
        )

        return html.Div([
                html.Div([
                    html.Div(id=f"{self.component_id}-point-rows-container", children=[self._build_point_header_row(),self._build_xy_row(xy_duration, xy_ramp)]),
                            dbc.Button("+ Add Point", id=f"{self.component_id}-add-point-btn", size="sm", className="mt-2"),
                        ],
                    id=f"{self.component_id}-point-sequence-section",
                    style={"display": "none"},
                    ), 
                    html.Div(
                        id=f"{self.component_id}-timeline-display",
                        children=initial_timeline,
                        style={"marginTop": "12px"},
                    ),
                ])

    def _build_point_row(self, index: int, available_points: list, timing_value = "pre", point_value = None, duration_value = None, ramp_value=None) -> dbc.Row: 
        macros = self._data_acquirer_instance.gate_set.get_macros()
        if point_value is None: 
            point_value = available_points[0]["value"] if available_points else None

        if duration_value is None:
            duration_value = 1000 
            if point_value and duration_value in macros:
                macro = macros[point_value]
                duration_value = macro.duration
        if ramp_value is None: 
            ramp_value = 16

        row = dbc.Row(
            [
                dbc.Col(
                    dbc.Select(
                        id={"type": "point-timing", "index": index},
                        options=[
                            {"label": "Before (x,y)", "value": "pre"},
                            {"label": "After (x,y)", "value": "post"},
                        ],
                        value=timing_value,
                        size="sm",
                        style={"width": "100%"},
                    ),
                    width=2,
                ),
                dbc.Col(
                    dbc.Select(
                        id = {"type": "point-select", "index": index}, 
                        options = available_points, 
                        value = point_value,
                        # clearable = True, 
                        size = "sm",
                        style = {"color": "black", "width": "100%"}
                    ), 
                    width = 5,
                ), 
                dbc.Col(
                    dbc.Input(
                        id={"type": "point-ramp", "index": index},
                        type="number",
                        value=ramp_value,
                        min=16,
                        step=4,
                        size="sm",
                        style={"width": "70px"},
                    ),
                    width=2,
                ),
                dbc.Col(
                    dbc.Input(
                        id={"type": "point-duration", "index": index},
                        type="number",
                        value=duration_value,
                        min=16,
                        step=4,
                        size="sm",
                        style={"width": "80px"},
                    ),
                    width=3,
                ),
                dbc.Col(
                    dbc.Button(
                        "X",
                        id={"type": "remove-point-btn", "index": index},
                        color="danger",
                        size="sm",
                        className="px-1",
                        style={"marginLeft": "-6px"}
                    ),
                    width="auto",
                    className="ps-0",
                ),
            ], 
            className="mb-1 g-1 align-items-center flex-nowrap pe-1"
        )

        return row

    def _build_timeline_visuals(
        self, 
        pre_points, 
        xy_duration, 
        xy_ramp,
        post_points, 
        pre_measurement_delay, 
        readout_duration, 
        time_of_flight,
    ) -> None: 
        """
        A function which builds the visual representation of the pixel timeline, based on the point sequence. Prints the total point duration
        """

        segments = []
        total_duration = 0

        for name, duration, ramp in pre_points: 
            segments.append({"label": name, "duration": duration, "type": "pre", "ramp": ramp})
            total_duration = total_duration + duration + ramp
        
        segments.append({"label": "(x,y)", "duration": xy_duration, "type": "xy", "ramp": xy_ramp})
        total_duration = total_duration + xy_duration + xy_ramp

        for name, duration, ramp in post_points: 
            segments.append({"label": name, "duration": duration, "type": "post", "ramp": ramp})
            total_duration = total_duration + duration + ramp

        segments.append({"label": "PMD", "duration": pre_measurement_delay, "type": "pmw", "ramp": 0})
        segments.append({"label": "Readout", "duration": readout_duration, "type": "readout", "ramp": 0})
        total_duration += pre_measurement_delay + readout_duration

        colours = {"pre": "#6c9bd1", "xy": "#f0ad4e", "post": "#5cb85c", "pmw": "#6f5e96", "readout": "#d9534f", "ramp": "#eaff00"}
        blocks = []
        block_height = "40px"
        for seg in segments: 
            if seg["ramp"] > 0: 
                blocks.append(html.Div(
                    f"Ramp", 
                    style = {
                        "width": f"{40}px",
                        "backgroundColor": colours["ramp"],
                        "textAlign": "center",
                        "lineHeight": block_height,
                        "height": block_height,
                        "fontSize": "10px",
                        "color": "black",
                    }
                ))
            blocks.append(html.Div(
                f"{seg['label']} {seg['duration']}ns", 
                style={
                    "width": f"{100}px",
                    "backgroundColor": colours[seg["type"]],
                    "textAlign": "center",
                    "fontSize": "10px",
                    "height": block_height,
                    "lineHeight": block_height,
                    "color": "black",
                    "overflow": "hidden",
                }
            ))
        readout_overhead = (time_of_flight + 52)
        blocks.append(
            html.Div(
                f"{readout_overhead}ns OH", 
                style={
                    "width": f"{50}px",
                    "backgroundColor": "#566161",
                    "textAlign": "center",
                    "fontSize": "10px",
                    "height": block_height,
                    "lineHeight": block_height,
                    "color": "black",
                    "overflow": "hidden",
                }
            )
        )
        total_duration = total_duration + readout_overhead
        return html.Div([
            html.Div(blocks, style={"display": "flex", "alignItems": "center", "marginBottom": "8px", "overflowX": "auto",}),
            html.Div(f"Total pixel duration: {total_duration} ns", style={"fontSize": "12px", "color": "#aaa"}),
        ])


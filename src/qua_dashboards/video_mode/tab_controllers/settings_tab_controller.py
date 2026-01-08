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
        ramp_duration_input = create_input_field(
            id={
                "type": "ramp_duration",
                "index": f"{self._data_acquirer_instance.component_id}::ramp_duration",
            },
            label="Ramp Duration",
            value=getattr(self._data_acquirer_instance.qua_inner_loop_action, "ramp_duration", 16),
            units="ns",
            step=4,
        )
        point_duration_input = create_input_field(
            id={
                "type": "point_duration",
                "index": f"{self._data_acquirer_instance.component_id}::point_duration",
            },
            label="Point Duration",
            value=getattr(self._data_acquirer_instance.qua_inner_loop_action, "point_duration", 1000),
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
                    ramp_duration_input,
                    point_duration_input,
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
            Input({"type": "point_duration", "index": ALL}, "value"),
            State({"type": "point_duration", "index": ALL}, "id"),
            prevent_initial_call=True,
        )
        def _apply_settings(
            inner_vals, inner_ids, select_vals, select_ids, ramp_vals, ramp_ids, point_vals, point_ids,
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
                params_to_update.setdefault(comp_id, {})[param] = ramp_vals[0]
            if point_vals and point_ids:
                idx = point_ids[0].get("index")
                comp_id, param = idx.split("::", 1)
                params_to_update.setdefault(comp_id, {})[param] = point_vals[0]

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
            State(f"{self.component_id}-point-rows-container", "children"),
            prevent_initial_call=True,
        )
        def _manage_point_rows(add_clicks, remove_clicks, current_rows):
            ctx = callback_context
            if not ctx.triggered:
                return no_update, no_update
            
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
            gate_set = acq.gate_set
            
            if not hasattr(gate_set, 'macros') or not gate_set.get_macros():
                return no_update, dbc.Alert("No points defined in the GateSet. Identify some in the Annotation Tab first.", color="warning", dismissable=True, duration=4000)
            
            available_points = [{"label": name, "value": name} for name in gate_set.get_macros().keys()]
            
            if trigger_id == f"{self.component_id}-add-point-btn":
                current_rows = current_rows or []
                new_index = len(current_rows)
                current_rows.append(self._build_point_row(new_index, available_points))
                return current_rows, no_update
            
            try:
                trigger_dict = eval(trigger_id)
                if trigger_dict.get("type") == "remove-point-btn":
                    remove_idx = trigger_dict["index"]
                    new_rows = []
                    for i, row in enumerate(current_rows):
                        if i != remove_idx:
                            new_rows.append(self._build_point_row(len(new_rows), available_points))
                    return new_rows, no_update
            except:
                pass
            
            return no_update, no_update
        
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
            prevent_initial_call = True,
        )
        def _update_point_sequence(point_names, durations, timings): 
            pre_points = []
            post_points = []

            for point, duration, timing in list(zip(point_names, durations, timings)): 
                if point and duration: 
                    if timing == "pre": 
                        pre_points.append((point, int(duration)))
                    else: 
                        post_points.append((point, int(duration)))

            def pre_loop_action(inner_loop_self): 
                for point, duration in pre_points: 
                    inner_loop_self.voltage_sequence.ramp_to_point(point, duration = duration, ramp_duration = inner_loop_self.ramp_duration)
            def loop_action(inner_loop_self): 
                for point, duration in post_points: 
                    inner_loop_self.voltage_sequence.ramp_to_point(point, duration = duration, ramp_duration = inner_loop_self.ramp_duration)

            acq.qua_inner_loop_action.loop_action = loop_action
            acq.qua_inner_loop_action.pre_loop_action = pre_loop_action
            logger.info(f"Point sequence updated - Pre: {pre_points}, Post: {post_points}")
            acq._compilation_flags |= ModifiedFlags.PROGRAM_MODIFIED
            return no_update

    def _build_point_sequence_section(self): 
        """
        Build a collapsible section for point sequence configuration
        """

        return html.Div(
            [
                html.Div(id=f"{self.component_id}-point-rows-container", children=[]),
                dbc.Button("+ Add Point", id=f"{self.component_id}-add-point-btn", size="sm", className="mt-2"),
            ],
            id=f"{self.component_id}-point-sequence-section",
            style={"display": "none"},
        )

    def _build_point_row(self, index: int, available_points: list) -> None: 
        macros = self._data_acquirer_instance.gate_set.get_macros()
        default_point = available_points[0]["value"] if available_points else None
        default_duration = 1000 
        if default_point and default_point in macros:
            macro = macros[default_point]
            default_duration = macro.duration

        row = dbc.Row(
            [
                dbc.Col(
                    dbc.Select(
                        id={"type": "point-timing", "index": index},
                        options=[
                            {"label": "Before (x,y)", "value": "pre"},
                            {"label": "After (x,y)", "value": "post"},
                        ],
                        value="pre",
                        size="sm",
                        style={"width": "150px"},
                    ),
                    width=3,
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id = {"type": "point-select", "index": index}, 
                        options = available_points, 
                        value = default_point,
                        clearable = True, 
                        style = {"color": "black"}
                    ), 
                    width = 4,
                ), 
                dbc.Col(
                    dcc.Input(
                        id = {"type": "point-duration", "index": index}, 
                        type = "number",
                        value = default_duration,
                        min = 16, 
                        step = 4, 
                        placeholder = "ns",
                        style={"width": "80px"},
                    ), 
                    width = 2,
                ), 
                dbc.Col(
                dbc.Button("X", id={"type": "remove-point-btn", "index": index}, color="danger", size="sm"),
                width=1,
            ),
            ], 
            className = "mb-2"
        )

        return row



    
        

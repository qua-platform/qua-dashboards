import logging
from typing import Any
from qua_dashboards.utils.dash_utils import create_input_field
import dash_bootstrap_components as dbc
from dash import html, ALL, no_update, Dash, Input, Output, State, dcc

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

    def get_layout(self):
        ramp_duration_input = create_input_field(
            id={
                "type": "ramp_duration",
                "index": f"{self._data_acquirer_instance.component_id}::ramp_duration",
            },
            label="Ramp Duration",
            value=self._data_acquirer_instance.qua_inner_loop_action.ramp_duration,
            units="ns",
            step=4,
        )
        inner_controls = (
            self._data_acquirer_instance.qua_inner_loop_action.get_dash_components(
                include_subcomponents=True
            )
        )
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

        @app.callback(
            Output(dummy_out, "children", allow_duplicate = True),
            Input({"type": "comp-inner-loop", "index": ALL}, "value"),
            State({"type": "comp-inner-loop", "index": ALL}, "id"),
            Input({"type": "select", "index": ALL}, "value"),
            State({"type": "select", "index": ALL}, "id"),
            Input({"type": "ramp_duration", "index": ALL}, "value"),
            State({"type": "ramp_duration", "index": ALL}, "id"),
            prevent_initial_call=True,
        )
        def _apply_settings(
            inner_vals, inner_ids, select_vals, select_ids, ramp_vals, ramp_ids
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

            if params_to_update:
                acq.update_parameters(params_to_update)

            return no_update
        
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
        

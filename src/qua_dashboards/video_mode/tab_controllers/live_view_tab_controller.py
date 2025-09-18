import logging
import uuid
from typing import Any, Dict, Union

import dash_bootstrap_components as dbc
import dash
from dash import Dash, Input, Output, State, html, ctx, ALL, dcc

from qua_dashboards.video_mode.data_acquirers.base_data_acquirer import (
    BaseDataAcquirer,
    ModifiedFlags,
)
import numpy as np
from qua_dashboards.video_mode.tab_controllers.base_tab_controller import (
    BaseTabController,
)
from qua_dashboards.video_mode import data_registry


logger = logging.getLogger(__name__)

__all__ = ["LiveViewTabController"]


class LiveViewTabController(BaseTabController):
    """
    Controls the 'Live View' tab in the Video Mode application.

    This tab allows users to start and stop the data acquisition process
    using a single toggle button and view the current status. It also allows
    configuration of the data acquirer parameters and sets the
    shared viewer to display live data from the acquirer.
    """

    _TAB_LABEL = "Live View"
    _TAB_VALUE = "live-view-tab"

    _TOGGLE_ACQ_BUTTON_ID_SUFFIX = "toggle-acq-button"
    _ACQUIRER_CONTROLS_DIV_ID_SUFFIX = "acquirer-controls-div"
    _ACQUIRER_STATUS_INDICATOR_ID_SUFFIX = "acquirer-status-indicator"
    _DUMMY_OUTPUT_ACQUIRER_UPDATE_SUFFIX = "dummy-output-acquirer-updates"

    def __init__(
        self,
        data_acquirer: BaseDataAcquirer,
        component_id: str = "live-view-tab-controller",
        is_active: bool = False,
        show_inner_loop_controls: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initializes the LiveViewTabController.

        Args:
            component_id: A unique string identifier for this component instance.
            data_acquirer: The data acquirer instance that this tab will control
                and interact with.
            **kwargs: Additional keyword arguments passed to BaseComponent.
        """
        super().__init__(component_id=component_id, is_active=is_active, **kwargs)
        self._data_acquirer_instance: BaseDataAcquirer = data_acquirer
        self._show_inner_loop_controls = show_inner_loop_controls
        logger.info(
            f"LiveViewTabController '{self.component_id}' initialized with "
            f"Data Acquirer '{self._data_acquirer_instance.component_id}'."
        )

    def get_layout(self) -> dbc.Card:
        """
        Generates the Dash layout for the Live View control panel.

        The layout includes a single toggle button for starting/stopping data
        acquisition, a status indicator, and embeds the data acquirer's
        specific parameter controls.

        Returns:
            An html.Div component containing the controls.
        """
        logger.debug(
            f"Generating layout for LiveViewTabController '{self.component_id}'"
        )

        toggle_button_and_status = dbc.Row(
            [
                dbc.Col(
                    dbc.Button(
                        "Start Acquisition",  # Initial text
                        id=self._get_id(self._TOGGLE_ACQ_BUTTON_ID_SUFFIX),
                        color="success",  # Initial color for "Start"
                        className="me-1",
                        style={"width": "100%"},
                    ),
                    width=8,  # Adjusted width
                ),
                dbc.Col(
                    [
                        html.Div(
                            dbc.Badge(
                                "STOPPED",  # Initial status text
                                id=self._get_id(self._ACQUIRER_STATUS_INDICATOR_ID_SUFFIX),
                                color="secondary",  # Initial color for STOPPED
                                className="ms-1 p-2",  # Added padding
                                style={
                                    "fontSize": "0.9rem",
                                    "width": "100%",
                                    "textAlign": "center",
                                },
                            ),
                            className=(
                                "d-flex align-items-center justify-content-center h-100"
                            ),
                        ),
                        html.Div(
                            [
                                dbc.Row(
                                    dbc.Col(
                                        dbc.Checklist(
                                            id=self._get_id("center-marker-toggle"),
                                            options=[{"label": "Gridlines", "value": "on"}],
                                            value=[],
                                            switch=True,
                                        ),
                                        width="auto",
                                    ),
                                    className="align-items-center g-1",
                                ),

                                dbc.Row(
                                    dbc.Col(
                                        html.Div(
                                            dcc.Slider(
                                                id=self._get_id("grid_opacity"),
                                                min=0, max=100, step=5, value=40,
                                                marks={0: "0%", 100: "100%"},
                                                tooltip={"always_visible": False, "placement": "bottom"},
                                                updatemode="drag",
                                            ),
                                            style={"width": "140px"},
                                        ),
                                        width="auto",
                                    ),
                                    className="mt-2", 
                                ),
                            ],
                            className="mt-4",
                        )
                    ],
                    width=4,
                ),
            ],
            className="mb-3 align-items-center",
        )

        acquirer_specific_controls = (
            self._data_acquirer_instance.get_dash_components(include_subcomponents=True, include_inner_loop_controls=self._show_inner_loop_controls)
            if self._data_acquirer_instance
            else [html.P("Data acquirer components could not be loaded.")]
        )

        acquirer_controls_div = html.Div(
            id=self._get_id(self._ACQUIRER_CONTROLS_DIV_ID_SUFFIX),  # type: ignore
            children=acquirer_specific_controls,
            className="mt-3 p-3 border rounded",
        )

        card_body = dbc.CardBody(
            [
                html.H5("Live Acquisition Control", className="card-title text-light"),
                toggle_button_and_status,
                html.Hr(),
                html.H6("Acquirer Parameters", className="text-light"),
                acquirer_controls_div,
                html.Div(
                    id=self._get_id(self._DUMMY_OUTPUT_ACQUIRER_UPDATE_SUFFIX),  # type: ignore
                    style={"display": "none"},
                ),
            ]
        )
        return dbc.Card(
            card_body, color="dark", inverse=True, className="tab-card-dark"
        )

    def on_tab_activated(self) -> Dict[str, Any]:
        """
        Called by the orchestrator when this tab becomes active.

        Sets the shared viewer to point to the live data stream from the
        data_registry.
        """
        from qua_dashboards.video_mode.video_mode_component import VideoModeComponent

        logger.info(f"LiveViewTabController '{self.component_id}' activated.")

        current_live_version = data_registry.get_current_version(
            data_registry.LIVE_DATA_KEY
        )

        if current_live_version is None:
            current_live_version = str(uuid.uuid4())
            logger.debug(
                f"{self.component_id}: No current live data version found. "
                f"Using placeholder: {current_live_version}"
            )

        viewer_data_payload = {
            "key": data_registry.LIVE_DATA_KEY,
            "version": current_live_version,
        }

        updates = {
            VideoModeComponent.VIEWER_DATA_STORE_SUFFIX: viewer_data_payload,
            VideoModeComponent.VIEWER_UI_STATE_STORE_SUFFIX: {},
            VideoModeComponent.VIEWER_LAYOUT_CONFIG_STORE_SUFFIX: {},
        }
        return updates

    def on_tab_deactivated(self) -> None:
        """Called by the orchestrator when this tab is no longer active."""
        logger.info(f"LiveViewTabController '{self.component_id}' deactivated.")
        pass

    def register_callbacks(
        self,
        app: Dash,
        orchestrator_stores: Dict[str, Any],
        shared_viewer_store_ids: Dict[str, Any],
    ) -> None:
        """
        Registers Dash callbacks for the Live View tab.

        Args:
            app: The main Dash application instance.
            orchestrator_stores: Dictionary of orchestrator-level store IDs.
            shared_viewer_store_ids: Dictionary of shared viewer store IDs.
        """
        logger.info(
            f"Registering callbacks for LiveViewTabController '{self.component_id}'."
        )
        self._register_acquisition_control_callback(app, orchestrator_stores)
        self._register_parameter_update_callback(app)
        self._register_gate_selection_callback(app)
        self._register_gridlines_callback(app, shared_viewer_store_ids)

    def _register_gridlines_callback(
            self, 
            app: Dash, 
            shared_viewer_store_ids: Dict[str, Any]
        ) -> None:
        @app.callback(
            Output(shared_viewer_store_ids["layout_config_store"], "data"),
            Input(self._get_id("center-marker-toggle"), "value"),
            Input(self._get_id("grid_opacity"), "value"),
            State(shared_viewer_store_ids["layout_config_store"], "data"),
            prevent_initial_call=True,
        )
        def _toggle_gridlines(toggle_val, opacity, existing_layout):
            layout = existing_layout or {}
            show = "on" in toggle_val
            shapes = []
            alpha = float(opacity)/100

            if show: 
                #Get the range from the data acquirer instance
                xr = list(self._data_acquirer_instance.x_axis.sweep_values_unattenuated)
                yr = list(self._data_acquirer_instance.y_axis.sweep_values_unattenuated)

                #Set grid lines points, and ensure that 0 is included
                xs, ys = np.linspace(xr[0], xr[-1], 15).tolist() + [0], np.linspace(yr[0], yr[-1], 15).tolist() + [0]

                for xv in xs:
                    #0 grid line has 5x higher alpha
                    grid_color = f"rgba(0,0,0,{alpha*5})" if xv == 0 else f"rgba(0,0,0,{alpha})"
                    shapes.append({
                        "type": "line",
                        "xref": "x", "yref": "paper",
                        "x0": xv, "x1": xv, "y0": 0, "y1": 1,
                        "line": {"width": 4, "color": grid_color},
                        "layer": "above",
                        "name": "grid-x",
                    })
                for yv in ys:
                    grid_color = f"rgba(0,0,0,{alpha*5})" if yv == 0 else f"rgba(0,0,0,{alpha})"
                    shapes.append({
                        "type": "line",
                        "xref": "paper", "yref": "y",
                        "x0": 0, "x1": 1, "y0": yv, "y1": yv,
                        "line": {"width": 4, "color": grid_color},
                        "layer": "above",
                        "name": "grid-y",
                    })
            layout["shapes"] = shapes
            return layout

    def _register_gate_selection_callback(
            self, app:Dash
    ) -> None:
        """Registers callback for gate selection"""

        @app.callback(
            Output(self._get_id(self._DUMMY_OUTPUT_ACQUIRER_UPDATE_SUFFIX), "children", allow_duplicate=True), 
            Output(self._get_id(self._ACQUIRER_CONTROLS_DIV_ID_SUFFIX), "children", allow_duplicate=True),
            Input(self._data_acquirer_instance._get_id("gate-select-x"), "value"), 
            Input(self._data_acquirer_instance._get_id("gate-select-y"), "value"), 
            prevent_initial_call = True
        )
        def on_gate_select(x_gate, y_gate):
            logger.info(f"New Gate Selected: x_axis {x_gate}, y_axis {y_gate}")           

            params = {
                self._data_acquirer_instance.component_id: {
                    "gate-select-x": x_gate, 
                    "gate-select-y": y_gate
                }
            }
            self._data_acquirer_instance.update_parameters(params)
            return "", self._data_acquirer_instance.get_dash_components(include_subcomponents=True, include_inner_loop_controls=self._show_inner_loop_controls)

    def _register_acquisition_control_callback(
        self, app: Dash, orchestrator_stores: Dict[str, Any]
    ) -> None:
        """Registers callback for acquisition control and status updates."""
        from qua_dashboards.video_mode.video_mode_component import VideoModeComponent

        main_status_alert_id = orchestrator_stores.get(
            VideoModeComponent._MAIN_STATUS_ALERT_ID_SUFFIX
        )
        if not main_status_alert_id:
            logger.error(
                f"Could not find {VideoModeComponent._MAIN_STATUS_ALERT_ID_SUFFIX} "
                "in orchestrator_stores. Status synchronization might be affected."
            )

        @app.callback(
            Output(self._get_id(self._TOGGLE_ACQ_BUTTON_ID_SUFFIX), "children"),
            Output(self._get_id(self._TOGGLE_ACQ_BUTTON_ID_SUFFIX), "color"),
            Output(self._get_id(self._ACQUIRER_STATUS_INDICATOR_ID_SUFFIX), "children"),
            Output(self._get_id(self._ACQUIRER_STATUS_INDICATOR_ID_SUFFIX), "color"),
            Input(self._get_id(self._TOGGLE_ACQ_BUTTON_ID_SUFFIX), "n_clicks"),
            Input(main_status_alert_id, "children"),
            prevent_initial_call=True,
        )
        def handle_acquisition_control_and_status_update(
            _toggle_clicks: Any, _status_alert_trigger: Any
        ) -> tuple[str, str, str, str]:
            """
            Handles acquisition toggle and updates UI based on acquirer status.
            """
            triggered_input_id_obj = ctx.triggered_id
            is_button_click = False
            if isinstance(triggered_input_id_obj, dict):  # Pattern matched ID
                is_button_click = (
                    triggered_input_id_obj.get("index")
                    == self._TOGGLE_ACQ_BUTTON_ID_SUFFIX
                )
            elif isinstance(triggered_input_id_obj, str):  # Simple string ID
                is_button_click = (
                    self._get_id(self._TOGGLE_ACQ_BUTTON_ID_SUFFIX)
                    == triggered_input_id_obj
                )

            acquirer_state = self._data_acquirer_instance.get_latest_data()
            current_status = acquirer_state.get("status", "unknown").upper()
            error_details = acquirer_state.get("error")

            button_text: str
            button_color: str
            status_text: str
            status_color: str

            if is_button_click:
                logger.debug(
                    f"LiveViewTab: Toggle acquisition button clicked. "
                    f"Current reported acquirer status: {current_status}"
                )
                if current_status == "RUNNING":
                    logger.info(
                        f"Attempting to stop acquisition for "
                        f"'{self._data_acquirer_instance.component_id}'"
                    )
                    self._data_acquirer_instance.stop_acquisition()
                    button_text, button_color = "Start Acquisition", "success"
                    status_text, status_color = "STOPPED", "secondary"
                else:  # Was STOPPED, ERROR, or UNKNOWN
                    logger.info(
                        f"Attempting to start acquisition for "
                        f"'{self._data_acquirer_instance.component_id}'"
                    )
                    self._data_acquirer_instance.start_acquisition()
                    button_text, button_color = "Stop Acquisition", "danger"
                    status_text, status_color = "RUNNING", "success"
            else:
                if current_status != "STOPPED":
                    logger.debug(
                        f"LiveViewTab: Status update triggered externally. "
                        f"Acquirer status: {current_status}"
                    )
                if current_status == "RUNNING":
                    button_text, button_color = "Stop Acquisition", "danger"
                    status_text, status_color = "RUNNING", "success"
                elif current_status == "STOPPED":
                    button_text, button_color = "Start Acquisition", "success"
                    status_text, status_color = "STOPPED", "secondary"
                elif current_status == "ERROR":
                    button_text, button_color = (
                        "Start Acquisition",
                        "success",
                    )
                    status_text = (
                        f"ERROR{(': ' + str(error_details)) if error_details else ''}"
                    )
                    status_text = status_text[:100]
                    status_color = "danger"
                else:  # Unknown or other states
                    button_text, button_color = "Start Acquisition", "warning"
                    status_text, status_color = current_status, "warning"

            return button_text, button_color, status_text, status_color

    def _register_parameter_update_callback(self, app: Dash) -> None:
        """Hybrid callback: pattern-matching for sweep axes, static inputs for inner loop/scan mode."""
        
        all_acquirer_components = self._data_acquirer_instance.get_components()
        static_components = [
            comp for comp in all_acquirer_components 
            if not hasattr(comp, 'span')  # Exclude SweepAxis components
            and getattr(comp, 'component_id', None) != 'inner-loop'
        ]


        static_inputs = []
        static_states = []
        if static_components:
            static_inputs = [
                Input(component._get_id(ALL), "value")
                for component in static_components
            ]
            static_states = [
                State(component._get_id(ALL), "id") 
                for component in static_components
            ]

        @app.callback(
            Output(self._get_id(self._DUMMY_OUTPUT_ACQUIRER_UPDATE_SUFFIX), "children", allow_duplicate=True),
            Input({"type": "number-input", "index": ALL}, "value"),
            *static_inputs,
            State({"type": "number-input", "index": ALL}, "id"),
            *static_states,
            prevent_initial_call=True,
        )
        def handle_hybrid_parameter_update(*args):
            """Handle both pattern-matched axis parameters and static component parameters."""

            num_pattern_inputs = 1
            num_static_inputs = len(static_inputs)
            num_pattern_states = 1
            num_static_states = len(static_states)

            pattern_values = args[:num_pattern_inputs]
            static_values = args[num_pattern_inputs:num_pattern_inputs + num_static_inputs]
            pattern_states = args[num_pattern_inputs + num_static_inputs:
                                num_pattern_inputs + num_static_inputs + num_pattern_states]
            static_states_data = args[num_pattern_inputs + num_static_inputs + num_pattern_states:]

            parameters_to_update = {}

            if len(pattern_values) >= 1 and len(pattern_states) >= 1:
                number_values = pattern_values[0]
                number_ids = pattern_states[0]
                if number_values and number_ids:
                    for value, id_dict in zip(number_values, number_ids):
                        if value is not None and isinstance(id_dict, dict):
                            param_key = id_dict.get("index", "")
                            if param_key:
                                component_id, param_name = self._parse_param_id(param_key)
                                if component_id and param_name:
                                    parameters_to_update.setdefault(component_id, {})[param_name] = value

            for i, component in enumerate(static_components):
                if i < len(static_values) and i < len(static_states_data):
                    component_params = self._parse_component_parameters(
                        component.component_id,
                        static_values[i],
                        static_states_data[i],
                    )
                    if component_params:
                        parameters_to_update[component.component_id] = component_params

            if parameters_to_update:
                self._data_acquirer_instance.update_parameters(parameters_to_update)
            return dash.no_update

    def _parse_param_id(self, param_id_str):
        """Parse parameter ID string to extract component_id and parameter name."""
        try:
            if "::" in param_id_str:
                component_id, param_name = param_id_str.split("::", 1)
                return component_id, param_name
            return None, None
        except:
            return None, None
    @staticmethod
    def _parse_component_parameters(
        component_id: Union[str, dict],
        values: Any,
        ids: Any,
    ) -> Dict[str, Any]:
        if not values or not ids:
            return {}

        current_type_params: Dict[str, Any] = {}
        for idx, param_id_dict in enumerate(ids):
            if isinstance(param_id_dict, dict) and "index" in param_id_dict:
                param_name = param_id_dict["index"]
                if idx < len(values):
                    param_value = values[idx]
                    current_type_params[param_name] = param_value
                else:
                    logger.warning(
                        f"Value missing for param_id {param_id_dict} "
                        f"of type {component_id}"
                    )
            else:
                logger.warning(
                    f"Unexpected ID format in acquirer params: "
                    f"{param_id_dict} of type {component_id}"
                )
        return current_type_params

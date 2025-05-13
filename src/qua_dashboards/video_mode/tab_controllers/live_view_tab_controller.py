import logging
import uuid
from typing import Any, Dict

import dash_bootstrap_components as dbc
import dash
from dash import Dash, Input, Output, State, html, ctx, ALL

from qua_dashboards.video_mode.data_acquirers.base_data_acquirer import (
    BaseDataAcquirer,
    ModifiedFlags,
)
from qua_dashboards.video_mode.tab_controllers.itab_controller import ITabController
from qua_dashboards.video_mode import data_registry


logger = logging.getLogger(__name__)

__all__ = ["LiveViewTabController"]


class LiveViewTabController(ITabController):
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
        logger.info(
            f"LiveViewTabController '{self.component_id}' initialized with "
            f"Data Acquirer '{self._data_acquirer_instance.component_id}'."
        )

    def get_layout(self) -> html.Div:
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
                    width=4,  # Adjusted width
                ),
            ],
            className="mb-3 align-items-center",
        )

        acquirer_specific_controls = (
            self._data_acquirer_instance.get_dash_components(include_subcomponents=True)
            if self._data_acquirer_instance
            else [html.P("Data acquirer components could not be loaded.")]
        )

        acquirer_controls_div = html.Div(
            id=self._get_id(self._ACQUIRER_CONTROLS_DIV_ID_SUFFIX),  # type: ignore
            children=acquirer_specific_controls,
            className="mt-3 p-3 border rounded",
        )

        return html.Div(
            [
                html.H5("Live Acquisition Control", className="card-title"),
                toggle_button_and_status,
                html.Hr(),
                html.H6("Acquirer Parameters"),
                acquirer_controls_div,
                html.Div(
                    id=self._get_id(self._DUMMY_OUTPUT_ACQUIRER_UPDATE_SUFFIX),  # type: ignore
                    style={"display": "none"},
                ),
            ]
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
        from qua_dashboards.video_mode.video_mode_component import VideoModeComponent

        logger.info(
            f"Registering callbacks for LiveViewTabController '{self.component_id}'."
        )

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
            # Add Input for the main status alert if available to react to polled status
            Input(main_status_alert_id, "children")
            if main_status_alert_id
            else Input(
                self._get_id(self._TOGGLE_ACQ_BUTTON_ID_SUFFIX), "id"
            ),  # Dummy input if alert id not found
            prevent_initial_call=True,
        )
        def handle_acquisition_control_and_status_update(
            _toggle_clicks: Any, _status_alert_trigger: Any
        ) -> tuple[str, str, str, str]:
            """
            Handles acquisition toggle and updates UI based on acquirer status.

            This callback is triggered by:
            1. Clicks on the toggle acquisition button.
            2. Changes in the main status alert (by polling in VideoModeComponent),
               allowing the UI to reflect the actual acquirer status even if the
               action was not initiated from this tab (e.g., after a refresh or
               if acquisition was stopped by other means).
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
                    # Optimistic UI update, will be confirmed by next status poll
                    button_text, button_color = "Start Acquisition", "success"
                    status_text, status_color = "STOPPED", "secondary"
                else:  # Was STOPPED, ERROR, or UNKNOWN
                    logger.info(
                        f"Attempting to start acquisition for "
                        f"'{self._data_acquirer_instance.component_id}'"
                    )
                    self._data_acquirer_instance.start_acquisition()
                    # Optimistic UI update
                    button_text, button_color = "Stop Acquisition", "danger"
                    status_text, status_color = "RUNNING", "success"
            else:
                # Status update triggered by polling or tab activation, not a button
                # click. Simply reflect the current state of the acquirer

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
                    )  # Allow attempt to restart
                    status_text = (
                        f"ERROR{(': ' + str(error_details)) if error_details else ''}"
                    )
                    status_text = status_text[:100]  # Truncate long errors
                    status_color = "danger"
                else:  # Unknown or other states
                    button_text, button_color = "Start Acquisition", "warning"
                    status_text, status_color = current_status, "warning"

            return button_text, button_color, status_text, status_color

        # Callbacks for Data Acquirer Parameters
        all_acquirer_managed_ids_types = (
            self._data_acquirer_instance.get_component_ids()
        )

        if not all_acquirer_managed_ids_types:
            logger.warning(
                f"DataAcquirer '{self._data_acquirer_instance.component_id}' "
                f"reported no managed component IDs for parameter updates."
            )
        else:
            dynamic_inputs = [
                Input({"type": comp_type_id, "index": ALL}, "value")
                for comp_type_id in all_acquirer_managed_ids_types
            ]
            dynamic_states_ids = [
                State({"type": comp_type_id, "index": ALL}, "id")
                for comp_type_id in all_acquirer_managed_ids_types
            ]

            @app.callback(
                Output(
                    self._get_id(self._DUMMY_OUTPUT_ACQUIRER_UPDATE_SUFFIX),
                    component_property="children",
                ),
                dynamic_inputs,
                dynamic_states_ids,
                prevent_initial_call=True,
            )
            def handle_acquirer_parameter_update(*args: Any):
                num_comp_types = len(all_acquirer_managed_ids_types)
                values_by_type_list = args[:num_comp_types]
                ids_by_type_list = args[num_comp_types : 2 * num_comp_types]

                parameters_to_update: Dict[str, Dict[str, Any]] = {}

                for i, comp_type_id_for_params in enumerate(
                    all_acquirer_managed_ids_types
                ):
                    param_values_for_this_type = values_by_type_list[i]
                    param_ids_dicts_for_this_type = ids_by_type_list[i]

                    if (
                        not param_values_for_this_type
                        or not param_ids_dicts_for_this_type
                    ):
                        continue

                    current_type_params: Dict[str, Any] = {}
                    for idx, param_id_dict in enumerate(param_ids_dicts_for_this_type):
                        if isinstance(param_id_dict, dict) and "index" in param_id_dict:
                            param_name = param_id_dict["index"]
                            if idx < len(param_values_for_this_type):
                                param_value = param_values_for_this_type[idx]
                                current_type_params[param_name] = param_value
                            else:
                                logger.warning(
                                    f"Value missing for param_id {param_id_dict} "
                                    f"of type {comp_type_id_for_params}"
                                )
                        else:
                            logger.warning(
                                f"Unexpected ID format in acquirer params: "
                                f"{param_id_dict} of type {comp_type_id_for_params}"
                            )

                    if current_type_params:
                        parameters_to_update[comp_type_id_for_params] = (
                            current_type_params
                        )

                if not parameters_to_update:
                    logger.debug(
                        "No valid parameters found to update for acquirer "
                        f"'{self._data_acquirer_instance.component_id}'."
                    )
                    return dash.no_update

                logger.debug(
                    f"Updating DataAcquirer "
                    f"'{self._data_acquirer_instance.component_id}' "
                    f"with parameters: {parameters_to_update}"
                )
                modified_flags = self._data_acquirer_instance.update_parameters(
                    parameters_to_update
                )
                logger.info(f"DataAcquirer parameters updated. Flags: {modified_flags}")

                if modified_flags != ModifiedFlags.NONE:
                    if modified_flags & ModifiedFlags.PROGRAM_MODIFIED:
                        logger.info(
                            f"Acquirer '{self._data_acquirer_instance.component_id}' "
                            f"program was modified and recompiled."
                        )
                    if modified_flags & ModifiedFlags.CONFIG_MODIFIED:
                        logger.info(
                            f"Acquirer '{self._data_acquirer_instance.component_id}' "
                            f"config was modified and reloaded."
                        )
                return dash.no_update

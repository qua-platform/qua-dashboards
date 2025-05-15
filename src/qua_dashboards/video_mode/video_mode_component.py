import logging
from typing import Any, Dict, List, Optional, Tuple

import dash_bootstrap_components as dbc

from dash import Dash, Input, Output, State, dcc, html, ctx, no_update
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate

from qua_dashboards.core import BaseComponent
from qua_dashboards.video_mode.data_acquirers.base_data_acquirer import (
    BaseDataAcquirer,
)
from qua_dashboards.video_mode.tab_controllers.base_tab_controller import (
    BaseTabController,
)
from qua_dashboards.video_mode import data_registry
from qua_dashboards.video_mode.utils.data_utils import save_data
from qua_dashboards.video_mode.shared_viewer_component import SharedViewerComponent
from qua_dashboards.video_mode.tab_controllers.live_view_tab_controller import (
    LiveViewTabController,
)
from qua_dashboards.video_mode.tab_controllers.annotation_tab_controller import (
    AnnotationTabController,
)

logger = logging.getLogger(__name__)

__all__ = ["VideoModeComponent"]


class VideoModeComponent(BaseComponent):
    """
    Orchestrates the Video Mode UI.

    This component manages data acquisition, a shared viewer for displaying
    visualizations, and a set of tab controllers for different interaction modes.
    It utilizes a server-side data registry for efficient handling of large
    datasets, minimizing data transfer to and from the frontend.
    """

    _TABS_ID_SUFFIX = "control-tabs"
    _DATA_POLLING_INTERVAL_ID_SUFFIX = "data-polling-interval"
    _MAIN_STATUS_ALERT_ID_SUFFIX = "main-status-alert"

    # SUFFIX for the dcc.Store that holds a reference to the latest data
    # processed by the data acquirer.
    # Content: A dictionary like
    # `{"key": "key_in_registry", "version": "uuid_str"}`
    # or None/no_update. This store is always updated by the polling callback
    # when new data is available from the acquirer.
    LATEST_PROCESSED_DATA_STORE_SUFFIX = "latest-processed-data-store"

    # SUFFIX for the dcc.Store that holds the reference to the data to be displayed
    # in the shared viewer.
    # Content: A dictionary `{"key": "live_data" | "static_data_key", "version": "uuid_str"}`.
    # The key points to an entry in the data_registry.
    # "live_data" (data_registry.LIVE_DATA_KEY) for live streaming (base image + annotations).
    # "static_data_key" (e.g., STATIC_DATA_KEY from AnnotationTabController) for static
    # objects (base image + annotations).
    VIEWER_DATA_STORE_SUFFIX = "viewer-data-store"

    # SUFFIX for the dcc.Store that holds configuration updates for the shared viewer's layout.
    # Content: A dictionary of Plotly layout updates (e.g., `{'clickmode': 'event+select'}`).
    # These updates are applied to the figure's layout.
    # Can also be an empty dictionary or dash.no_update.
    VIEWER_LAYOUT_CONFIG_STORE_SUFFIX = "viewer-layout-config-store"

    DEFAULT_COMPONENT_ID = "video-mode-component"

    def __init__(
        self,
        data_acquirer: BaseDataAcquirer,
        tab_controllers: Optional[List[BaseTabController]] = None,
        default_tab_value: Optional[str] = None,
        component_id: str = DEFAULT_COMPONENT_ID,
        data_polling_interval_s: float = 0.1,
        layout_columns: int = 12,
        **kwargs: Any,
    ):
        """Initializes the VideoModeComponent.

        Args:
            data_acquirer: The data acquirer instance.
            tab_controllers: Optional list of ITabController instances.
                If None, default controllers (LiveView, Annotation) are used.
            default_tab_value: Value of the tab to be active by default.
            component_id: Unique ID for this component instance.
            data_polling_interval_s: Interval for polling data.
            layout_columns: Number of layout columns (for grid systems).
            **kwargs: Additional keyword arguments for BaseComponent.
        """
        super().__init__(component_id=component_id, **kwargs)
        self.layout_columns = layout_columns

        if not isinstance(data_acquirer, BaseDataAcquirer):
            raise TypeError("data_acquirer must be an instance of BaseDataAcquirer.")

        self.data_acquirer: BaseDataAcquirer = data_acquirer
        self.shared_viewer: SharedViewerComponent = SharedViewerComponent(
            component_id=f"{self.component_id}-shared-viewer"
        )

        if tab_controllers is None:
            logger.info(
                f"No tab_controllers provided for {self.component_id}. "
                "Initializing with default LiveViewTabController and "
                "AnnotationTabController."
            )
            live_view_tab = LiveViewTabController(
                component_id=f"{self.component_id}-live-view-tab",
                data_acquirer=self.data_acquirer,  # Pass data_acquirer here
            )
            annotation_tab = AnnotationTabController(
                component_id=f"{self.component_id}-annotation-tab",
            )
            self.tab_controllers: List[BaseTabController] = [
                live_view_tab,
                annotation_tab,
            ]
        else:
            self.tab_controllers = tab_controllers

        if not self.tab_controllers:
            logger.warning(
                f"VideoModeComponent ({self.component_id}) initialized with no "
                "tab controllers. The UI will lack tab-based interactions."
            )
            self._active_tab_value: str = "no-tabs-dummy"
        elif not all(isinstance(tc, BaseTabController) for tc in self.tab_controllers):
            raise TypeError(
                "All items in tab_controllers must implement ITabController."
            )
        else:
            for tc in self.tab_controllers:
                if tc.get_tab_value() == default_tab_value:
                    self._active_tab_value = tc.get_tab_value()
                    tc.is_active = True
                    break
            else:
                self._active_tab_value = self.tab_controllers[0].get_tab_value()
                self.tab_controllers[0].is_active = True

        self.data_polling_interval_ms: int = int(data_polling_interval_s * 1000)
        self._last_known_acquirer_status: str = "stopped"
        self._current_live_data_version: Optional[str] = None

        logger.info(
            f"VideoModeComponent '{self.component_id}' initialized with "
            f"Data Acquirer: {type(data_acquirer).__name__} and "
            f"{len(self.tab_controllers)} tab controllers. "
            f"Active tab: {self._active_tab_value}"
        )

    def _get_store_id(self, suffix: str) -> Dict[str, str]:
        """Generates a namespaced Dash component ID for internal stores."""
        return self._get_id(suffix)

    def get_layout(self) -> Component:
        """Generates the Dash layout for the VideoModeComponent."""
        logger.debug(f"Generating layout for VideoModeComponent '{self.component_id}'")

        tabs_children = []
        if not self.tab_controllers:
            tabs_children.append(
                dcc.Tab(
                    label="No Tabs Configured",
                    value="no-tabs-dummy",
                    children=[
                        html.Div(
                            "Please provide or configure tab controllers.",
                            style={"padding": "20px", "textAlign": "center"},
                        )
                    ],
                )
            )
        else:
            for tc in self.tab_controllers:
                tabs_children.append(
                    dcc.Tab(
                        label=tc.get_tab_label(),
                        value=tc.get_tab_value(),
                        id=self._get_id(f"tab-ctrl-wrapper-{tc.get_tab_value()}"),
                        children=[
                            html.Div(
                                tc.get_layout(),  # type: ignore
                                style={"padding": "10px"},
                                className="tab-controller-content-wrapper",
                            )
                        ],
                    )
                )

        left_panel_controls = dcc.Tabs(
            id=self._get_id(self._TABS_ID_SUFFIX),
            value=self._active_tab_value,
            children=tabs_children,
            className="mb-3",
            # persistence=True, # Optional: persist active tab
            # persistence_type="session",
        )

        # --- Save Button Row (for future extensibility, use dbc.Row with one button for now) ---
        bottom_left_row = dbc.Row(
            [
                dbc.Col(
                    dbc.Button(
                        "Save",
                        id=self._get_id("save-button"),
                        color="success",
                        className="me-2",
                    ),
                    width=3,
                ),
            ],
            className="mb-3 g-2 justify-content-start",
        )

        right_panel_viewer = self.shared_viewer.get_layout()

        shared_stores_for_layout = [
            dcc.Store(
                id=self._get_store_id(self.LATEST_PROCESSED_DATA_STORE_SUFFIX),
                data=None,
            ),
            dcc.Store(
                id=self._get_store_id(self.VIEWER_DATA_STORE_SUFFIX),
                data={"key": "live_data", "version": None},
            ),
            dcc.Store(
                id=self._get_store_id(self.VIEWER_LAYOUT_CONFIG_STORE_SUFFIX), data={}
            ),
        ]

        main_layout = dbc.Container(
            fluid=True,
            children=[
                html.Div(
                    id=self._get_id(self._MAIN_STATUS_ALERT_ID_SUFFIX), className="mb-2"
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                left_panel_controls,
                                bottom_left_row,  # Insert buttons (save, ...)
                            ],
                            width=12,
                            lg=4,
                            className="mb-3 mb-lg-0",
                            style={
                                "maxHeight": "calc(100vh - 70px)",
                                "overflowY": "auto",
                            },
                        ),
                        dbc.Col(
                            right_panel_viewer,
                            width=12,
                            lg=8,
                            style={
                                "maxHeight": "calc(100vh - 70px)",
                                "overflowY": "hidden",  # Graph usually handles its own scroll
                            },
                        ),
                    ],
                    className="g-3",  # Gutters
                ),
                dcc.Interval(
                    id=self._get_id(self._DATA_POLLING_INTERVAL_ID_SUFFIX),
                    interval=self.data_polling_interval_ms,
                    n_intervals=0,
                    disabled=False,  # Initially disabled, enabled by acquirer status
                ),
                *shared_stores_for_layout,
            ],
            style={"paddingTop": "10px"},
        )
        return main_layout

    def register_callbacks(self, app: Dash, **kwargs: Any) -> None:
        """Registers all necessary Dash callbacks for this component."""
        logger.info(
            f"Registering callbacks for VideoModeComponent '{self.component_id}'"
        )

        # Orchestrator stores that tab controllers might need to read from or influence
        orchestrator_stores_for_tabs: Dict[str, Any] = {
            VideoModeComponent.LATEST_PROCESSED_DATA_STORE_SUFFIX: self._get_store_id(
                self.LATEST_PROCESSED_DATA_STORE_SUFFIX
            ),
            VideoModeComponent._MAIN_STATUS_ALERT_ID_SUFFIX: self._get_id(
                self._MAIN_STATUS_ALERT_ID_SUFFIX
            ),
            VideoModeComponent._TABS_ID_SUFFIX: self._get_id(self._TABS_ID_SUFFIX),
            VideoModeComponent.VIEWER_DATA_STORE_SUFFIX: self._get_store_id(
                self.VIEWER_DATA_STORE_SUFFIX
            ),
        }
        # Shared viewer stores that tab controllers will write to
        shared_viewer_store_ids_for_tabs = {
            "viewer_data_store": self._get_store_id(self.VIEWER_DATA_STORE_SUFFIX),
            "layout_config_store": self._get_store_id(
                self.VIEWER_LAYOUT_CONFIG_STORE_SUFFIX
            ),
        }

        self.shared_viewer.register_callbacks(
            app,
            viewer_data_store_id=shared_viewer_store_ids_for_tabs["viewer_data_store"],
            layout_config_store_id=shared_viewer_store_ids_for_tabs[
                "layout_config_store"
            ],
        )

        shared_viewer_graph_actual_id = self.shared_viewer.get_graph_id()

        for tc in self.tab_controllers:
            callback_kwargs_for_tab = {
                "orchestrator_stores": orchestrator_stores_for_tabs,
                "shared_viewer_store_ids": shared_viewer_store_ids_for_tabs,
            }
            if isinstance(tc, AnnotationTabController):
                callback_kwargs_for_tab["shared_viewer_graph_id"] = (
                    shared_viewer_graph_actual_id
                )

            tc.register_callbacks(app, **callback_kwargs_for_tab)  # type: ignore

        self._register_data_polling_interval_callback(app)
        self._register_tab_switching_callback(app)

        # Register save button callback
        @app.callback(
            Output(
                self._get_id(self._MAIN_STATUS_ALERT_ID_SUFFIX),
                "children",
                allow_duplicate=True,
            ),
            Input(self._get_id("save-button"), "n_clicks"),
            State(self._get_store_id(self.VIEWER_DATA_STORE_SUFFIX), "data"),
            prevent_initial_call=True,
        )
        def handle_save_button(
            n_clicks, current_viewer_data_ref: Optional[Dict[str, str]]
        ):
            if not n_clicks:
                raise PreventUpdate

            if current_viewer_data_ref and "key" in current_viewer_data_ref:
                results_key = current_viewer_data_ref["key"]
            else:
                results_key = data_registry.LIVE_DATA_KEY

            try:
                details = self.save(results_key=results_key)
            except Exception as e:
                message = (
                    f"Error saving data, please configure the data_root_folder. "
                    "To change the data_root_folder, please run "
                    "**qualibrate-config config --storage-location "
                    "<your-desired-root-folder>}**"
                )
                logger.error(f"{message}: \n{e}")
                return dbc.Alert(
                    dcc.Markdown(message),
                    color="danger",
                    dismissable=True,
                )

            message = f"Data **#{details['idx']}** saved to {details['path']}."
            if ".qualibrate" in str(details["path"]):
                message += "To change the data root folder, please run "
                "**qualibrate-config config --storage-location "
                "<your-desired-root-folder>}**"

            return dbc.Alert(
                dcc.Markdown(message),
                color="success",
                dismissable=True,
                # No duration: stays until dismissed or replaced
            )

    def _register_data_polling_interval_callback(self, app: Dash) -> None:
        """Registers callback for periodically polling the data acquirer."""

        @app.callback(
            Output(self._get_store_id(self.LATEST_PROCESSED_DATA_STORE_SUFFIX), "data"),
            Output(
                self._get_store_id(self.VIEWER_DATA_STORE_SUFFIX),
                "data",
                allow_duplicate=True,  # Allow update from here and tab switch
            ),
            Output(
                self._get_id(self._MAIN_STATUS_ALERT_ID_SUFFIX),
                "children",
                allow_duplicate=True,
            ),
            Input(self._get_id(self._DATA_POLLING_INTERVAL_ID_SUFFIX), "n_intervals"),
            State(self._get_store_id(self.VIEWER_DATA_STORE_SUFFIX), "data"),
            prevent_initial_call=True,
        )
        def handle_data_polling(
            _n_intervals: int, current_viewer_data_ref: Optional[Dict[str, str]]
        ) -> Tuple[Any, Any, Any]:
            if not ctx.triggered_id:
                raise PreventUpdate

            acquirer_response = self.data_acquirer.get_latest_data()
            data_object = acquirer_response.get("data")
            error = acquirer_response.get("error")
            status = acquirer_response.get("status", "unknown")
            self._last_known_acquirer_status = status

            alert_children: Any = no_update
            latest_processed_data_ref_for_store: Any = no_update
            viewer_primary_data_ref_update: Any = no_update

            if error:
                logger.error(f"Acquisition error from acquirer: {error}")
                alert_children = dbc.Alert(
                    f"Acquisition Error: {str(error)[:200]}",
                    color="danger",
                    dismissable=True,
                    # No duration: stays until dismissed or replaced
                )
            elif status == "error" and not error:
                alert_children = dbc.Alert(
                    "Acquisition system in error state.",
                    color="warning",
                    dismissable=True,
                    # No duration: stays until dismissed or replaced
                )

            if data_object is not None and status == "running":
                new_version = data_registry.set_data(
                    data_registry.LIVE_DATA_KEY,
                    {
                        "base_image_data": data_object,
                        "annotations": {"points": [], "lines": []},
                    },
                )
                self._current_live_data_version = new_version

                data_reference = {
                    "key": data_registry.LIVE_DATA_KEY,
                    "version": new_version,
                }
                latest_processed_data_ref_for_store = data_reference

                current_viewer_key = (
                    current_viewer_data_ref.get("key")
                    if current_viewer_data_ref
                    else None
                )
                if current_viewer_key == data_registry.LIVE_DATA_KEY:
                    viewer_primary_data_ref_update = data_reference

            return (
                latest_processed_data_ref_for_store,
                viewer_primary_data_ref_update,
                alert_children,
            )

    def _register_tab_switching_callback(self, app: Dash) -> None:
        """Registers callback for handling tab activation and deactivation."""
        outputs = [
            Output(
                self._get_store_id(self.VIEWER_DATA_STORE_SUFFIX),
                "data",
                allow_duplicate=True,
            ),
            Output(
                self._get_store_id(self.VIEWER_LAYOUT_CONFIG_STORE_SUFFIX),
                "data",
                allow_duplicate=True,
            ),
        ]
        inputs = [Input(self._get_id(self._TABS_ID_SUFFIX), "value")]

        @app.callback(outputs, inputs, prevent_initial_call=True)
        def handle_tab_switch(new_active_tab_value: str) -> Tuple[Any, Any]:
            if not self.tab_controllers or not new_active_tab_value:
                logger.warning("Tab switch triggered with no tabs or no new value.")
                return no_update, no_update

            # Only call deactivate if the tab actually changed from a known previous one
            if ctx.triggered_id and self._active_tab_value != new_active_tab_value:
                prev_tc = next(
                    (
                        tc
                        for tc in self.tab_controllers
                        if tc.get_tab_value() == self._active_tab_value
                    ),
                    None,
                )
                if prev_tc:
                    prev_tc.on_tab_deactivated()

            active_tc = next(
                (
                    tc
                    for tc in self.tab_controllers
                    if tc.get_tab_value() == new_active_tab_value
                ),
                None,
            )
            if not active_tc:
                logger.warning(
                    f"No tab controller found for tab value: {new_active_tab_value}"
                )
                return no_update, no_update

            active_tc.is_active = True

            store_updates_from_tab = active_tc.on_tab_activated()
            self._active_tab_value = new_active_tab_value  # Update active tab

            viewer_data_update = store_updates_from_tab.get(
                self.VIEWER_DATA_STORE_SUFFIX, no_update
            )
            layout_config_update = store_updates_from_tab.get(
                self.VIEWER_LAYOUT_CONFIG_STORE_SUFFIX, no_update
            )

            # If the new mode is 'live' (from tab activation), ensure primary_update
            # points to the latest known live data version.
            if (
                isinstance(viewer_data_update, dict)
                and viewer_data_update.get("key") == data_registry.LIVE_DATA_KEY
            ):
                if self._current_live_data_version:  # A version from polling exists
                    viewer_data_update = {
                        "key": data_registry.LIVE_DATA_KEY,
                        "version": self._current_live_data_version,
                    }

            return viewer_data_update, layout_config_update

    def save(self, results_key: str) -> Dict[str, Any]:
        image_data = data_registry.get_data(results_key)

        if image_data is None:
            logger.error(
                f"Error saving results: {results_key} not found in data registry."
            )
            live_data = data_registry.get_data(data_registry.LIVE_DATA_KEY)
            static_data = data_registry.get_data(data_registry.STATIC_DATA_KEY)
            if isinstance(live_data, dict):
                image_data = live_data
            elif isinstance(static_data, dict):
                image_data = static_data
            else:
                raise ValueError(
                    "No data found in data registry for saving. "
                    "Please provide data to save."
                )
        results = {"image_data": image_data}

        # Add plotly graphs from the shared viewer
        try:
            # Temporary import check since qualang tools does not yet support plotly
            from qualang_tools.results.data_handler.data_processors import (
                plotly_graph_saver,
            )

            results["figures"] = self.shared_viewer.get_figures()
        except ImportError:
            pass
        return save_data(results)

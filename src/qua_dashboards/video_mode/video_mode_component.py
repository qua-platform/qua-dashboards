import logging
from datetime import datetime
from pathlib import Path
from typing import Union

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ALL, dcc, html
from dash.exceptions import PreventUpdate
import xarray as xr

from qua_dashboards.video_mode.data_acquirers import BaseDataAcquirer
from qua_dashboards.video_mode.dash_tools import xarray_to_plotly
from qua_dashboards.core import BaseComponent, BaseUpdatableComponent

from qua_dashboards.video_mode.annotation import AnnotationComponent

logger = logging.getLogger(__name__)

__all__ = ["VideoModeComponent"]


class VideoModeComponent(BaseComponent):
    """
    Dashboard component for video mode with optional annotation enhancement.

    Displays live data acquisition, controls parameters, and can integrate
    an AnnotationComponent to provide snapshot annotation features within
    a tabbed interface.

    Attributes:
        data_acquirer: The data acquisition backend.
        component_id: Unique ID for Dash elements.
        save_path: Path for saving live data/images.
        update_interval_ms: Update interval in milliseconds.
        include_update_button: Whether to show the 'Update Params' button.
        annotation_enhancer: An AnnotationComponent instance.
    """

    # Define constants for element IDs
    # Internal VMC elements
    _LIVE_HEATMAP_ID = "live-heatmap"
    _INTERVAL_ID = "interval-component"
    _ITERATION_OUTPUT_ID = "iteration-output"
    _MANUAL_PAUSE_BUTTON_ID = "manual-pause-button"
    _UPDATE_PARAMS_BUTTON_ID = "update-button"
    _SAVE_LIVE_BUTTON_ID = "save-live-button"
    _LIVE_GRAPH_CONTAINER_ID = "live-graph-container"
    _SETTINGS_PANEL_CONTAINER_ID = "settings-panel-container"
    # Elements related to annotation enhancement
    _SNAPSHOT_BUTTON_ID = "snapshot-button"
    _TABS_ID = "control-tabs"
    _SETTINGS_TAB_ID = "tab-settings"
    _ANNOTATION_TAB_ID = "tab-annotation"
    # Shared Stores (managed by VMC when enhancer is present)
    _SNAPSHOT_STORE_ID = "snapshot-store"
    _MODE_SIGNAL_STORE_ID = (
        "mode-signal-store"  # Used by AnnotationComponent to signal return
    )

    def __init__(
        self,
        data_acquirer: BaseDataAcquirer,
        component_id: str = "video-mode",
        save_path: Union[str, Path] = "./video_mode_output",
        update_interval_sec: float = 0.1,
        include_update_button: bool = True,
    ):
        """
        Initializes the VideoModeComponent.

        Args:
            data_acquirer: Instance for data acquisition.
            component_id: Unique Dash ID prefix.
            save_path: Directory for saving live snapshots.
            update_interval_sec: Live update interval.
            include_update_button: Show button to update acquirer params.
        """
        super().__init__(component_id=component_id)
        if not isinstance(data_acquirer, BaseDataAcquirer):
            raise TypeError(
                "data_acquirer must be an instance of BaseDataAcquirer or subclass."
            )

        self.data_acquirer = data_acquirer
        self.save_path = Path(save_path)
        self.update_interval_ms = int(update_interval_sec * 1000)
        self.include_update_button = include_update_button
        self.annotation_enhancer = AnnotationComponent()  # Store the enhancer

        logger.info(f"Initializing VideoModeComponent ({self.component_id=})")
        logger.info(f"Using Data Acquirer: {type(data_acquirer).__name__}")

        # --- Internal State ---
        self._manual_paused: bool = False
        self._last_save_clicks: int = 0
        self._last_update_clicks: int = 0
        # Initial plot
        try:
            self.fig = xarray_to_plotly(self.data_acquirer.data_array)
        except Exception as e:
            logger.error(f"Failed to create initial plot: {e}", exc_info=True)
            self.fig = go.Figure()

    # --- Layout Methods ---

    def _get_settings_panel_layout(self) -> html.Div:
        """Generates the layout for the settings controls ONLY."""
        # --- VMC-Specific Controls ---
        vmc_controls = [
            html.H4("Live Video Mode", className="mb-3"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Pause Live",
                            id=self._get_id(self._MANUAL_PAUSE_BUTTON_ID),
                            n_clicks=0,
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        html.Div(
                            f"Iteration: {self.data_acquirer.num_acquisitions}",
                            id=self._get_id(self._ITERATION_OUTPUT_ID),
                            className="d-flex align-items-center",
                        ),
                        width="auto",
                    ),
                ],
                className="mb-3",
            ),
        ]

        # --- Data Acquirer Controls ---
        acquirer_controls = []
        try:
            if isinstance(self.data_acquirer, BaseUpdatableComponent):
                acquirer_controls = self.data_acquirer.get_dash_components(
                    include_subcomponents=True
                )
            else:
                logger.warning(
                    f"Data acquirer {type(self.data_acquirer)} does not provide Dash components."
                )
                acquirer_controls = [html.Div("Data acquirer controls unavailable.")]
        except Exception as e:
            logger.error(
                f"Error getting controls from Data Acquirer: {e}", exc_info=True
            )
            acquirer_controls = [html.Div("Error loading data acquirer controls.")]

        # --- Action Buttons ---
        action_buttons_list = [
            dbc.Col(
                dbc.Button(
                    "Save Live", id=self._get_id(self._SAVE_LIVE_BUTTON_ID), n_clicks=0
                ),
                width="auto",
            ),
        ]
        # Add Snapshot button only if enhancer is present
        if self.annotation_enhancer:
            action_buttons_list.append(
                dbc.Col(
                    dbc.Button(
                        "Snapshot & Annotate",
                        id=self._get_id(self._SNAPSHOT_BUTTON_ID),
                        n_clicks=0,
                        color="info",
                    ),
                    width="auto",
                )
            )
        # Conditionally add Update button
        if self.include_update_button:
            action_buttons_list.append(
                dbc.Col(
                    dbc.Button(
                        "Update Params",
                        id=self._get_id(self._UPDATE_PARAMS_BUTTON_ID),
                        n_clicks=0,
                    ),
                    width="auto",
                )
            )
        action_buttons = dbc.Row(action_buttons_list, className="mt-3 g-2")

        settings_panel_elements = vmc_controls + acquirer_controls + [action_buttons]
        # Return wrapped in a container div
        return html.Div(
            settings_panel_elements, id=self._get_id(self._SETTINGS_PANEL_CONTAINER_ID)
        )

    def _get_graph_panel_layout(self) -> html.Div:
        """Generates the layout for the live graph display area ONLY."""
        return html.Div(
            id=self._get_id(self._LIVE_GRAPH_CONTAINER_ID),
            style={
                "height": "100%",
                "width": "100%",
                "display": "block",
            },  # Initially visible
            children=[
                dcc.Graph(
                    id=self._get_id(self._LIVE_HEATMAP_ID),
                    figure=self.fig,
                    style={"height": "100%", "min-height": "400px"},
                    config={"responsive": True},
                ),
                dcc.Interval(
                    id=self._get_id(self._INTERVAL_ID),
                    interval=self.update_interval_ms,
                    n_intervals=0,
                    disabled=self._manual_paused,
                ),
            ],
        )

    def get_layout(self) -> html.Div:
        """
        Generates the complete layout for the component, including tabs if enhanced.
        """
        logger.debug(f"Generating full layout for {self.component_id}")

        # --- Create Tabs ---
        settings_tab = dcc.Tab(
            self._get_settings_panel_layout(),
            label="Settings",
            value=self._SETTINGS_TAB_ID,  # Can't use get_id() here
        )
        tabs_list = [settings_tab]
        annotation_layout = html.Div()  # Placeholder if no enhancer

        if self.annotation_enhancer:
            try:
                # Get controls layout from enhancer
                enhancer_controls = self.annotation_enhancer.get_controls_layout()
                annotation_tab = dcc.Tab(
                    enhancer_controls,
                    label="Annotations & Analysis",
                    value=self._ANNOTATION_TAB_ID,  # Can't use get_id() here
                )
                tabs_list.append(annotation_tab)
                # Get the main analysis container layout from enhancer
                annotation_layout = self.annotation_enhancer.get_layout()

            except AttributeError:
                logger.error(
                    f"Annotation enhancer {type(self.annotation_enhancer)} "
                    "is missing get_controls_layout() or get_layout() method."
                )
                # Add a placeholder tab indicating error
                tabs_list.append(
                    dcc.Tab("Error loading annotations", label="Annotations")
                )
            except Exception as e:
                logger.error(
                    f"Error getting layout from annotation enhancer: {e}", exc_info=True
                )
                tabs_list.append(
                    dcc.Tab("Error loading annotations", label="Annotations")
                )

        # --- Assemble Full Layout ---
        control_panel = dcc.Tabs(
            id=self._get_id(self._TABS_ID),
            value=self._SETTINGS_TAB_ID,
            children=tabs_list,
            className="mb-3",
        )

        display_panel = html.Div(
            [
                self._get_graph_panel_layout(),
                annotation_layout,  # Add enhancer's main layout (initially hidden)
            ],
            style={"position": "relative", "height": "100%"},
        )  # Container for graph panels

        # Add shared stores if enhancer is present
        shared_stores = []
        if self.annotation_enhancer:
            shared_stores = [
                dcc.Store(
                    id=self._get_id(self._SNAPSHOT_STORE_ID), storage_type="memory"
                ),
                dcc.Store(
                    id=self._get_id(self._MODE_SIGNAL_STORE_ID),
                    storage_type="memory",
                    data=0,
                ),
            ]

        # Main structure
        full_layout = html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            control_panel, width=12, lg=5, className="mb-3 mb-lg-0"
                        ),
                        # dbc.Col(tabs_list, width=12, lg=5, className="mb-3 mb-lg-0"),
                        dbc.Col(display_panel, width=12, lg=7),
                    ],
                    className="g-3",
                    style={"height": "calc(100vh - 50px)"},  # Example height adjustment
                ),
                *shared_stores,  # Unpack stores into the layout
            ],
            id=self.component_id,
            style={"height": "100vh", "padding": "15px"},  # Example padding
        )

        return full_layout

    # --- Callback Registration ---

    def register_callbacks(self, app: Dash):
        """Registers all callbacks for the component and its enhancer."""
        logger.debug(f"Registering callbacks for {self.component_id}")

        # Register internal VMC callbacks
        self._register_manual_pause_callback(app)
        self._register_heatmap_update_callback(app)
        self._register_parameter_update_callback(app)
        self._register_save_live_callback(app)

        # Register orchestration and enhancer callbacks if enhancer exists
        if self.annotation_enhancer:
            self._register_orchestration_callbacks(app)
            try:
                # Pass the actual IDs of the shared stores to the enhancer
                orchestrator_stores = {
                    "snapshot_store": self._get_id(self._SNAPSHOT_STORE_ID),
                    "signal_store": self._get_id(self._MODE_SIGNAL_STORE_ID),
                    # Add other stores if needed (e.g., matrix update store)
                }
                self.annotation_enhancer.register_callbacks(app, orchestrator_stores)
                logger.info(
                    f"Registered callbacks for enhancer: {type(self.annotation_enhancer).__name__}"
                )
            except AttributeError:
                logger.error(
                    f"Annotation enhancer {type(self.annotation_enhancer)} "
                    "is missing register_callbacks() method."
                )
            except Exception as e:
                logger.error(
                    f"Error registering callbacks for annotation enhancer: {e}",
                    exc_info=True,
                )

    # --- Internal VMC Callbacks --- (Mostly unchanged from previous version)

    def _register_manual_pause_callback(self, app: Dash):
        """Callback for the manual Pause/Resume button."""

        @app.callback(
            Output(self._get_id(self._MANUAL_PAUSE_BUTTON_ID), "children"),
            Output(self._get_id(self._INTERVAL_ID), "disabled"),
            Input(self._get_id(self._MANUAL_PAUSE_BUTTON_ID), "n_clicks"),
            State(self._get_id(self._INTERVAL_ID), "disabled"),
            prevent_initial_call=True,
        )
        def toggle_manual_pause(n_clicks, is_disabled):
            if n_clicks is None:
                raise PreventUpdate
            newly_disabled = not is_disabled
            self._manual_paused = newly_disabled
            logger.info(
                f"Component {self.component_id} manual pause toggled: {newly_disabled}"
            )
            button_text = "Resume Live" if newly_disabled else "Pause Live"
            return button_text, newly_disabled

    def _register_heatmap_update_callback(self, app: Dash):
        """Callback for Live Update triggered by interval"""

        @app.callback(
            Output(self._get_id(self._LIVE_HEATMAP_ID), "figure"),
            Output(self._get_id(self._ITERATION_OUTPUT_ID), "children"),
            Input(self._get_id(self._INTERVAL_ID), "n_intervals"),
            State(self._get_id(self._INTERVAL_ID), "disabled"),
            prevent_initial_call=True,
        )
        def update_heatmap(n_intervals, is_disabled):
            if is_disabled:
                raise PreventUpdate
            t_start = datetime.now()
            try:
                current_data_array = self.data_acquirer.update_data()
                self.fig = xarray_to_plotly(current_data_array)
                iteration_text = f"Iteration: {self.data_acquirer.num_acquisitions}"
                logger.debug(
                    f"{self.component_id} plot updated in {(datetime.now() - t_start).total_seconds():.3f}s"
                )
                return self.fig, iteration_text
            except Exception as e:
                logger.error(
                    f"Error during data acquisition/plotting for {self.component_id}: {e}",
                    exc_info=True,
                )
                iteration_text = (
                    f"Iteration: {self.data_acquirer.num_acquisitions} (Error)"
                )
                raise PreventUpdate  # Avoid updating graph with error

    def _register_parameter_update_callback(self, app: Dash):
        """Callback for Parameter Updates button."""
        if not self.include_update_button or not isinstance(
            self.data_acquirer, BaseUpdatableComponent
        ):
            return

        all_param_states = []
        acquirer_component_ids = []  # Store IDs for use in callback logic
        try:
            acquirer_component_ids = self.data_acquirer.get_component_ids() or []
            for da_comp_id_pattern in acquirer_component_ids:
                all_param_states.extend(
                    [
                        State({"type": da_comp_id_pattern, "index": ALL}, "id"),
                        State({"type": da_comp_id_pattern, "index": ALL}, "value"),
                    ]
                )
        except Exception as e:
            logger.error(
                f"Failed to get component IDs/states from Data Acquirer: {e}",
                exc_info=True,
            )
            return

        if not all_param_states:
            return

        @app.callback(
            Output(self._get_id(self._LIVE_HEATMAP_ID), "figure", allow_duplicate=True),
            Output(
                self._get_id(self._ITERATION_OUTPUT_ID),
                "children",
                allow_duplicate=True,
            ),
            Input(self._get_id(self._UPDATE_PARAMS_BUTTON_ID), "n_clicks"),
            *all_param_states,  # Unpack states here
            prevent_initial_call=True,
        )
        def update_parameters(n_update_clicks, *component_states):
            if n_update_clicks is None or n_update_clicks <= self._last_update_clicks:
                raise PreventUpdate
            self._last_update_clicks = n_update_clicks

            logger.info(f"Update parameters triggered for {self.component_id}")
            params_by_component_type = {}
            states_iter = iter(component_states)
            for da_comp_id_pattern in acquirer_component_ids:  # Use stored IDs
                try:
                    ids = next(states_iter, [])
                    values = next(states_iter, [])
                    if not ids:
                        continue
                    params_by_component_type[da_comp_id_pattern] = {
                        id_dict["index"]: value
                        for id_dict, value in zip(ids, values)
                        if isinstance(id_dict, dict) and "index" in id_dict
                    }
                except StopIteration:
                    break
                except Exception as e:
                    logger.error(
                        f"Error processing states for {da_comp_id_pattern}: {e}"
                    )

            if not params_by_component_type:
                raise PreventUpdate

            logger.debug(
                f"Updating Data Acquirer with parameters: {params_by_component_type}"
            )
            try:
                self.data_acquirer.update_parameters(params_by_component_type)
                logger.info(f"Data Acquirer parameters updated for {self.component_id}")
                # Force refresh
                current_data_array = self.data_acquirer.update_data()
                self.fig = xarray_to_plotly(current_data_array)
                iteration_text = f"Iteration: {self.data_acquirer.num_acquisitions}"
                return self.fig, iteration_text
            except Exception as e:
                logger.error(
                    f"Error updating params/refreshing plot for {self.component_id}: {e}",
                    exc_info=True,
                )
                raise PreventUpdate

    def _register_save_live_callback(self, app: Dash):
        """Callback for Save Live Button."""

        @app.callback(
            Output(self._get_id(self._SAVE_LIVE_BUTTON_ID), "children"),
            Input(self._get_id(self._SAVE_LIVE_BUTTON_ID), "n_clicks"),
            State(self._get_id(self._LIVE_HEATMAP_ID), "figure"),
            prevent_initial_call=True,
        )
        def save_live_data_and_image(n_clicks, current_figure):
            if n_clicks is None or n_clicks <= self._last_save_clicks:
                raise PreventUpdate
            logger.info(f"Save Live triggered for {self.component_id}")
            self._last_save_clicks = n_clicks
            try:
                fig_obj = go.Figure(current_figure)
                idx = self._save_live_snapshot(figure_to_save=fig_obj)
                logger.info(
                    f"Live data/image saved index {idx} for {self.component_id}"
                )
                return f"Saved Live ({idx})"
            except Exception as e:
                logger.error(
                    f"Error saving live data/image for {self.component_id}: {e}",
                    exc_info=True,
                )
                return "Save Failed"

    # --- Orchestration Callbacks (Only registered if enhancer exists) ---

    def _register_orchestration_callbacks(self, app: Dash):
        """Registers callbacks to manage mode switching and data passing."""

        # Callback to take snapshot and switch to analysis mode
        @app.callback(
            Output(self._get_id(self._SNAPSHOT_STORE_ID), "data"),
            Output(self._get_id(self._INTERVAL_ID), "disabled", allow_duplicate=True),
            Output(self._get_id(self._TABS_ID), "value"),
            Output(self._get_id(self._LIVE_GRAPH_CONTAINER_ID), "style"),
            Output(
                self.annotation_enhancer._get_id("analysis-container"), "style"
            ),  # Target enhancer's container
            Input(self._get_id(self._SNAPSHOT_BUTTON_ID), "n_clicks"),
            State(self._get_id(self._LIVE_HEATMAP_ID), "figure"),
            prevent_initial_call=True,
        )
        def take_snapshot_and_switch_mode(n_clicks, current_figure_dict):
            if n_clicks is None:
                raise PreventUpdate

            logger.info(
                f"Snapshot triggered for {self.component_id}. Switching to Analysis Mode."
            )

            # Prepare snapshot data (currently just the figure)
            # TODO: Consider adding self.data_acquirer.data_array if needed by analysis
            snapshot_data = {"figure": current_figure_dict}

            # Disable live updates, switch tab, switch visible graph container
            return (
                snapshot_data,  # Output snapshot data
                True,  # Output: Disable interval
                self._ANNOTATION_TAB_ID,  # Output: Set active tab
                {
                    "display": "none",
                    "height": "100%",
                    "width": "100%",
                },  # Output: Hide live graph
                {
                    "display": "block",
                    "height": "100%",
                    "width": "100%",
                },  # Output: Show analysis graph
            )

        # Callback to return to live mode
        @app.callback(
            Output(self._get_id(self._INTERVAL_ID), "disabled", allow_duplicate=True),
            Output(self._get_id(self._TABS_ID), "value", allow_duplicate=True),
            Output(
                self._get_id(self._LIVE_GRAPH_CONTAINER_ID),
                "style",
                allow_duplicate=True,
            ),
            Output(
                self.annotation_enhancer._get_id("analysis-container"),
                "style",
                allow_duplicate=True,
            ),
            Input(
                self._get_id(self._MODE_SIGNAL_STORE_ID), "data"
            ),  # Triggered by enhancer's return button
            prevent_initial_call=True,
        )
        def return_to_live_mode(signal_data):
            if signal_data is None or signal_data == 0:  # Check if signal is valid
                raise PreventUpdate

            logger.info(f"Return to Live Mode triggered for {self.component_id}.")

            # Re-enable updates (unless manually paused), switch tab, switch visible graph
            return (
                self._manual_paused,  # Respect manual pause state when returning
                self._SETTINGS_TAB_ID,
                {
                    "display": "block",
                    "height": "100%",
                    "width": "100%",
                },  # Show live graph
                {
                    "display": "none",
                    "height": "100%",
                    "width": "100%",
                },  # Hide analysis graph
            )

    # --- Internal Saving Methods --- (Unchanged from previous version)

    def _find_next_save_index(self) -> int:
        self.save_path.mkdir(parents=True, exist_ok=True)
        idx = 1
        data_save_path = self.save_path / "live_data"
        image_save_path = self.save_path / "live_images"
        while (data_save_path / f"data_{idx:04d}.nc").exists() or (
            image_save_path / f"image_{idx:04d}.png"
        ).exists():
            idx += 1
            if idx > 9999:
                raise ValueError("Max save index reached.")
        return idx

    def _save_live_data(self, idx: int) -> None:
        data_save_path = self.save_path / "live_data"
        data_save_path.mkdir(parents=True, exist_ok=True)
        filepath = data_save_path / f"data_{idx:04d}.nc"
        logger.debug(f"Saving live data to {filepath}")
        try:
            current_data = self.data_acquirer.data_array
            if isinstance(current_data, xr.DataArray):
                current_data.to_netcdf(filepath)
            else:
                raise TypeError("Invalid data type")
        except Exception as e:
            logger.error(f"Failed save: {e}")
            raise

    def _save_live_image(self, idx: int, figure_to_save: go.Figure) -> None:
        image_save_path = self.save_path / "live_images"
        image_save_path.mkdir(parents=True, exist_ok=True)
        filepath = image_save_path / f"image_{idx:04d}.png"
        logger.debug(f"Saving live image to {filepath}")
        try:
            if isinstance(figure_to_save, go.Figure):
                figure_to_save.write_image(filepath)
            else:
                raise TypeError("Invalid figure type")
        except Exception as e:
            logger.error(f"Failed save: {e}")
            raise

    def _save_live_snapshot(self, figure_to_save: go.Figure) -> int:
        idx = self._find_next_save_index()
        self._save_live_data(idx)
        self._save_live_image(idx, figure_to_save)
        logger.info(f"Live snapshot save index: {idx}")
        return idx

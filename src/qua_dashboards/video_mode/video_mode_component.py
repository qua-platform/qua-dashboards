import logging
from datetime import datetime
from pathlib import Path
from typing import Union

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ALL, dcc, html, ctx, no_update
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
    # Elements related to annotation enhancement
    _TABS_ID = "control-tabs"
    _LIVE_VIEW_TAB_ID = "tab-live-view"  # Renamed from _SETTINGS_TAB_ID
    _ANNOTATION_TAB_ID = "tab-annotate"  # Renamed for clarity
    # Shared Stores (managed by VMC when enhancer is present)
    _SNAPSHOT_STORE_ID = "snapshot-store"

    def __init__(
        self,
        data_acquirer: BaseDataAcquirer,
        component_id: str = "video-mode",
        save_path: Union[str, Path] = "./video_mode_output",
        update_interval_sec: float = 0.1,
        include_update_button: bool = True,
        layout_columns: int = 12,
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
        self.layout_columns = layout_columns
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
                    f"Data acquirer {type(self.data_acquirer)} does not provide Dash "
                    f"components."
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
        return html.Div(settings_panel_elements)

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
        live_view_tab = dcc.Tab(
            self._get_settings_panel_layout(),
            label="Live view",  # Renamed
            value=self._LIVE_VIEW_TAB_ID,
        )
        tabs_list = [live_view_tab]
        annotation_layout_children = html.Div()  # Placeholder

        if self.annotation_enhancer:
            try:
                enhancer_controls = self.annotation_enhancer.get_controls_layout()
                annotation_tab = dcc.Tab(
                    enhancer_controls,
                    label="Annotate",  # Renamed
                    value=self._ANNOTATION_TAB_ID,
                )
                tabs_list.append(annotation_tab)
                annotation_layout_children = self.annotation_enhancer.get_layout()
            except AttributeError:
                logger.error(
                    f"Annotation enhancer {type(self.annotation_enhancer)} "
                    "is missing get_controls_layout() or get_layout() method."
                )
                tabs_list.append(
                    dcc.Tab(
                        "Error loading annotations",
                        label="Annotations",
                        value="tab-error-anno",
                    )
                )
            except Exception as e:
                logger.error(
                    f"Error getting layout from annotation enhancer: {e}", exc_info=True
                )
                tabs_list.append(
                    dcc.Tab(
                        "Error loading annotations",
                        label="Annotations",
                        value="tab-error-anno",
                    )
                )

        # --- Assemble Full Layout ---
        control_panel = dcc.Tabs(
            id=self._get_id(self._TABS_ID),
            value=self._LIVE_VIEW_TAB_ID,  # Default to live view
            children=tabs_list,
            className="mb-3",
        )

        display_panel = html.Div(
            [
                self._get_graph_panel_layout(),
                annotation_layout_children,  # Add enhancer's main layout
            ],
            style={"position": "relative", "height": "100%"},
        )

        shared_stores = []
        if self.annotation_enhancer:
            shared_stores = [
                dcc.Store(
                    id=self._get_id(self._SNAPSHOT_STORE_ID), storage_type="memory"
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
        self._register_tab_navigation_callback(app)

        # Register orchestration and enhancer callbacks if enhancer exists
        if self.annotation_enhancer:
            try:
                # Pass the actual IDs of the shared stores to the enhancer
                orchestrator_stores = {
                    "snapshot_store": self._get_id(self._SNAPSHOT_STORE_ID),
                }
                self.annotation_enhancer.register_callbacks(app, orchestrator_stores)
                logger.info(
                    f"Registered callbacks for enhancer: "
                    f"{type(self.annotation_enhancer).__name__}"
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

    def _register_tab_navigation_callback(self, app: Dash):
        """
        Central callback to manage UI state based on active tab.
        Handles snapshotting when 'Annotate' tab is selected, and
        resuming live view when 'Live view' tab is selected.
        """
        outputs = [
            Output(self._get_id(self._INTERVAL_ID), "disabled", allow_duplicate=True),
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
        ]
        if self.annotation_enhancer:
            outputs.append(
                Output(
                    self._get_id(self._SNAPSHOT_STORE_ID), "data", allow_duplicate=True
                )
            )

        @app.callback(
            *outputs,
            Input(self._get_id(self._TABS_ID), "value"),
            State(self._get_id(self._LIVE_HEATMAP_ID), "figure"),
            prevent_initial_call=True,  # Important to avoid issues on load
        )
        def handle_tab_change(active_tab_value, current_live_figure):
            if not active_tab_value or not ctx.triggered_id:
                raise PreventUpdate

            logger.info(f"Tab changed to: {active_tab_value}")

            # Default outputs (no change)
            interval_disabled = no_update
            live_graph_style = no_update
            analysis_container_style = no_update
            snapshot_data_out = no_update

            if active_tab_value == self._LIVE_VIEW_TAB_ID:
                logger.info(f"Switching to Live View mode for {self.component_id}.")
                interval_disabled = self._manual_paused  # Respect manual pause
                live_graph_style = {
                    "display": "block",
                    "height": "100%",
                    "width": "100%",
                }
                analysis_container_style = {
                    "display": "none",
                    "height": "100%",
                    "width": "100%",
                }
                # snapshot_data_out remains no_update

            elif (
                active_tab_value == self._ANNOTATION_TAB_ID and self.annotation_enhancer
            ):
                logger.info(
                    f"Switching to Annotation mode for {self.component_id}. "
                    f"Taking snapshot."
                )
                if current_live_figure:
                    snapshot_data_out = {"figure": current_live_figure}
                else:
                    logger.warning("No live figure data available to snapshot.")
                    snapshot_data_out = {"figure": go.Figure()}  # Send empty figure

                interval_disabled = True  # Pause live updates
                live_graph_style = {
                    "display": "none",
                    "height": "100%",
                    "width": "100%",
                }
                analysis_container_style = {
                    "display": "block",
                    "height": "100%",
                    "width": "100%",
                }
            else:
                # Other tabs or initial load state, do nothing specific here for now
                raise PreventUpdate

            if self.annotation_enhancer:
                return (
                    interval_disabled,
                    live_graph_style,
                    analysis_container_style,
                    snapshot_data_out,
                )
            else:  # Should not happen if _ANNOTATION_TAB_ID selected without enhancer
                return interval_disabled, live_graph_style, analysis_container_style

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

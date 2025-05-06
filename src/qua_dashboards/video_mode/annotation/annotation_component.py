# qua_dashboards/annotation/annotation_component.py
"""
Dash component for annotating static figures (snapshots).
Manages annotation state, controls, and interactions.
"""

import json
import logging
from pathlib import Path
from typing import Dict

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, ctx
from dash.exceptions import PreventUpdate

from qua_dashboards.core import BaseComponent

# Assuming annotation_tools.py is in the same directory
from .annotation_tools import (
    calculate_slopes,
    find_closest_line_index,
    generate_figure_with_annotations,
    list_annotation_files,
    load_annotation_file,
    save_annotation_file,
)

logger = logging.getLogger(__name__)

# Default state for annotation stores (using deep copies later)
DEFAULT_POINTS_STATE_STR = json.dumps(
    {
        "added_points": {"x": [], "y": [], "index": []},  # 'index' holds unique IDs
        "selected_point": {"move": False, "index": None},  # 'index' holds unique ID
    }
)
DEFAULT_LINES_STATE_STR = json.dumps(
    {
        "selected_indices": [],  # Holds unique point IDs being selected for a new line
        "added_lines": {"start_index": [], "end_index": []},  # Hold unique point IDs
    }
)

__all__ = ["AnnotationComponent"]


class AnnotationComponent(BaseComponent):
    """
    Dash component for annotating static snapshot figures.

    Handles drawing points, lines, loading/saving annotations, and
    placeholder for slope calculation. Designed to be embedded within
    a larger dashboard layout (like VideoModeComponent).
    """

    # Define constants for internal element IDs
    _ANALYSIS_CONTAINER_ID = "analysis-container"
    _ANALYSIS_GRAPH_ID = "analysis-graph"
    _POINTS_STORE_ID = "points-store"
    _LINES_STORE_ID = "lines-store"
    _BASE_FIGURE_STORE_ID = "base-figure-store"  # Stores the snapshot figure dict
    _MODE_SELECTOR_ID = "mode-selector"
    _CLEAR_BUTTON_ID = "clear-button"
    _LOAD_DROPDOWN_ID = "load-dropdown"
    _LOAD_BUTTON_ID = "load-button"
    _SAVE_BUTTON_ID = "save-button"
    _SAVE_STATUS_ID = "save-status"
    _CALCULATE_BUTTON_ID = "calculate-button"
    _ANALYSIS_RESULTS_ID = "analysis-results"
    _KEY_LISTENER_ID = "key-listener"  # For shortcuts

    def __init__(
        self,
        component_id: str = "annotation-analysis",
        save_load_path: str = "./annotations",
        point_select_tolerance: float = 0.01,  # Relative tolerance (e.g., 1% of axis range)
    ):
        """
        Initializes the AnnotationComponent.

        Args:
            component_id: Unique ID for this component instance.
            save_load_path: Directory path to save/load annotation JSON files.
            point_select_tolerance: Click tolerance (relative) for selecting
                                      existing points/lines.
        """
        super().__init__(component_id=component_id)
        self.save_load_path = Path(save_load_path)
        # Store relative tolerance, absolute value calculated based on figure
        self._relative_click_tolerance = max(0.001, point_select_tolerance)  # Min 0.1%
        self._absolute_click_tolerance = 1.0  # Default, updated when figure loads
        logger.info(
            f"AnnotationComponent '{self.component_id}' initialized. "
            f"Path: {self.save_load_path}, RelTol: {self._relative_click_tolerance}"
        )

    def _get_default_points_state(self) -> Dict:
        """Returns a deep copy of the default points state."""
        return json.loads(DEFAULT_POINTS_STATE_STR)

    def _get_default_lines_state(self) -> Dict:
        """Returns a deep copy of the default lines state."""
        return json.loads(DEFAULT_LINES_STATE_STR)

    def get_layout(self) -> html.Div:
        """
        Generates the main layout container for the annotation analysis view.

        This layout is intended to be shown/hidden by an orchestrator component.
        It contains the graph and the necessary stores.
        """
        logger.debug(f"Generating main layout for {self.component_id}")
        # Note: The main container is hidden by default via style attribute.
        # The orchestrator (e.g., VideoModeComponent) controls visibility.
        return html.Div(
            id=self._get_id(self._ANALYSIS_CONTAINER_ID),
            style={"display": "none", "height": "100%", "width": "100%"},
            children=[
                # Stores for component state
                dcc.Store(
                    id=self._get_id(self._POINTS_STORE_ID),
                    data=self._get_default_points_state(),
                ),
                dcc.Store(
                    id=self._get_id(self._LINES_STORE_ID),
                    data=self._get_default_lines_state(),
                ),
                # Store to receive the snapshot figure data from orchestrator
                dcc.Store(id=self._get_id(self._BASE_FIGURE_STORE_ID), data={}),
                # Graph for displaying static snapshot and annotations
                dcc.Graph(
                    id=self._get_id(self._ANALYSIS_GRAPH_ID),
                    figure=go.Figure(),  # Initially empty
                    style={
                        "height": "100%",
                        "width": "100%",
                    },  # Takes full container space
                    config={"scrollZoom": True},  # Enable zoom
                ),
                # Add hidden div to capture keyboard events (if needed for shortcuts)
                # This might be better placed in the main app layout by the orchestrator
                # html.Div(id=self._get_id(self._KEY_LISTENER_ID), style={'display': 'none'}),
            ],
        )

    def get_controls_layout(self) -> dbc.Card:
        """
        Generates the layout for the annotation controls panel.

        This is intended to be placed within a Tab by the orchestrator.
        """
        logger.debug(f"Generating controls layout for {self.component_id}")
        radio_options = [
            {"label": "Add/Move Points", "value": "point"},
            {"label": "Add Lines", "value": "line"},
            {"label": "Delete", "value": "delete"},
        ]
        initial_files = list_annotation_files(self.save_load_path)

        return dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Annotation Controls", className="card-title"),
                    dbc.RadioItems(
                        id=self._get_id(self._MODE_SELECTOR_ID),
                        options=radio_options,
                        value="point",  # Default mode
                        className="mb-3",
                    ),
                    dbc.Button(
                        "Clear Annotations",
                        id=self._get_id(self._CLEAR_BUTTON_ID),
                        color="warning",
                        size="sm",
                        className="mb-3 me-2",
                    ),
                    html.Hr(),
                    html.H6("Load/Save Annotations"),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Dropdown(
                                    id=self._get_id(self._LOAD_DROPDOWN_ID),
                                    placeholder="Select annotation file...",
                                    className="mb-2",
                                    options=initial_files,  # Initial load
                                ),
                                width=12,
                            ),  # Make dropdown full width initially
                        ],
                        className="mb-2",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Button(  # Move buttons below dropdown
                                    "Load",
                                    id=self._get_id(self._LOAD_BUTTON_ID),
                                    color="secondary",
                                    size="sm",
                                ),
                                width="auto",
                            ),
                            dbc.Col(
                                dbc.Button(
                                    "Save",
                                    id=self._get_id(self._SAVE_BUTTON_ID),
                                    color="success",
                                    size="sm",
                                ),
                                width="auto",
                            ),
                        ],
                        className="mb-3 g-2 justify-content-start",
                    ),  # Align buttons left
                    html.Div(
                        id=self._get_id(self._SAVE_STATUS_ID),
                        children="",
                        className="text-muted small mb-3",
                    ),
                    html.Hr(),
                    html.H6("Analysis"),
                    dbc.Button(
                        "Calculate Slopes",
                        id=self._get_id(self._CALCULATE_BUTTON_ID),
                        color="info",
                        size="sm",
                        className="mb-2 me-2",  # Margin bottom
                    ),
                    # Placeholder for displaying results if needed
                    html.Pre(
                        id=self._get_id(self._ANALYSIS_RESULTS_ID),
                        className="border rounded bg-light p-1 mb-3",
                        style={
                            "maxHeight": "100px",
                            "overflowY": "auto",
                            "fontSize": "0.8em",
                        },
                    ),
                    html.Hr(),
                ]
            ),
            className="h-100",  # Make card fill height if needed
        )

    def register_callbacks(self, app: Dash, orchestrator_stores: Dict[str, str]):
        """
        Registers callbacks for the annotation component.

        Args:
            app: The main Dash application instance.
            orchestrator_stores: A dictionary mapping store purposes (e.g.,
                                'snapshot_store') to their
                                actual IDs generated by the orchestrator.
                                Expected keys: 'snapshot_store'.
        """
        # Validate required store IDs
        if "snapshot_store" not in orchestrator_stores:
            raise ValueError("Orchestrator store ID for 'snapshot_store' is required.")

        self._register_figure_update_callback(
            app, orchestrator_stores["snapshot_store"]
        )
        self._register_interaction_callback(app)
        self._register_clear_callback(app)
        self._register_load_save_callbacks(app)
        self._register_analysis_callback(app)
        self._register_shortcut_callback(app)  # Assumes listener div is in main layout
        logger.info(f"Registered callbacks for {self.component_id}")

    def _register_figure_update_callback(self, app: Dash, snapshot_store_id: str):
        """Callback to update the static graph display when snapshot or annotations change."""

        # Update graph when a new snapshot is received
        @app.callback(
            Output(self._get_id(self._ANALYSIS_GRAPH_ID), "figure"),
            Output(
                self._get_id(self._BASE_FIGURE_STORE_ID), "data"
            ),  # Store the base figure dict
            Output(
                self._get_id(self._POINTS_STORE_ID), "data", allow_duplicate=True
            ),  # Reset annotations
            Output(
                self._get_id(self._LINES_STORE_ID), "data", allow_duplicate=True
            ),  # Reset annotations
            Input(snapshot_store_id, "data"),  # Triggered by orchestrator's store
            prevent_initial_call=True,
        )
        def update_analysis_figure_from_snapshot(snapshot_data):
            if not snapshot_data or "figure" not in snapshot_data:
                logger.warning(f"{self.component_id}: No valid snapshot data received.")
                raise PreventUpdate

            base_fig_dict = snapshot_data["figure"]
            logger.info(f"{self.component_id}: Received new snapshot data.")

            # Reset annotation state when a new snapshot is loaded
            points_state = self._get_default_points_state()
            lines_state = self._get_default_lines_state()

            # Update click tolerance based on the new figure's ranges
            try:
                temp_fig = go.Figure(base_fig_dict)
                x_range = temp_fig.layout.xaxis.range
                y_range = temp_fig.layout.yaxis.range
                if x_range and y_range:
                    diag = np.sqrt(
                        (x_range[1] - x_range[0]) ** 2 + (y_range[1] - y_range[0]) ** 2
                    )
                    if diag > 1e-9:  # Avoid division by zero if range is tiny
                        self._absolute_click_tolerance = (
                            diag * self._relative_click_tolerance
                        )
                    else:
                        self._absolute_click_tolerance = (
                            0.01  # Default small absolute value
                        )
                    logger.debug(
                        f"{self.component_id}: Absolute click tolerance set to: {self._absolute_click_tolerance:.4g}"
                    )
                else:
                    logger.warning(
                        f"{self.component_id}: Could not determine figure range for tolerance."
                    )
            except Exception as e:
                logger.warning(
                    f"{self.component_id}: Error calculating tolerance from figure: {e}"
                )

            # Generate initial figure for analysis view (with cleared annotations)
            fig = generate_figure_with_annotations(
                base_fig_dict, points_state, lines_state
            )

            return fig, base_fig_dict, points_state, lines_state

        # Redraw annotations when points/lines stores change
        @app.callback(
            Output(
                self._get_id(self._ANALYSIS_GRAPH_ID), "figure", allow_duplicate=True
            ),
            Input(self._get_id(self._POINTS_STORE_ID), "data"),
            Input(self._get_id(self._LINES_STORE_ID), "data"),
            State(
                self._get_id(self._BASE_FIGURE_STORE_ID), "data"
            ),  # Get the stored base figure
            prevent_initial_call=True,
        )
        def redraw_annotations_on_change(points_state, lines_state, base_fig_dict):
            """Redraws annotations only when points/lines stores are updated."""
            triggered = ctx.triggered_id
            # Avoid redraw if triggered by the snapshot load simultaneously
            # Check if snapshot_store_id is the trigger source ID
            # This comparison might be fragile depending on Dash version and exact ID format
            if isinstance(triggered, str) and snapshot_store_id in triggered:
                logger.debug(
                    f"{self.component_id}: Skipping annotation redraw triggered by snapshot load."
                )
                raise PreventUpdate

            if not base_fig_dict:
                logger.debug(
                    f"{self.component_id}: Skipping annotation redraw, no base figure."
                )
                raise PreventUpdate  # Don't redraw if base isn't loaded

            # logger.debug(f"{self.component_id}: Redrawing annotations.")
            fig = generate_figure_with_annotations(
                base_fig_dict, points_state, lines_state
            )
            return fig

    def _register_interaction_callback(self, app: Dash):
        """Callback to handle clicks and hovers on the analysis graph."""

        @app.callback(
            Output(self._get_id(self._POINTS_STORE_ID), "data", allow_duplicate=True),
            Output(self._get_id(self._LINES_STORE_ID), "data", allow_duplicate=True),
            Output(
                self._get_id(self._ANALYSIS_GRAPH_ID), "clickData"
            ),  # Reset clickData
            Input(self._get_id(self._ANALYSIS_GRAPH_ID), "clickData"),
            Input(self._get_id(self._ANALYSIS_GRAPH_ID), "hoverData"),
            State(self._get_id(self._MODE_SELECTOR_ID), "value"),
            State(self._get_id(self._POINTS_STORE_ID), "data"),
            State(self._get_id(self._LINES_STORE_ID), "data"),
            prevent_initial_call=True,
        )
        def handle_graph_interactions(
            click_data, hover_data, mode, points_state, lines_state
        ):
            interaction_type = list(ctx.triggered_prop_ids)[0].split(".")[-1]

            # Use deep copies to avoid modifying state directly before output
            points_state_out = json.loads(json.dumps(points_state))
            lines_state_out = json.loads(json.dumps(lines_state))
            reset_click_data = dash.no_update

            # --- Click Event ---
            if (
                interaction_type == "clickData"
                and click_data
                and click_data.get("points")
            ):
                point = click_data["points"][0]
                x, y = point["x"], point["y"]
                # Check if click was on heatmap or annotation scatter plot
                is_heatmap_click = (
                    point.get("curveNumber") == 0
                    or point.get("name") != "annotations_points"
                )
                clicked_point_id = (
                    point.get("customdata") if not is_heatmap_click else None
                )

                logger.debug(
                    f"Click: Mode={mode}, Coords=({x:.3f}, {y:.3f}), HeatmapClick={is_heatmap_click}, PointID={clicked_point_id}"
                )

                added_points = points_state_out["added_points"]
                selected_point = points_state_out[
                    "selected_point"
                ]  # Info about point being moved
                added_lines = lines_state_out["added_lines"]
                selected_indices = lines_state_out[
                    "selected_indices"
                ]  # Point IDs selected for new line

                # Determine absolute click tolerance (should be updated when figure loads)
                current_tolerance = self._absolute_click_tolerance

                # --- Point Mode ---
                if mode == "point":
                    reset_click_data = None  # Consume click
                    if selected_point["move"] and selected_point["index"] is not None:
                        # Second click: Place the selected point
                        try:
                            list_idx = added_points["index"].index(
                                selected_point["index"]
                            )
                            added_points["x"][list_idx] = x
                            added_points["y"][list_idx] = y
                            logger.debug(
                                f"Moved point ID {selected_point['index']} to ({x:.3f}, {y:.3f})"
                            )
                        except (ValueError, IndexError):
                            logger.warning(
                                f"Error finding point ID {selected_point['index']} to move."
                            )
                        # Always reset move state after placement attempt
                        selected_point["move"] = False
                        selected_point["index"] = None

                    elif clicked_point_id is not None:
                        # First click on existing point: Select it for moving
                        selected_point["move"] = True
                        selected_point["index"] = clicked_point_id
                        logger.debug(f"Selected point ID {clicked_point_id} to move.")
                    elif is_heatmap_click:
                        # Click on background: Add new point
                        new_id = (
                            max(added_points["index"]) + 1
                            if added_points["index"]
                            else 0
                        )
                        added_points["x"].append(x)
                        added_points["y"].append(y)
                        added_points["index"].append(new_id)
                        logger.debug(f"Added point ID {new_id} at ({x:.3f}, {y:.3f})")

                # --- Line Mode ---
                elif mode == "line":
                    reset_click_data = None
                    if clicked_point_id is not None:
                        if clicked_point_id not in selected_indices:
                            selected_indices.append(clicked_point_id)
                            logger.debug(
                                f"Selected point ID {clicked_point_id} for line. Current selection: {selected_indices}"
                            )
                        else:
                            selected_indices.remove(
                                clicked_point_id
                            )  # Allow unselecting
                            logger.debug(
                                f"Unselected point ID {clicked_point_id} for line. Current selection: {selected_indices}"
                            )

                        if len(selected_indices) == 2:
                            id1, id2 = selected_indices
                            # Check for duplicates
                            is_duplicate = any(
                                (s == id1 and e == id2) or (s == id2 and e == id1)
                                for s, e in zip(
                                    added_lines["start_index"], added_lines["end_index"]
                                )
                            )
                            if not is_duplicate:
                                added_lines["start_index"].append(id1)
                                added_lines["end_index"].append(id2)
                                logger.debug(
                                    f"Added line between points {id1} and {id2}"
                                )
                            else:
                                logger.debug(f"Duplicate line {id1}-{id2} not added.")
                            selected_indices.clear()  # Reset selection

                # --- Delete Mode ---
                elif mode == "delete":
                    reset_click_data = None
                    if clicked_point_id is not None:
                        # Delete the clicked point and associated lines
                        try:
                            list_index_to_del = added_points["index"].index(
                                clicked_point_id
                            )
                            del added_points["x"][list_index_to_del]
                            del added_points["y"][list_index_to_del]
                            deleted_id = added_points["index"].pop(list_index_to_del)
                            logger.debug(f"Deleted point ID {deleted_id}")

                            # Filter lines connected to the deleted point
                            original_starts = added_lines["start_index"]
                            original_ends = added_lines["end_index"]
                            new_starts, new_ends = [], []
                            for start_id, end_id in zip(original_starts, original_ends):
                                if start_id == deleted_id or end_id == deleted_id:
                                    logger.debug(
                                        f"Removing line associated with deleted point {deleted_id}"
                                    )
                                    continue
                                new_starts.append(start_id)
                                new_ends.append(end_id)
                            added_lines["start_index"] = new_starts
                            added_lines["end_index"] = new_ends

                            # Important: Do NOT re-index remaining point IDs here. IDs must remain stable.

                        except (ValueError, IndexError):
                            logger.warning(
                                f"Error finding point ID {clicked_point_id} to delete."
                            )
                    else:
                        # Clicked on background/line: Try deleting closest line
                        line_pos_idx_to_del = find_closest_line_index(
                            x, y, added_points, added_lines, current_tolerance
                        )
                        if line_pos_idx_to_del is not None:
                            try:
                                del added_lines["start_index"][line_pos_idx_to_del]
                                del added_lines["end_index"][line_pos_idx_to_del]
                                logger.debug(
                                    f"Deleted line at index {line_pos_idx_to_del}"
                                )
                            except IndexError:
                                logger.warning(
                                    f"Error deleting line index {line_pos_idx_to_del}"
                                )

            # --- Hover Event (for moving points) ---
            elif (
                interaction_type == "hoverData"
                and hover_data
                and hover_data.get("points")
            ):
                if mode == "point":
                    selected_point = points_state_out["selected_point"]
                    if selected_point["move"] and selected_point["index"] is not None:
                        # Ensure hover is on the heatmap (curveNumber 0) or background
                        hover_point = hover_data["points"][0]
                        # if hover_point.get('curveNumber') == 0: # Only track hover over background heatmap
                        x_hover, y_hover = hover_point["x"], hover_point["y"]
                        try:
                            list_idx = added_points["index"].index(
                                selected_point["index"]
                            )
                            added_points["x"][list_idx] = x_hover
                            added_points["y"][list_idx] = y_hover
                        except (ValueError, IndexError):
                            logger.warning(
                                f"Hover target point ID {selected_point['index']} not found, resetting move state."
                            )
                            selected_point["move"] = False
                            selected_point["index"] = None

            # Return updated states and reset clickData if processed
            return points_state_out, lines_state_out, reset_click_data

    def _register_clear_callback(self, app: Dash):
        """Callback to clear all annotations."""

        @app.callback(
            Output(self._get_id(self._POINTS_STORE_ID), "data", allow_duplicate=True),
            Output(self._get_id(self._LINES_STORE_ID), "data", allow_duplicate=True),
            Input(self._get_id(self._CLEAR_BUTTON_ID), "n_clicks"),
            prevent_initial_call=True,
        )
        def clear_annotations(n_clicks):
            logger.info("Clearing all annotations.")
            # Return fresh copies of defaults
            return self._get_default_points_state(), self._get_default_lines_state()

    def _register_load_save_callbacks(self, app: Dash):
        """Callbacks for loading and saving annotation files."""

        # Update dropdown options (can be triggered by interval or button if needed)
        @app.callback(
            Output(self._get_id(self._LOAD_DROPDOWN_ID), "options"),
            Input(
                self._get_id(self._LOAD_BUTTON_ID), "n_clicks"
            ),  # Refresh on load attempt
            prevent_initial_call=True,
        )
        def update_load_dropdown_options(n_clicks):
            logger.debug(f"{self.component_id}: Refreshing annotation file list.")
            return list_annotation_files(self.save_load_path)

        @app.callback(
            # Update stores on successful load
            Output(self._get_id(self._POINTS_STORE_ID), "data", allow_duplicate=True),
            Output(self._get_id(self._LINES_STORE_ID), "data", allow_duplicate=True),
            # Also update dropdown value to reflect loaded file? Optional.
            # Output(self._get_id("load-dropdown"), "value"),
            Input(self._get_id(self._LOAD_BUTTON_ID), "n_clicks"),
            State(
                self._get_id(self._LOAD_DROPDOWN_ID), "value"
            ),  # Get selected path string
            prevent_initial_call=True,
        )
        def load_annotations(n_clicks, selected_file_path_str):
            if not selected_file_path_str:
                logger.warning("Load annotations: No file selected.")
                raise PreventUpdate

            points_data, lines_data = load_annotation_file(selected_file_path_str)

            if points_data is not None and lines_data is not None:
                # Reset state and load new data
                new_points_state = self._get_default_points_state()
                new_lines_state = self._get_default_lines_state()
                new_points_state["added_points"] = points_data
                new_lines_state["added_lines"] = lines_data
                logger.info(
                    f"Loaded annotations from {Path(selected_file_path_str).name}"
                )
                return (
                    new_points_state,
                    new_lines_state,
                )  # , selected_file_path_str # Update dropdown value
            else:
                logger.error(
                    f"Failed to load annotations from {selected_file_path_str}"
                )
                raise PreventUpdate  # Keep existing state

        @app.callback(
            Output(self._get_id(self._SAVE_STATUS_ID), "children"),
            Input(self._get_id(self._SAVE_BUTTON_ID), "n_clicks"),
            State(self._get_id(self._POINTS_STORE_ID), "data"),
            State(self._get_id(self._LINES_STORE_ID), "data"),
            prevent_initial_call=True,
        )
        def save_annotations(n_clicks, points_state, lines_state):
            saved_path = save_annotation_file(
                self.save_load_path,
                points_state.get("added_points", {}),
                lines_state.get("added_lines", {}),
            )
            if saved_path:
                # Clear status after few seconds? Requires clientside callback or interval.
                return f"Saved: {saved_path.name}"
            else:
                return "Save failed."

    def _register_analysis_callback(self, app: Dash):
        """Callback for analysis (e.g., slope calculation)."""

        @app.callback(
            Output(self._get_id(self._ANALYSIS_RESULTS_ID), "children"),
            # Output(orchestrator_stores['matrix_update_store'], "data"), # Example output to orchestrator
            Input(self._get_id(self._CALCULATE_BUTTON_ID), "n_clicks"),
            State(self._get_id(self._POINTS_STORE_ID), "data"),
            State(self._get_id(self._LINES_STORE_ID), "data"),
            prevent_initial_call=True,
        )
        def run_analysis(n_clicks, points_state, lines_state):
            logger.info("Calculating slopes...")
            slopes = calculate_slopes(
                points_state.get("added_points", {}), lines_state.get("added_lines", {})
            )
            # Format results for display
            results_str = json.dumps(slopes, indent=2)
            logger.info(f"Slope calculation results: {results_str}")

            # Prepare data for potential matrix update (example)
            # matrix_update_data = {"slopes": slopes, "timestamp": datetime.now().isoformat()}

            return results_str  # , matrix_update_data

    def _register_shortcut_callback(self, app: Dash):
        """Client-side callback to handle keyboard shortcuts for mode switching."""
        # This assumes the orchestrator includes a div with id='main-key-listener'
        # in the main app layout to capture events globally.
        # We target the mode selector specific to this component instance.
        mode_selector_dash_id = self._get_id(self._MODE_SELECTOR_ID)

        app.clientside_callback(
            f"""
            function(dummy_input) {{
                const componentId = '{self.component_id}'; // Pass component ID to scope listener
                const listenerAttachedFlag = `annotationKeyListenerAttached_${{componentId}}`;
                const modeSelectorId = JSON.stringify({json.dumps(mode_selector_dash_id)});

                // Only attach listener once per component instance
                if (!window[listenerAttachedFlag]) {{
                     // console.log(`Attaching key listener for ${componentId}`);
                     const handleKeyDown = function(event) {{
                        // Check if the annotation tab/mode is active - requires knowing active tab state
                        // This logic might be better handled if the listener is attached/detached
                        // based on the analysis mode being active. For now, it's always listening.

                        const activeElement = document.activeElement;
                        const isInputFocused = activeElement && (activeElement.tagName === 'INPUT' || activeElement.tagName === 'TEXTAREA' || activeElement.isContentEditable);

                        if (event.shiftKey && !isInputFocused) {{ // Shift key + not input focused
                            let key = event.key.toLowerCase();
                            let mapping = {{"p": "point", "l": "line", "d": "delete"}};

                            if (mapping.hasOwnProperty(key)) {{
                                // Check if the annotation tab is actually visible/active
                                // This is tricky client-side without more state.
                                // Assume for now if the callback runs, we can try to set props.
                                event.preventDefault();
                                // console.log(`Annotation Shortcut for ${componentId}:`, mapping[key]);
                                try {{
                                    // Use the dynamically generated ID string
                                    dash_clientside.set_props(JSON.parse(modeSelectorId), {{value: mapping[key]}});
                                }} catch (e) {{
                                     // This might fail if the component isn't rendered
                                     // console.error(`Error setting mode selector props for ${componentId}:`, e);
                                }}
                            }}
                        }}
                    }};
                    document.addEventListener("keydown", handleKeyDown);
                    window[listenerAttachedFlag] = handleKeyDown; // Store the handler to potentially remove later
                }}
                return window.dash_clientside.no_update;
            }}
            """,
            Output(
                f"dummy-output-{self.component_id}", "data"
            ),  # Dummy output per instance
            Input(
                f"dummy-input-{self.component_id}", "data"
            ),  # Dummy input per instance
            # The orchestrator needs to add these dummy Divs to the layout:
            # html.Div(id=f'dummy-input-{self.component_id}', style={'display':'none'}),
            # html.Div(id=f'dummy-output-{self.component_id}', style={'display':'none'})
        )

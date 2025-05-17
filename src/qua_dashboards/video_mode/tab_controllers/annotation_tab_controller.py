import json
import logging
from typing import Dict, Any, List, Optional
import re

import dash
import dash_bootstrap_components as dbc
import numpy as np
import xarray as xr
from dash import Dash, Input, Output, State, html, ctx
from dash.exceptions import PreventUpdate

from qua_dashboards.video_mode.tab_controllers.base_tab_controller import (
    BaseTabController,
)
from qua_dashboards.video_mode import data_registry
from qua_dashboards.video_mode.utils.dash_utils import xarray_to_plotly
from qua_dashboards.video_mode.utils.annotation_utils import (
    calculate_slopes,
    find_closest_line_id,
)
from qua_dashboards.video_mode.utils.data_utils import load_data

logger = logging.getLogger(__name__)

__all__ = ["AnnotationTabController"]


class AnnotationTabController(BaseTabController):
    """
    Controls the 'Annotation & Analysis' tab in the Video Mode application.

    This tab allows users to import a static data frame (snapshot), annotate it
    with points and lines, save/load annotations, and perform basic analysis.
    It interacts with the SharedViewerComponent by updating the 'static_data'
    entry in the data_registry.
    """

    _TAB_LABEL = "Annotation & Analysis"
    _TAB_VALUE = "annotation-tab"

    # Control Suffixes (IDs for Dash components)
    _MODE_SELECTOR_SUFFIX = "mode-selector"
    _CLEAR_BUTTON_SUFFIX = "clear-button"
    _LOAD_DROPDOWN_SUFFIX = "load-dropdown"
    _LOAD_BUTTON_SUFFIX = "load-button"
    _SAVE_BUTTON_SUFFIX = "save-button"
    _SAVE_STATUS_SUFFIX = "save-status"
    _CALCULATE_BUTTON_SUFFIX = "calculate-button"
    _ANALYSIS_RESULTS_SUFFIX = "analysis-results"
    _IMPORT_LIVE_FRAME_BUTTON_SUFFIX = "import-live-frame-button"
    _LOAD_FROM_DISK_BUTTON_SUFFIX = "load-from-disk-button"
    _LOAD_FROM_DISK_INPUT_SUFFIX = "load-from-disk-input"
    _SHORTCUT_DUMMY_INPUT_SUFFIX = "shortcuts-dummy-input"
    _SHORTCUT_DUMMY_OUTPUT_SUFFIX = "shortcuts-dummy-output"

    def __init__(
        self,
        component_id: str = "annotation-tab-controller",
        point_select_tolerance: float = 0.01,  # Relative
        **kwargs: Any,
    ) -> None:
        """Initializes the AnnotationTabController.

        Args:
            component_id: A unique string identifier for this component instance.
            point_select_tolerance: Click tolerance (relative to figure diagonal)
                for selecting existing points/lines.
            **kwargs: Additional keyword arguments passed to BaseComponent.
        """
        super().__init__(component_id=component_id, **kwargs)
        self._relative_click_tolerance = max(0.001, point_select_tolerance)
        self._absolute_click_tolerance: float = 1.0  # Default, updated with figure

        # Transient UI state for interactions
        self._selected_point_to_move: Optional[Dict[str, Any]] = {
            "is_moving": False,
            "point_id": None,
        }
        self._selected_indices_for_line: List[str] = []
        self._next_point_id_counter: int = 0  # Reset when new base_image is loaded
        self._next_line_id_counter: int = 0  # Reset when new base_image is loaded

        logger.info(f"AnnotationTabController '{self.component_id}' initialized.")

    def _get_default_static_data_object(
        self, base_image: Optional[xr.DataArray] = None
    ) -> Dict[str, Any]:
        """Creates a default structure for the 'static_data' registry object."""
        return {
            "base_image_data": base_image,
            "annotations": {"points": [], "lines": []},
        }

    def _extract_base_image_from_live_data(self) -> Optional[xr.DataArray]:
        """Extracts the base image from the current live data in the registry

        Returns:
            xr.DataArray: The base image data, or None if not found.
        """
        live_data = data_registry.get_data(data_registry.LIVE_DATA_KEY)
        if not isinstance(live_data, dict):
            logger.warning(
                f"{self.component_id}: Live data is not a dictionary. Using empty "
                "static data object."
            )
            return None
        base_image = live_data.get("base_image_data")
        if not isinstance(base_image, xr.DataArray):
            logger.warning(
                f"{self.component_id}: Live data does not contain a valid base image. "
                "Using empty static data object."
            )
            return None
        return base_image

    def _get_new_point_id(self) -> str:
        """Generates a unique ID for a new point."""
        # Simple counter for this session, reset if base image changes.
        # Or use UUIDs for global uniqueness if points might be merged across sessions.
        pid = f"p_{self._next_point_id_counter}"
        self._next_point_id_counter += 1
        return pid

    def _get_new_line_id(self) -> str:
        """Generates a unique ID for a new line."""
        lid = f"l_{self._next_line_id_counter}"
        self._next_line_id_counter += 1
        return lid

    def _reset_transient_state(self):
        """Resets transient UI state, typically when a new image is loaded."""
        self._selected_point_to_move = {"is_moving": False, "point_id": None}
        self._selected_indices_for_line = []
        self._next_point_id_counter = 0
        self._next_line_id_counter = 0
        logger.debug(f"{self.component_id}: Transient annotation state reset.")

    def get_layout(self) -> html.Div:
        """Generates the Dash layout for the Annotation tab."""
        logger.debug(f"Generating layout for {self.component_id}")
        radio_options = [
            {"label": "Add/Move Points (Shift+P)", "value": "point"},
            {"label": "Add Lines (Shift+L)", "value": "line"},
            {"label": "Delete (Shift+D)", "value": "delete"},
        ]

        controls_layout = dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button(
                                "Import Live Frame",
                                id=self._get_id(self._IMPORT_LIVE_FRAME_BUTTON_SUFFIX),
                                color="primary",
                                className="mb-3 w-100",
                            ),
                            width=6,
                        ),
                        dbc.Col(
                            dbc.InputGroup(
                                [
                                    dbc.Button(
                                        "Load from Disk",
                                        id=self._get_id(
                                            self._LOAD_FROM_DISK_BUTTON_SUFFIX
                                        ),
                                        color="secondary",
                                    ),
                                    dbc.Input(
                                        id=self._get_id(
                                            self._LOAD_FROM_DISK_INPUT_SUFFIX
                                        ),
                                        placeholder="idx",
                                        type="text",
                                    ),
                                ],
                                className="mb-3",
                            ),
                            width=6,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                html.Hr(),
                dbc.RadioItems(
                    id=self._get_id(self._MODE_SELECTOR_SUFFIX),
                    options=radio_options,
                    value="point",
                    className="mb-3",
                ),
                dbc.Button(
                    "Clear Annotations",
                    id=self._get_id(self._CLEAR_BUTTON_SUFFIX),
                    color="warning",
                    size="sm",
                    className="mb-3 me-2",
                ),
                html.Hr(),
                html.H6("Analysis"),
                dbc.Button(
                    "Calculate Slopes",
                    id=self._get_id(self._CALCULATE_BUTTON_SUFFIX),
                    color="info",
                    size="sm",
                    className="mb-2",
                ),
                html.Pre(
                    id=self._get_id(self._ANALYSIS_RESULTS_SUFFIX),
                    className="border rounded analysis-results-dark p-1 mb-3",
                    style={
                        "maxHeight": "100px",
                        "overflowY": "auto",
                        "fontSize": "0.8em",
                    },
                ),
            ]
        )

        other_components = [
            html.Div(
                id=self._get_id(self._SHORTCUT_DUMMY_INPUT_SUFFIX),
                style={"display": "none"},
            ),
            html.Div(
                id=self._get_id(self._SHORTCUT_DUMMY_OUTPUT_SUFFIX),
                style={"display": "none"},
            ),
        ]

        return html.Div(
            [
                dbc.Card(
                    controls_layout,
                    color="dark",
                    inverse=True,
                    className="h-100 tab-card-dark",
                ),
                *other_components,
            ]
        )

    def on_tab_activated(self) -> Dict[str, Any]:
        """Handles logic when the annotation tab becomes active."""
        from qua_dashboards.video_mode.video_mode_component import VideoModeComponent

        logger.info(f"{self.component_id} activated.")

        # Ensure a default static_data object exists in the registry
        static_data_obj = data_registry.get_data(data_registry.STATIC_DATA_KEY)
        current_version = data_registry.get_current_version(
            data_registry.STATIC_DATA_KEY
        )

        if static_data_obj is None or current_version is None:
            logger.info(
                f"{self.component_id}: No static data found in registry. "
                "Initializing with default or latest live data."
            )
            base_image = self._extract_base_image_from_live_data()
            default_obj = self._get_default_static_data_object(base_image=base_image)
            current_version = data_registry.set_data(
                data_registry.STATIC_DATA_KEY, default_obj
            )
            self._reset_transient_state()  # Reset IDs if creating new
            # Update click tolerance if there's a base image in the default
            if default_obj.get("base_image_data") is not None:
                fig_dict = xarray_to_plotly(default_obj["base_image_data"]).to_dict()
                self._update_click_tolerance(fig_dict=fig_dict)
            static_data_obj = (
                default_obj  # Ensure static_data_obj is set for counter logic
            )

        # Ensure point/line counters are set to the next available unique index
        def get_next_index(items, prefix):
            max_idx = -1
            pattern = re.compile(rf"{prefix}_(\d+)")
            for item in items:
                match = pattern.fullmatch(str(item.get("id", "")))
                if not match:
                    continue

                idx = int(match.group(1))
                max_idx = max(max_idx, idx)
            return max_idx + 1

        annotations = static_data_obj.get("annotations", {}) if static_data_obj else {}
        self._next_point_id_counter = get_next_index(annotations.get("points", []), "p")
        self._next_line_id_counter = get_next_index(annotations.get("lines", []), "l")

        viewer_data_store_payload = {
            "key": data_registry.STATIC_DATA_KEY,
            "version": current_version,
        }
        layout_config_payload = {"clickmode": "event+select"}

        return {
            VideoModeComponent.VIEWER_DATA_STORE_SUFFIX: viewer_data_store_payload,
            VideoModeComponent.VIEWER_LAYOUT_CONFIG_STORE_SUFFIX: layout_config_payload,
        }

    def on_tab_deactivated(self) -> None:
        """Handles logic when the annotation tab becomes inactive."""
        logger.info(f"{self.component_id} deactivated.")
        self._reset_transient_state()  # Clear selections when tab is left

    def _update_click_tolerance(
        self,
        base_image_data: Optional[xr.DataArray] = None,
        fig_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Updates the absolute click tolerance based on figure dimensions."""
        if base_image_data is None and fig_dict is None:
            logger.warning(
                f"{self.component_id}: No base image or figure dict provided. "
                "Using default tolerance."
            )
            self._absolute_click_tolerance = 0.01  # Default small absolute
            return

        if isinstance(base_image_data, xr.DataArray):
            fig_dict = xarray_to_plotly(base_image_data).to_dict()

        if fig_dict is None:
            logger.warning(
                f"{self.component_id}: No figure dict provided. "
                "Using default tolerance."
            )
            self._absolute_click_tolerance = 0.01  # Default small absolute
            return
        try:
            # Convert to figure dict to get ranges (assuming xarray_to_plotly works)
            layout = fig_dict.get("layout", {})

            x_axis_key = next(
                (key for key in layout if key.startswith("xaxis")), "xaxis"
            )
            y_axis_key = next(
                (key for key in layout if key.startswith("yaxis")), "yaxis"
            )

            x_range = layout.get(x_axis_key, {}).get("range")
            y_range = layout.get(y_axis_key, {}).get("range")

            if (
                x_range
                and y_range
                and isinstance(x_range, (list, tuple))
                and len(x_range) == 2
                and isinstance(y_range, (list, tuple))
                and len(y_range) == 2
                and all(isinstance(v, (int, float)) for v in x_range + y_range)
            ):
                diag = np.sqrt(
                    (x_range[1] - x_range[0]) ** 2 + (y_range[1] - y_range[0]) ** 2
                )
                if diag > 1e-9:  # Avoid division by zero or tiny diagonals
                    self._absolute_click_tolerance = (
                        diag * self._relative_click_tolerance
                    )
                else:
                    self._absolute_click_tolerance = (
                        0.01  # Fallback for zero-size image
                    )
                logger.debug(
                    f"{self.component_id}: Absolute click tolerance set to: "
                    f"{self._absolute_click_tolerance:.4g}"
                )
            else:
                logger.warning(
                    f"{self.component_id}: Could not determine figure range for "
                    f"tolerance from base_image_data. Using default. "
                    f"x_range: {x_range}, y_range: {y_range}"
                )
                self._absolute_click_tolerance = 0.01
        except Exception as e:
            logger.warning(
                f"{self.component_id}: Error calculating tolerance from base_image_data: {e}"
            )
            self._absolute_click_tolerance = 0.01

    def register_callbacks(
        self,
        app: Dash,
        orchestrator_stores: Dict[str, Any],
        shared_viewer_store_ids: Dict[str, Any],
        shared_viewer_graph_id: Dict[str, str],
    ) -> None:
        """Registers all callbacks for the AnnotationTabController."""
        from qua_dashboards.video_mode.video_mode_component import VideoModeComponent

        logger.info(f"Registering callbacks for {self.component_id}")

        latest_processed_data_store_id = orchestrator_stores[
            VideoModeComponent.LATEST_PROCESSED_DATA_STORE_SUFFIX
        ]
        viewer_data_store_id = orchestrator_stores[
            VideoModeComponent.VIEWER_DATA_STORE_SUFFIX
        ]
        self._register_import_live_frame_callback(
            app, latest_processed_data_store_id, viewer_data_store_id
        )
        self._register_graph_interaction_callback(
            app, shared_viewer_graph_id, viewer_data_store_id
        )
        self._register_clear_annotations_callback(app, viewer_data_store_id)
        self._register_load_from_disk_callback(
            app, viewer_data_store_id, orchestrator_stores
        )
        self._register_analysis_callback(app, viewer_data_store_id)
        self._register_shortcut_callback(app)

    def _register_import_live_frame_callback(
        self,
        app: Dash,
        latest_processed_data_store_id: Dict[str, str],
        viewer_data_store_output_id: Dict[str, str],
    ) -> None:
        """Callback to import the current live frame as a static snapshot."""

        @app.callback(
            Output(viewer_data_store_output_id, "data"),
            Input(self._get_id(self._IMPORT_LIVE_FRAME_BUTTON_SUFFIX), "n_clicks"),
            State(latest_processed_data_store_id, "data"),
            prevent_initial_call=True,
        )
        def _import_live_frame(
            n_clicks: int, live_data_ref: Optional[Dict[str, Any]]
        ) -> Dict[str, Any]:
            if live_data_ref is None:
                logger.warning("Import Live Frame: No live data reference found.")
                raise PreventUpdate
            data_key = live_data_ref.get("key")
            if not data_key:
                logger.warning("Import Live Frame: Live data reference missing key.")
                raise PreventUpdate

            base_image = self._extract_base_image_from_live_data()
            if base_image is None:
                raise PreventUpdate
            new_static_object = self._get_default_static_data_object(
                base_image=base_image
            )
            new_version = data_registry.set_data(
                data_registry.STATIC_DATA_KEY, new_static_object
            )
            self._reset_transient_state()
            self._update_click_tolerance(base_image_data=base_image)

            logger.info(
                f"{self.component_id}: Imported live frame. New static data version: {new_version}"
            )
            return {"key": data_registry.STATIC_DATA_KEY, "version": new_version}

    def _handle_point_mode_interaction(
        self,
        x: float,
        y: float,
        clicked_annotation_point_id: Optional[str],  # Now string ID
        current_annotations: Dict[str, List[Dict[str, Any]]],
    ) -> bool:
        """Handles graph interactions for 'point' mode. Modifies current_annotations. Returns True if changed."""
        points_list = current_annotations["points"]
        changed = False

        if (
            self._selected_point_to_move.get("is_moving")
            and self._selected_point_to_move.get("point_id") is not None
        ):
            point_id_to_move = self._selected_point_to_move["point_id"]
            for point in points_list:
                if point["id"] == point_id_to_move:
                    if point["x"] != x or point["y"] != y:
                        point["x"], point["y"] = x, y
                        changed = True
                    break
            self._selected_point_to_move = {"is_moving": False, "point_id": None}
        elif clicked_annotation_point_id is not None:
            self._selected_point_to_move = {
                "is_moving": True,
                "point_id": clicked_annotation_point_id,
            }
            # No change to data yet, just UI state
        else:  # Click on background, add new point
            new_point_id = self._get_new_point_id()
            points_list.append({"id": new_point_id, "x": x, "y": y})
            changed = True
        return changed

    def _handle_line_mode_interaction(
        self,
        clicked_annotation_point_id: Optional[str],  # Now string ID
        current_annotations: Dict[str, List[Dict[str, Any]]],
    ) -> bool:
        """Handles graph interactions for 'line' mode. Modifies current_annotations. Returns True if changed."""
        if clicked_annotation_point_id is None:
            return False

        lines_list = current_annotations["lines"]
        points_list = current_annotations["points"]
        changed = False

        # Ensure the clicked point actually exists
        if not any(p["id"] == clicked_annotation_point_id for p in points_list):
            logger.warning(
                f"Line mode: Clicked point ID {clicked_annotation_point_id} not found."
            )
            return False

        if clicked_annotation_point_id not in self._selected_indices_for_line:
            self._selected_indices_for_line.append(clicked_annotation_point_id)
        else:
            self._selected_indices_for_line.remove(clicked_annotation_point_id)

        if len(self._selected_indices_for_line) == 2:
            id1, id2 = self._selected_indices_for_line
            is_dup = any(
                (line["start_point_id"] == id1 and line["end_point_id"] == id2)
                or (line["start_point_id"] == id2 and line["end_point_id"] == id1)
                for line in lines_list
            )
            if not is_dup and id1 != id2:
                new_line_id = self._get_new_line_id()
                lines_list.append(
                    {
                        "id": new_line_id,
                        "start_point_id": id1,
                        "end_point_id": id2,
                    }
                )
                changed = True
            self._selected_indices_for_line.clear()
        return changed

    def _handle_delete_mode_interaction(
        self,
        x: float,
        y: float,
        clicked_annotation_point_id: Optional[str],  # Now string ID
        current_annotations: Dict[str, List[Dict[str, Any]]],
    ) -> bool:
        """Handles graph interactions for 'delete' mode. Modifies current_annotations. Returns True if changed."""
        points_list = current_annotations["points"]
        lines_list = current_annotations["lines"]
        changed = False

        if clicked_annotation_point_id is not None:
            original_len = len(points_list)
            current_annotations["points"] = [
                p for p in points_list if p["id"] != clicked_annotation_point_id
            ]
            if len(current_annotations["points"]) < original_len:
                changed = True
                # Also remove lines connected to this point
                current_annotations["lines"] = [
                    line
                    for line in lines_list
                    if line["start_point_id"] != clicked_annotation_point_id
                    and line["end_point_id"] != clicked_annotation_point_id
                ]
        else:
            line_to_delete_id = find_closest_line_id(
                x, y, current_annotations, self._absolute_click_tolerance
            )
            if line_to_delete_id is not None:
                original_len = len(lines_list)
                current_annotations["lines"] = [
                    line for line in lines_list if line["id"] != line_to_delete_id
                ]
                if len(current_annotations["lines"]) < original_len:
                    changed = True
        return changed

    def _handle_hover_interaction(
        self,
        hover_data: Dict[str, Any],
        mode: str,
        current_annotations: Dict[str, List[Dict[str, Any]]],
    ) -> bool:
        """Handles graph hover for 'point' move. Modifies current_annotations. Returns True if changed."""
        if not (
            mode == "point"
            and self._selected_point_to_move.get("is_moving")
            and self._selected_point_to_move.get("point_id") is not None
        ):
            return False

        points_list = current_annotations["points"]
        point_id_to_move = self._selected_point_to_move["point_id"]
        changed = False

        hover_point_info = hover_data["points"][0]
        is_hover_on_annotation = hover_point_info.get(
            "curveNumber", 0
        ) > 0 and hover_point_info.get("name", "").startswith("annotations_")

        if not is_hover_on_annotation:  # Only move if hovering over the base heatmap
            x_hover, y_hover = hover_point_info["x"], hover_point_info["y"]
            for point in points_list:
                if point["id"] == point_id_to_move:
                    if point["x"] != x_hover or point["y"] != y_hover:
                        point["x"], point["y"] = x_hover, y_hover
                        changed = True
                    break
            else:  # Point to move not found (e.g., deleted mid-drag)
                self._selected_point_to_move = {"is_moving": False, "point_id": None}
        return changed

    def _register_graph_interaction_callback(
        self,
        app: Dash,
        shared_viewer_graph_id: Dict[str, str],
        viewer_data_store_id: Dict[str, str],
    ) -> None:
        """Callback to handle user interactions with the graph (clicks, hovers)."""

        @app.callback(
            Output(viewer_data_store_id, "data", allow_duplicate=True),
            Input(shared_viewer_graph_id, "clickData"),
            Input(shared_viewer_graph_id, "hoverData"),
            State(self._get_id(self._MODE_SELECTOR_SUFFIX), "value"),
            State(viewer_data_store_id, "data"),  # Get current key and version
            prevent_initial_call=True,
        )
        def _handle_graph_interactions(
            click_data: Optional[Dict[str, Any]],
            hover_data: Optional[Dict[str, Any]],
            mode: str,
            current_viewer_data_ref: Optional[Dict[str, str]],
        ) -> Dict[str, Any]:
            if not self.is_active:
                raise PreventUpdate
            elif (
                not current_viewer_data_ref
                or current_viewer_data_ref.get("key") != data_registry.STATIC_DATA_KEY
            ):
                raise PreventUpdate

            logger.debug("-- AnnotationTabController._handle_graph_interactions --")
            logger.debug(f"  click_data: {click_data}")
            logger.debug(f"  hover_data: {hover_data}")
            logger.debug(f"  mode: {mode}")
            logger.debug(f"  current_viewer_data_ref: {current_viewer_data_ref}")

            static_data_object = data_registry.get_data(data_registry.STATIC_DATA_KEY)
            if not static_data_object or not isinstance(
                static_data_object.get("annotations"), dict
            ):
                logger.error(
                    f"{self.component_id}: Static data object or annotations missing/invalid."
                )
                raise PreventUpdate

            # Deep copy of annotations for modification
            annotations_copy = json.loads(json.dumps(static_data_object["annotations"]))
            interaction_changed_data = False

            graph_id_json_str = json.dumps(shared_viewer_graph_id, sort_keys=True)
            interaction_type = list(ctx.triggered_prop_ids)[0].split(".")[-1]
            logger.debug(f"triggered_prop_id: {interaction_type}")

            if (
                interaction_type == "clickData"
                and click_data
                and click_data.get("points")
            ):
                point_info = click_data["points"][0]
                x, y = point_info["x"], point_info["y"]
                logger.debug(f"  clickData point_info: {point_info}")
                curve_name = point_info.get("name", "")
                logger.debug(f"  curve_name: {curve_name}")
                custom_data_val = point_info.get("customdata")
                logger.debug(f"  custom_data_val: {custom_data_val}")
                clicked_annotation_point_id = None
                if isinstance(custom_data_val, list) and len(custom_data_val) > 0:
                    clicked_annotation_point_id = str(custom_data_val[0])
                elif isinstance(
                    custom_data_val, (str, int, float)
                ):  # If it's a direct ID
                    clicked_annotation_point_id = str(custom_data_val)

                if mode == "point":
                    interaction_changed_data = self._handle_point_mode_interaction(
                        x, y, clicked_annotation_point_id, annotations_copy
                    )
                elif mode == "line":
                    interaction_changed_data = self._handle_line_mode_interaction(
                        clicked_annotation_point_id, annotations_copy
                    )
                elif mode == "delete":
                    interaction_changed_data = self._handle_delete_mode_interaction(
                        x, y, clicked_annotation_point_id, annotations_copy
                    )
            elif (
                interaction_type == "hoverData"
                and hover_data
                and hover_data.get("points")
            ):
                interaction_changed_data = self._handle_hover_interaction(
                    hover_data, mode, annotations_copy
                )

            if interaction_changed_data:
                new_static_data_object = {
                    "base_image_data": static_data_object["base_image_data"],
                    "annotations": annotations_copy,
                }
                new_version = data_registry.set_data(
                    data_registry.STATIC_DATA_KEY, new_static_data_object
                )
                logger.debug(
                    f"{self.component_id}: Annotations updated. New version: {new_version}"
                )
                return {"key": data_registry.STATIC_DATA_KEY, "version": new_version}

            logger.debug("No changes to annotations, raising PreventUpdate")
            raise PreventUpdate

    def _register_clear_annotations_callback(
        self, app: Dash, viewer_data_store_id: Dict[str, str]
    ) -> None:
        """Callback to clear all current annotations."""

        @app.callback(
            Output(viewer_data_store_id, "data", allow_duplicate=True),
            Input(self._get_id(self._CLEAR_BUTTON_SUFFIX), "n_clicks"),
            State(viewer_data_store_id, "data"),
            prevent_initial_call=True,
        )
        def _clear_annotations(
            _n_clicks: int, current_viewer_data_ref: Optional[Dict[str, str]]
        ) -> Dict[str, Any]:
            if (
                not current_viewer_data_ref
                or current_viewer_data_ref.get("key") != data_registry.STATIC_DATA_KEY
            ):
                logger.warning(
                    f"{self.component_id}: Clear annotations called "
                    "but not in static data mode."
                )
                raise PreventUpdate

            static_data_object = data_registry.get_data(data_registry.STATIC_DATA_KEY)
            if not static_data_object:
                base_image = None  # Or fetch a default if necessary
            else:
                base_image = static_data_object.get("base_image_data")

            cleared_static_object = self._get_default_static_data_object(
                base_image=base_image
            )
            new_version = data_registry.set_data(
                data_registry.STATIC_DATA_KEY, cleared_static_object
            )
            self._reset_transient_state()  # Also reset point/line ID counters

            logger.info(
                f"{self.component_id}: Cleared annotations. New version: {new_version}"
            )
            return {"key": data_registry.STATIC_DATA_KEY, "version": new_version}

    def _register_analysis_callback(
        self, app: Dash, viewer_data_store_id: Dict[str, str]
    ) -> None:
        """Callback for performing analysis (e.g., calculating slopes)."""

        @app.callback(
            Output(self._get_id(self._ANALYSIS_RESULTS_SUFFIX), "children"),
            Input(self._get_id(self._CALCULATE_BUTTON_SUFFIX), "n_clicks"),
            State(viewer_data_store_id, "data"),
            prevent_initial_call=True,
        )
        def _run_slope_analysis(
            _n_clicks: int, current_viewer_data_ref: Optional[Dict[str, str]]
        ) -> str:
            if (
                not current_viewer_data_ref
                or current_viewer_data_ref.get("key") != data_registry.STATIC_DATA_KEY
            ):
                return "Analysis requires static data to be active."

            static_data_object = data_registry.get_data(data_registry.STATIC_DATA_KEY)
            if not static_data_object or not static_data_object.get("annotations"):
                return "No annotations found in static data for analysis."

            annotations_data = static_data_object["annotations"]
            slopes = calculate_slopes(annotations_data)
            if slopes:
                return json.dumps(slopes, indent=2)
            return "No lines to analyze or error in calculation."

    def _register_shortcut_callback(self, app: Dash) -> None:
        """Client-side callback for handling keyboard shortcuts."""
        app.clientside_callback(
            f"""
            function(dummy_input_value) {{
                const componentId = '{self.component_id}';
                const listenerAttachedFlag = `annotationKeyListenerAttached_${{componentId}}`;
                const modeSelectorIdObj = {json.dumps(self._get_id(self._MODE_SELECTOR_SUFFIX))};

                if (!window[listenerAttachedFlag]) {{
                     const handleKeyDown = function(event) {{
                        const activeElement = document.activeElement;
                        const isInputFocused = activeElement &&
                            (activeElement.tagName === 'INPUT' ||
                             activeElement.tagName === 'TEXTAREA' ||
                             activeElement.isContentEditable);

                        let modeSelectorElement = document.getElementById(JSON.stringify(modeSelectorIdObj));
                        if (!modeSelectorElement) {{ return; }}
                        
                        const isTabActive = modeSelectorElement.offsetParent !== null;

                        if (event.shiftKey && !isInputFocused && isTabActive) {{
                            let key = event.key.toLowerCase();
                            let mapping = {{"p": "point", "l": "line", "d": "delete"}};
                            if (mapping.hasOwnProperty(key)) {{
                                event.preventDefault();
                                try {{
                                    dash_clientside.set_props(modeSelectorIdObj, {{value: mapping[key]}});
                                }} catch (e) {{
                                     console.error(`Error setting mode selector for ${{componentId}}:`, e);
                                }}
                            }}
                        }}
                    }};
                    document.addEventListener("keydown", handleKeyDown);
                    window[listenerAttachedFlag] = true;
                }}
                return window.dash_clientside.no_update;
            }}
            """,
            Output(
                component_id=self._get_id(self._SHORTCUT_DUMMY_OUTPUT_SUFFIX),
                component_property="children",
            ),
            Input(
                component_id=self._get_id(self._SHORTCUT_DUMMY_INPUT_SUFFIX),
                component_property="children",
            ),
        )

    def _register_load_from_disk_callback(
        self,
        app: Dash,
        viewer_data_store_id: Dict[str, str],
        orchestrator_stores: Dict[str, Any],
    ) -> None:
        from qua_dashboards.video_mode.video_mode_component import VideoModeComponent

        main_status_alert_id = orchestrator_stores.get(
            VideoModeComponent._MAIN_STATUS_ALERT_ID_SUFFIX
        )

        @app.callback(
            Output(viewer_data_store_id, "data", allow_duplicate=True),
            Output(main_status_alert_id, "children", allow_duplicate=True),
            Input(self._get_id(self._LOAD_FROM_DISK_BUTTON_SUFFIX), "n_clicks"),
            State(self._get_id(self._LOAD_FROM_DISK_INPUT_SUFFIX), "value"),
            State(viewer_data_store_id, "data"),
            prevent_initial_call=True,
        )
        def load_from_disk_callback(n_clicks, idx, current_viewer_data_ref):
            if not n_clicks:
                raise PreventUpdate
            if not idx:
                return (
                    dash.no_update,
                    dbc.Alert(
                        "Please provide an input idx to load data.",
                        color="warning",
                        dismissable=True,
                    ),
                )
            data = load_data(idx)
            if not isinstance(data, dict):
                raise PreventUpdate
            new_version = data_registry.set_data(data_registry.STATIC_DATA_KEY, data)
            self._reset_transient_state()
            if data.get("base_image_data") is not None:
                self._update_click_tolerance(base_image_data=data["base_image_data"])
            self._next_point_id_counter = len(
                data.get("annotations", {}).get("points", [])
            )
            self._next_line_id_counter = len(
                data.get("annotations", {}).get("lines", [])
            )
            return (
                {"key": data_registry.STATIC_DATA_KEY, "version": new_version},
                dbc.Alert(
                    f"Successfully loaded data for idx '{idx}'.",
                    color="success",
                    dismissable=True,
                ),
            )

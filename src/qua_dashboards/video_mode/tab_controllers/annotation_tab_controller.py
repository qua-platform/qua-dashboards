import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import re

import dash
import dash_bootstrap_components as dbc
import numpy as np
import xarray as xr
from dash import Dash, Input, Output, State, html, ctx, dcc, no_update, ALL
from dash.exceptions import PreventUpdate
from plotly import graph_objects as go

from qua_dashboards.utils import get_axis_limits
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
from qua_dashboards.video_mode.data_acquirers.base_data_acquirer import BaseDataAcquirer

logger = logging.getLogger(__name__)

__all__ = ["AnnotationTabController"]


# Ensure point/line counters are set to the next available unique index
def get_next_index(items: List[str], prefix: str) -> int:
    """Get the next available unique index for a given prefix.

    example:
    items = ["p_0", "p_1", "l_0", "l_1"]
    prefix = "p"
    get_next_index(items, prefix)
    >>> 2
    """
    max_idx = -1
    pattern = re.compile(rf"{prefix}_(\d+)")
    for item in items:
        match = pattern.fullmatch(str(item.get("id", "")))
        if not match:
            continue

        idx = int(match.group(1))
        max_idx = max(max_idx, idx)
    return max_idx + 1


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
    _SHOW_LABELS_CHECKLIST_SUFFIX = "show-labels-checklist"
    _LINE_PROFILE_BUTTON_SUFFIX = "line-profile-button"

    def __init__(
        self,
        component_id: str = "annotation-tab-controller",
        point_select_tolerance: float = 0.025,  # Relative
        data_acquirer: BaseDataAcquirer = None,
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
        self._translate_all: Optional[Dict[str, Any]] = {
            "translate": False,
            "clicked_point": None,
        }
        self._show_labels = ['points']
        self._data_acquirer = data_acquirer
        self._axis_modes = {"x_mode": None, "y_mode": None}

        logger.info(f"AnnotationTabController '{self.component_id}' initialized.")

    @property
    def _get_axis_modes(self) -> Dict[str, str]: 
        if not self._data_acquirer: 
            return {"x_mode": None, "y_mode": None}
        
        return {
            "x_mode": (self._data_acquirer.x_mode or "").lower(),
            "y_mode": (self._data_acquirer.y_mode or "").lower(),
        }
    
    def _get_current_params(self, kind: str, axis: str) -> Optional[float]: 
        if not self._data_acquirer:
            return None
        try:
            if axis == "x":
                axis_obj = self._data_acquirer.x_axis
            elif axis == "y":
                axis_obj = self._data_acquirer.y_axis
            else:
                return None
            
            if kind == "frequency":
                if hasattr(axis_obj, 'offset_parameter') and axis_obj.offset_parameter:
                    pulse = axis_obj.offset_parameter
                    if hasattr(pulse, 'channel') and hasattr(pulse.channel, 'intermediate_frequency'):
                        return float(pulse.channel.intermediate_frequency)
            
            elif kind == "amplitude":
                if hasattr(axis_obj, 'offset_parameter') and axis_obj.offset_parameter:
                    pulse = axis_obj.offset_parameter
                    if hasattr(pulse, 'amplitude'):
                        return float(pulse.amplitude)
            
            return None
        
        except Exception as e:
            logger.warning(f"Error getting current param for {kind}/{axis}: {e}")
            return None

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
    
    def _apply_dropdown_options(self): 
        opts = []
        axis_modes = self._get_axis_modes
        xm, ym = axis_modes.get("x_mode"), axis_modes.get("y_mode")
        if xm == "voltage" or ym == "voltage":
            opts.append({"label": "Save points to VirtualGateSet", "value": "save_points_to_vgs"})
        if xm == "frequency" or ym == "frequency":
            opts.append({"label": "Frequency", "value": "frequency"})
        if xm == "amplitude" or ym == "amplitude":
            opts.append({"label": "Amplitude", "value": "amplitude"})
        return opts


    def _reset_transient_state(self):
        """Resets transient UI state, typically when a new image is loaded."""
        self._selected_point_to_move = {"is_moving": False, "point_id": None}
        self._selected_indices_for_line = []
        self._next_point_id_counter = 0
        self._next_line_id_counter = 0
        self._translate_all = {"translate": False, "clicked_point_x": None, "clicked_point_y": None}
        logger.debug(f"{self.component_id}: Transient annotation state reset.")

    def get_layout(self) -> html.Div:
        """Generates the Dash layout for the Annotation tab."""
        logger.debug(f"Generating layout for {self.component_id}")
        checklist_options = [
            {"label": "point labels", "value": "points"},  # For point labels
            #{"label": "line labels", "value": "lines"},   # If we also want to introduce line labels            
        ]
        radio_options = [
            {"label": "Add/Move Points", "value": "point"},
            {"label": "Add Lines", "value": "line"},
            {"label": "Delete", "value": "delete"},
            {"label": "Translate all points and lines", "value": "translate-all"},
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
                html.H6("Show labels"),
                dbc.Checklist(
                    id=self._get_id(self._SHOW_LABELS_CHECKLIST_SUFFIX),
                    options = checklist_options,
                    value = ["points"],
                    className="mb-3 me-2",
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
                dbc.Row([
                    dbc.Col(
                        dcc.Dropdown(
                            id=self._get_id("analysis-apply-action"),
                            options=[],
                            placeholder="Select action…",
                            clearable=True,
                            style={"color": "black"},
                        ),
                        width=8,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Apply to QUAM state",
                            id=self._get_id("apply-analysis-button"),
                            color="primary",
                            className="w-100",
                        ),
                        width=4,
                    ),
                ], className="mb-2"),
                html.Div(
                    id=self._get_id("analysis-param-editor"),
                    className="mt-2",
                ),
                html.Div(id=self._get_id("apply-analysis-status"), className="text-muted", style={"fontSize":"0.8em"}),
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
                html.Hr(), 
                html.H6("Select Line to Plot"), 
                dbc.Col(
                    dcc.Dropdown(
                        id = f"{self.component_id}-line-selector", 
                        options = [],  
                        value = None, 
                        placeholder = "Select line", 
                        clearable = True, 
                        style = {"color": "black"}
                    ), 
                width = 12),
                dbc.Col(
                    dcc.Checklist(
                        id = self._get_id("interpolation_toggle"), 
                        options = [{"label":"Toggle Interpolation", "value":"on"}], 
                        value = [], 
                        inline = True, 
                        inputStyle={"margin":"0 5px 0 0"},
                    )
                ),
                dbc.Row([
                    dbc.Col(
                        dbc.Button(
                            "Plot Line Profile",
                            id=self._get_id(self._LINE_PROFILE_BUTTON_SUFFIX),
                            color="success",
                            className="mt-2 w-100",
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Show Original Plot",
                            id=self._get_id("reset-2d-button"),
                            color="secondary",
                            className="mt-2 w-100",
                        ),
                        width=6,
                    ),
                ]),
            ],
        )
        other_components = []

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
    
    def _get_active_point(self, annotations: dict):
        pts = (annotations or {}).get("points", []) or []
        sel_id = None
        if isinstance(self._selected_point_to_move, dict):
            sel_id = self._selected_point_to_move.get("point_id")
        if sel_id:
            for p in pts:
                if p.get("id") == sel_id:
                    return p
        return pts[-1] if pts else None

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
            self._reset_transient_state() # Reset IDs if creating new
            # Update click tolerance if there's a base image in the default
            bi = default_obj.get("base_image_data")
            if isinstance(bi, xr.DataArray) and bi.ndim == 2 and "readout" not in bi.dims:
                fig = xarray_to_plotly(bi)
                self._update_click_tolerance(fig=fig)

            static_data_obj = default_obj

        annotations = static_data_obj.get("annotations", {}) if static_data_obj else {}
        self._next_point_id_counter = get_next_index(annotations.get("points", []), "p")
        self._next_line_id_counter = get_next_index(annotations.get("lines", []), "l")

        viewer_data_store_payload = {
            "key": data_registry.STATIC_DATA_KEY,
            "version": current_version,
        }
        viewer_ui_state_store_payload = {
            "selected_point_to_move": self._selected_point_to_move["point_id"],
            "selected_point_for_line": self._selected_indices_for_line,
            "show_labels": self._show_labels,
        }
        layout_config_payload = {"clickmode": "event+select"}

        self._axis_modes = self._get_axis_modes

        return {
            VideoModeComponent.VIEWER_DATA_STORE_SUFFIX: viewer_data_store_payload,
            VideoModeComponent.VIEWER_UI_STATE_STORE_SUFFIX: viewer_ui_state_store_payload,
            VideoModeComponent.VIEWER_LAYOUT_CONFIG_STORE_SUFFIX: layout_config_payload,
        }

    def on_tab_deactivated(self) -> None:
        """Handles logic when the annotation tab becomes inactive."""
        logger.info(f"{self.component_id} deactivated.")
        self._reset_transient_state()  # Clear selections when tab is left

    def _is_1d_profile_mode(self) -> bool:
        static = data_registry.get_data(data_registry.STATIC_DATA_KEY) or {}
        return "profile_plot" in static

    def _is_subplot_mode(self) -> bool:
        """Check if current data has subplots (readout dimension)"""
        static_data = data_registry.get_data(data_registry.STATIC_DATA_KEY)
        if not static_data:
            return False
        base_image = static_data.get("base_image_data")
        return (isinstance(base_image, xr.DataArray) and 
                base_image.ndim == 3 and 
                "readout" in base_image.dims)
    def _update_click_tolerance(
        self,
        base_image_data: Optional[xr.DataArray] = None,
        fig: Optional[go.Figure] = None,
    ) -> None:
        """Updates the absolute click tolerance based on figure dimensions."""
        if base_image_data is None and fig is None:
            logger.warning(
                f"{self.component_id}: No base image or figure dict provided. "
                "Using default tolerance."
            )
            self._absolute_click_tolerance = 0.025  # Default small absolute
            return
        
        if fig is None and isinstance(base_image_data, xr.DataArray):
            if base_image_data.ndim == 1:
                dim = base_image_data.dims[0]
                coords = np.asarray(base_image_data.coords[dim].values, dtype=float)
                span = float(coords.max() - coords.min()) if coords.size else 1.0
                self._absolute_click_tolerance = span * self._relative_click_tolerance
                return
            if base_image_data.ndim == 2 and "readout" not in base_image_data.dims:
                try:
                    fig = xarray_to_plotly(base_image_data)
                except Exception as e:
                    logger.warning(
                        f"{self.component_id}: xarray_to_plotly failed in tolerance calc ({e}); using default tolerance."
                    )
                    fig = None
            else:
                fig = None  

        if fig is None:
            logger.warning(
                f"{self.component_id}: No figure or base image provided. "
                "Using default tolerance."
            )
            self._absolute_click_tolerance = 0.025  # Default small absolute
            return

        try:
            x_limits, y_limits = get_axis_limits(fig=fig)
            if x_limits is None or y_limits is None:
                logger.warning(
                    f"{self.component_id}: Could not determine figure range for "
                    f"tolerance from base_image_data. Using default. "
                    f"x_limits: {x_limits}, y_limits: {y_limits}"
                )
                self._absolute_click_tolerance = 0.025
                return

            x_span, y_span = x_limits[1] - x_limits[0], y_limits[1] - y_limits[0]
            diag = np.sqrt(x_span**2 + y_span**2)
            if diag > 1e-9:  # Avoid division by zero or tiny diagonals
                self._absolute_click_tolerance = diag * self._relative_click_tolerance
            else:
                self._absolute_click_tolerance = 0.025  # Fallback for zero-size image

            logger.debug(
                f"{self.component_id}: Absolute click tolerance set to: "
                f"{self._absolute_click_tolerance:.4g}"
            )
        except Exception as e:
            logger.warning(
                f"{self.component_id}: Error calculating tolerance from fig: {e}"
            )
            self._absolute_click_tolerance = 0.025

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
        viewer_ui_state_store_id = orchestrator_stores[
            VideoModeComponent.VIEWER_UI_STATE_STORE_SUFFIX
        ]
        self._register_import_live_frame_callback(
            app, latest_processed_data_store_id, viewer_data_store_id, viewer_ui_state_store_id
        )
        self._register_graph_interaction_callback(
            app, shared_viewer_graph_id, viewer_data_store_id, viewer_ui_state_store_id
        )
        self._register_clear_annotations_callback(
            app, viewer_data_store_id, viewer_ui_state_store_id
        )
        self._register_load_from_disk_callback(
            app, viewer_data_store_id, orchestrator_stores, viewer_ui_state_store_id
        )
        self._register_analysis_callback(
            app, viewer_data_store_id
        )
        self._register_mode_change(
            app, viewer_ui_state_store_id
        )
        self._register_line_select(
            app, viewer_data_store_id
        )
        self._register_line_profile_callback(
            app, viewer_data_store_id
        )
        self._register_reset_2d_callback(
            app, viewer_data_store_id
        )
        self._register_param_editor_callback(app, viewer_data_store_id)
        self._register_apply_options_callback(app, viewer_data_store_id)
        self._register_apply_quam_callback(app, viewer_data_store_id)

    def _register_apply_quam_callback(self, app: Dash, viewer_data_store_id: Dict[str, str]) -> None: 
        @app.callback(
            Output(self._get_id("apply-analysis-status"), "children"),
            Input(self._get_id("apply-analysis-button"), "n_clicks"),
            State(self._get_id("analysis-apply-action"), "value"),
            State(viewer_data_store_id, "data"),
            State({"comp": self.component_id, "kind": "save_points_to_vgs", "field": ALL}, "id"),
            State({"comp": self.component_id, "kind": "save_points_to_vgs", "field": ALL}, "value"),
            State({"axis": ALL, "comp": self.component_id, "kind": "frequency", "type": "param-input"}, "id"),
            State({"axis": ALL, "comp": self.component_id, "kind": "frequency", "type": "param-input"}, "value"),
            State({"axis": ALL, "comp": self.component_id, "kind": "amplitude", "type": "param-input"}, "id"),
            State({"axis": ALL, "comp": self.component_id, "kind": "amplitude", "type": "param-input"}, "value"),

            prevent_initial_call=True,
        )
        def _apply_to_quam(
                n_clicks, 
                action, 
                viewer_ref,
                spv_ids, 
                spv_vals,
                freq_ids,
                freq_vals,
                amp_ids,
                amp_vals,
            ):
                if not n_clicks or not action:
                    raise PreventUpdate
                
                name = selector = duration = None
                for sid, val in zip(spv_ids or [], spv_vals or []):
                    if sid.get("field") == "name": name = val
                    elif sid.get("field") == "selector": selector = val
                    elif sid.get("field") == "duration": duration = None if val in (None, "") else float(val)

                if duration is not None:
                    duration_int = int(duration)
                    if duration_int % 4 != 0: 
                        raise ValueError("Duration must be multiple of 4")
                    duration = duration_int
                                
                freq_map = {}
                for sid, val in zip(freq_ids or [], freq_vals or []):
                    if isinstance(sid, dict) and sid.get("comp") == self.component_id:
                        axis = sid.get("axis")
                        if axis in ("x", "y"):
                            freq_map[axis] = val

                amp_map = {}
                for sid, val in zip(amp_ids or [], amp_vals or []):
                    if isinstance(sid, dict) and sid.get("comp") == self.component_id:
                        axis = sid.get("axis")
                        if axis in ("x", "y"):
                            amp_map[axis] = val

                freq_x, freq_y = freq_map.get("x"), freq_map.get("y")
                amp_x,  amp_y  = amp_map.get("x"),  amp_map.get("y")
                
                if not viewer_ref or viewer_ref.get("key") != data_registry.STATIC_DATA_KEY:
                    return dbc.Alert("No static data available", color="warning", dismissable=True)
                
                try:
                    static_obj = data_registry.get_data(data_registry.STATIC_DATA_KEY) or {}
                    annotations = static_obj.get("annotations", {}) or {}
                    points = annotations.get("points", []) or []
                    axis_modes = self._get_axis_modes

                    if action == "save_points_to_vgs":
                        if selector:
                            point = next((p for p in points if p.get("id") == selector), None)
                        else:
                            point = points[-1] if points else None

                        if not point:
                            raise ValueError("No point selected")
                        if not name:
                            raise ValueError("No point name")

                        ac_voltages_dict = {}
                        dc_voltages_dict = {}
                        if axis_modes.get("x_mode") == "voltage":
                            x_sweep_axis = self._data_acquirer.x_axis
                            if x_sweep_axis.offset_parameter: 
                                x_dc_offset = float(x_sweep_axis.offset_parameter.get_latest())
                            else: 
                                x_dc_offset = 0
                            gate = self._data_acquirer.x_axis.name
                            dc_voltages_dict[gate] = float(x_dc_offset)
                            ac_voltages_dict[gate] = float(point["x"]) - x_dc_offset
                        if axis_modes.get("y_mode") == "voltage":
                            y_sweep_axis = self._data_acquirer.y_axis
                            if y_sweep_axis.offset_parameter: 
                                y_dc_offset = float(y_sweep_axis.offset_parameter.get_latest())
                            else: 
                                y_dc_offset = 0
                            gate = self._data_acquirer.y_axis.name
                            dc_voltages_dict[gate] = float(y_dc_offset)
                            if gate != "dummy":
                                ac_voltages_dict[gate] = float(point["y"]) - y_dc_offset
                        if not ac_voltages_dict:
                            raise ValueError("No voltage axes found")

                        vgs = self._data_acquirer.gate_set
                        if self._data_acquirer.voltage_control_component:
                            dc_set = self._data_acquirer.voltage_control_component.dc_set
                        vgs.add_point(name, ac_voltages_dict, duration=duration)
                        if dc_set is not None: 
                            dc_set.add_point(name, dc_voltages_dict, duration = duration)

                    elif action == "frequency":
                
                        if freq_x is not None and axis_modes.get("x_mode") == "frequency":
                            pulse = self._data_acquirer.x_axis.offset_parameter
                            pulse.channel.intermediate_frequency = float(freq_x)
                        
                        if freq_y is not None and axis_modes.get("y_mode") == "frequency":
                            pulse = self._data_acquirer.y_axis.offset_parameter
                            pulse.channel.intermediate_frequency = float(freq_y)

                    elif action == "amplitude":
                
                        if amp_x is not None and axis_modes.get("x_mode") == "amplitude":
                            pulse = self._data_acquirer.x_axis.offset_parameter
                            pulse.amplitude = float(amp_x)
                        
                        if amp_y is not None and axis_modes.get("y_mode") == "amplitude":
                            pulse = self._data_acquirer.y_axis.offset_parameter
                            pulse.amplitude = float(amp_y)
                    else:
                        raise ValueError(f"Unknown action: {action}")
                    
                    return dbc.Alert("Update Applied", color = "success", dismissable = True, duration = 4000)
                
                except Exception as e:
                    logger.error(f"Error applying to QUAM: {e}")
                    return dbc.Alert(f"Error: {str(e)}", color="danger", dismissable=True)



    def _register_param_editor_callback(self, app:Dash, viewer_data_store_id: Dict[str, str]) -> None: 
        @app.callback(
            Output(self._get_id("analysis-param-editor"), "children"), 
            Input(self._get_id("analysis-apply-action"), "value"), 
            Input(viewer_data_store_id, "data"), 
            prevent_initial_call = True, 
        )

        def _render_param_rows(selected_kind, viewer_ref):
            if not viewer_ref or viewer_ref.get("key") != data_registry.STATIC_DATA_KEY:
                raise PreventUpdate

            static_obj = data_registry.get_data(data_registry.STATIC_DATA_KEY) or {}
            annotations = static_obj.get("annotations", {}) or {}
            points = annotations.get("points", []) or []
            active_pt = self._get_active_point(annotations)

            if not selected_kind:
                return []

            selected_kind = str(selected_kind).lower()
            logger.info(f"selected_kind: {selected_kind}")
            logger.info(f"self._axis_modes: {self._axis_modes}")
            if selected_kind == "save_points_to_vgs":
                opts = [{"label": p.get("id", f"p?{i}"), "value": p.get("id", f"p?{i}")} for i, p in enumerate(points)]
                default_val = active_pt.get("id") if active_pt else None
                return [
                    dbc.Row(
                        [
                            dbc.Col(html.Div("Point to save:"), width=4),
                            dbc.Col(
                                dcc.Dropdown(
                                    id={"comp": self.component_id, "kind": "save_points_to_vgs", "field": "selector"},
                                    options=opts,
                                    value=default_val,
                                    placeholder="Select a point (optional)",
                                    clearable=True,
                                    style={"color": "black"},
                                ),
                                width=8,
                            ),
                        ],
                        className="g-2 mb-2",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(html.Div("Point name:"), width=4),
                            dbc.Col(
                                dbc.Input(
                                    id={"comp": self.component_id, "kind": "save_points_to_vgs", "field": "name"},
                                    type="text",
                                    value=None,
                                    placeholder="Enter point name",
                                ),
                                width=8,
                            ),
                        ],
                        className="g-2 mb-2",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(html.Div("Point duration:"), width=4),
                            dbc.Col(
                                dbc.Input(
                                    id={"comp": self.component_id, "kind": "save_points_to_vgs", "field": "duration"},
                                    type="number",
                                    value=None,
                                    placeholder="Enter point duration",
                                ),
                                width=8,
                            ),
                        ],
                        className="g-2 mb-2",
                    ),
                    html.Div("Press 'Apply to QUAM state' to save.", className="text-muted"),
                ]

            rows = []
            for axis_key in ("x", "y"):
                axis_mode = self._axis_modes.get(f"{axis_key}_mode")
                logger.info(f"axis_key={axis_key}, axis_mode={axis_mode}, selected_kind={selected_kind}, match={axis_mode == selected_kind}")
                if axis_mode != selected_kind:
                    continue 
                try:
                    current_val = self._get_current_params(selected_kind, axis_key)
                except Exception:
                    current_val = None

                new_val = None
                if active_pt is not None:
                    if axis_key == "x" and ("x" in active_pt):
                        new_val = float(active_pt["x"])
                    elif axis_key == "y" and ("y" in active_pt):
                        yv = active_pt.get("y", None)
                        if yv is not None:
                            new_val = float(yv)

                label = "IF" if selected_kind == "frequency" else "Amplitude"
                axis_label = f"{axis_key.upper()}-axis"

                rows.append(
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(
                                    f"Current {label}: {current_val if current_val is not None else '—'}  →  New {label}:",
                                    className="mb-2",
                                ),
                                width=7,
                            ),
                            dbc.Col(
                                dbc.Input(
                                    id={"axis": axis_key, "comp": self.component_id, "kind": selected_kind, "type": "param-input"},
                                    type="number",
                                    value=new_val,
                                    placeholder=f"Enter {label}",
                                ),
                                width=5,
                            ),
                        ],
                        className="g-2 mb-2",
                    )
                )

            if not rows:
                label = "IF" if selected_kind == "frequency" else "Amplitude"
                return [html.Div(f"No {label.lower()} axes active on tab activation.")]

            return rows

    def _register_apply_options_callback(self, app: Dash, viewer_data_store_id) -> None:
        @app.callback(
            Output(self._get_id("analysis-apply-action"), "options"),
            Output(self._get_id("analysis-apply-action"), "value"),
            Input(viewer_data_store_id, "data"),
            State(self._get_id("analysis-apply-action"), "value"),
            prevent_initial_call=False,
        )
        def _refresh_opts(_viewer_ref, current_value):
            opts = self._apply_dropdown_options()
            valid_values = {o["value"] for o in opts}
            new_value = current_value if current_value in valid_values else None
            return opts, new_value

    def _register_mode_change(
            self,
            app:Dash,
            viewer_ui_state_store_id: Dict[str, Any],
    ) -> None:
        """Callback to reset all selected points when changing the mode"""

        @app.callback(
            Output(viewer_ui_state_store_id, "data", allow_duplicate=True),
            Input(self._get_id(self._MODE_SELECTOR_SUFFIX), "value"),
            prevent_initial_call = True,
        )
        def _mode_change(
            mode: str,
        ) -> Dict[str, Any]:
            # NOT self._reset_transient_state(), otherwise labels are reset!
            self._selected_point_to_move = {"is_moving": False, "point_id": None}
            self._selected_indices_for_line = []
            viewer_ui_state_store_payload = {
                "selected_point_to_move": self._selected_point_to_move["point_id"],
                "selected_point_for_line": self._selected_indices_for_line,
                "show_labels": self._show_labels,
            }
            return viewer_ui_state_store_payload

    def _register_import_live_frame_callback(
        self,
        app: Dash,
        latest_processed_data_store_id: Dict[str, str],
        viewer_data_store_id: Dict[str, str],
        viewer_ui_state_store_id: Dict[str, Any],
    ) -> None:
        """Callback to import the current live frame as a static snapshot."""

        @app.callback(
            Output(viewer_data_store_id, "data"),
            Output(viewer_ui_state_store_id, "data"),
            Input(self._get_id(self._IMPORT_LIVE_FRAME_BUTTON_SUFFIX), "n_clicks"),
            State(latest_processed_data_store_id, "data"),
            prevent_initial_call=True,
        )
        def _import_live_frame(
            n_clicks: int, live_data_ref: Optional[Dict[str, Any]]
        ) -> Tuple[Dict[str, Any],Dict[str, Any]]:
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
            self._axis_modes = self._get_axis_modes
            viewer_ui_state_store_payload = {
                "selected_point_to_move": self._selected_point_to_move["point_id"],
                "selected_point_for_line": self._selected_indices_for_line,
                "show_labels": self._show_labels,
            }
            bi = new_static_object.get("base_image_data")
            if isinstance(bi, xr.DataArray) and bi.ndim == 2 and "readout" not in bi.dims:
                try:
                    self._update_click_tolerance(base_image_data=bi)
                except Exception as e:
                    logger.warning(
                        f"{self.component_id}: tolerance from base_image failed ({e}); using default."
                    )
                    self._update_click_tolerance()
            else:
                self._update_click_tolerance()

            logger.info(
                f"{self.component_id}: Imported live frame. New static data version: {new_version}"
            )
            return {"key": data_registry.STATIC_DATA_KEY, "version": new_version}, viewer_ui_state_store_payload
    def _register_line_select(self, app: Dash, viewer_data_store_id) -> None: 
        @app.callback(
            Output(f"{self.component_id}-line-selector", "options"), 
            Output(f"{self.component_id}-line-selector", "value"), 
            Input(viewer_data_store_id, "data"),
            State(f"{self.component_id}-line-selector", "value"), 
            prevent_initial_call = True
        )
        def _refresh_line_dropdown(viewer_ref: Optional[Dict[str,Any]], current_value: Optional[str]):
            if (not viewer_ref or viewer_ref.get("key") != data_registry.STATIC_DATA_KEY):
                raise PreventUpdate
            
            annotations = data_registry.get_data(data_registry.STATIC_DATA_KEY).get("annotations", {})
            lines = annotations.get("lines", [])

            options = [{"label": f'{ln["id"]} ({ln["start_point_id"]}-{ln["end_point_id"]})',"value": ln["id"],}
            for ln in lines if "id" in ln]

            if not options: 
                return [], None
            
            valid_ids = {o["value"] for o in options}

            if current_value in valid_ids:
                return options, no_update

            return options, None
        
    def _register_reset_2d_callback(self, app: Dash, viewer_data_store_id: Dict[str, str]) -> None:
        @app.callback(
            Output(viewer_data_store_id, "data", allow_duplicate=True),
            Input(self._get_id("reset-2d-button"), "n_clicks"),
            State(viewer_data_store_id, "data"),
            prevent_initial_call=True,
        )
        def _reset_to_2d(n_clicks, current_viewer_data_ref):
            if not current_viewer_data_ref or current_viewer_data_ref.get("key") != data_registry.STATIC_DATA_KEY:
                raise PreventUpdate
            static_obj = data_registry.get_data(data_registry.STATIC_DATA_KEY)
            if not static_obj:
                raise PreventUpdate
            new_static = dict(static_obj)
            new_static.pop("profile_plot", None)
            new_version = data_registry.set_data(data_registry.STATIC_DATA_KEY, new_static)
            return {"key": data_registry.STATIC_DATA_KEY, "version": new_version}
    def _extract_1d_profile(
            self,
            data,
            x0: float,
            y0: float,
            x1: float,
            y1: float,
            n_samples: int = 200,
            interpolate: bool = False) -> Tuple[np.ndarray, np.ndarray]: 
        if data.ndim != 2:
            raise ValueError("Expected a 2D xarray")
        y_dim, x_dim = data.dims
        x_coordinates, y_coordinates = np.asarray(data.coords[x_dim].values), np.asarray(data.coords[y_dim].values)
        x = np.linspace(x0, x1, n_samples)
        y = np.linspace(y0, y1, n_samples)
        dx = np.diff(x)
        dy = np.diff(y)
        s = np.concatenate([[0.0], np.cumsum(np.sqrt(dx**2 + dy**2))])

        if not interpolate: 
            ix = np.abs(x[:, None] - x_coordinates[None, :]).argmin(axis=1)
            iy = np.abs(y[:, None] - y_coordinates[None, :]).argmin(axis=1)
            
            vals = data.values[iy, ix]
            return s, vals
        
        A = data.values
        def neighbours_and_weight(coord: np.ndarray, q: np.ndarray):
            N = coord.size
            asc = coord[0] <= coord[-1]

            if asc: 
                i1 = np.searchsorted(coord, q, side="left")
                i1 = np.clip(i1, 1, N - 1)
                i0 = i1 - 1
            else:
                coord_rev = coord[::-1]
                j1 = np.searchsorted(coord_rev, q, side="left")
                j1 = np.clip(j1, 1, N - 1)
                j0 = j1 - 1
                i0 = N - j1
                i1 = N - j1 - 1

            c0 = coord[i0]
            c1 = coord[i1]
            denom = (c1 - c0)
            w = np.zeros_like(q, dtype=float)
            valid = denom != 0
            w[valid] = (q[valid] - c0[valid]) / denom[valid]
            w = np.clip(w, 0.0, 1.0)
            return i0, i1, w

        i0_y, i1_y, wy = neighbours_and_weight(y_coordinates, y)
        j0_x, j1_x, wx = neighbours_and_weight(x_coordinates, x)

        v00 = A[i0_y, j0_x]
        v10 = A[i0_y, j1_x]
        v01 = A[i1_y, j0_x]
        v11 = A[i1_y, j1_x]

        vals = (1 - wx) * (1 - wy) * v00 \
            + wx  * (1 - wy) * v10 \
            + (1 - wx) * wy  * v01 \
            + wx  * wy  * v11

        return s, vals
            
    def _register_line_profile_callback(self, app: Dash, viewer_data_store_id: Dict[str, str]) -> None:

        @app.callback(
            Output(viewer_data_store_id, "data", allow_duplicate=True), 
            Input(self._get_id(self._LINE_PROFILE_BUTTON_SUFFIX), "n_clicks"),
            Input(self._get_id("interpolation_toggle"), "value"),
            Input(f"{self.component_id}-line-selector", "value"),
            State(viewer_data_store_id, "data"), 
            prevent_initial_call = True,
        )
        def _extract_profile(_n_clicks: int, toggle_value, selected_line_id, current_viewer_data_ref: Optional[Dict[str,str]]):
            if (not current_viewer_data_ref or current_viewer_data_ref.get("key") != data_registry.STATIC_DATA_KEY): 
                raise PreventUpdate
            
            trig_id = ctx.triggered_id
            dropdown_id = f"{self.component_id}-line-selector"
            if trig_id == dropdown_id and selected_line_id is None: 
                raise PreventUpdate
            
            static_data_object = data_registry.get_data(data_registry.STATIC_DATA_KEY)
            interpolate = bool(toggle_value) and ("on" in toggle_value)
            bi = static_data_object.get("base_image_data")
            target_readout = None
            if isinstance(bi, xr.DataArray) and bi.ndim == 3 and "readout" in bi.dims:
                rd = bi.coords["readout"].values[0]     # default: first readout
                target_readout = str(rd)
                data_2d = bi.sel(readout=rd).drop_vars("readout", errors="ignore").squeeze(drop=True)
            elif isinstance(bi, xr.DataArray) and bi.ndim == 2:
                data_2d = bi
            else:
                # Nothing we can profile
                raise PreventUpdate

            annotations = static_data_object.get("annotations", {})
            lines = annotations.get("lines", [])
            points = {p["id"]: p for p in annotations.get("points", [])}
            if len(lines) == 0 or len(points) < 2: 
                raise PreventUpdate
            
            line_id = selected_line_id
            line = next((ln for ln in lines if ln.get("id") == line_id), None)
            if line is None: 
                raise PreventUpdate
            
            p1 = points.get(line["start_point_id"])
            p2 = points.get(line["end_point_id"])

            if p1 is None or p2 is None: 
                raise PreventUpdate
            
            x0, y0, x1, y1 = float(p1["x"]),  float(p1["y"]), float(p2["x"]), float(p2["y"])
            s, vals = self._extract_1d_profile(data_2d, x0, y0, x1, y1, 200, interpolate)
            target_col = int(line.get("subplot_col", 1))
            new_static = dict(static_data_object)
            new_static["profile_plot"] = {
                "s": s.tolist(),
                "vals": vals.tolist(),
                "name": f"Line {line['id']}",
                "y_label": str((data_2d.name or "Value")),
                "x_label": "Distance",
                "readout": target_readout,
                "subplot_col": target_col, 
            }
            new_version = data_registry.set_data(data_registry.STATIC_DATA_KEY, new_static)
            return {"key": data_registry.STATIC_DATA_KEY, "version": new_version}

    def _handle_point_mode_interaction(
        self,
        x: float,
        y: float,
        clicked_annotation_point_id: Optional[str],  # Now string ID
        current_annotations: Dict[str, List[Dict[str, Any]]],
        subplot_col: int = 1, 
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
            new_point = {"id": new_point_id, "x": x, "y": y}
            if self._is_subplot_mode():
                new_point["subplot_col"] = subplot_col
            points_list.append(new_point)
            changed = True
        return changed

    def _handle_line_mode_interaction(
        self,
        clicked_annotation_point_id: Optional[str],
        current_annotations: Dict[str, List[Dict[str, Any]]],
        subplot_col: int = 1,
    ) -> bool:
        """Handles graph interactions for 'line' mode."""
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
                new_line = {
                    "id": new_line_id,
                    "start_point_id": id1,
                    "end_point_id": id2,
                }
                if self._is_subplot_mode():
                    new_line["subplot_col"] = subplot_col
                lines_list.append(new_line)
                changed = True
            self._selected_indices_for_line.clear()
        return changed
    
    def _handle_delete_mode_interaction(
        self,
        x: float,
        y: float,
        clicked_annotation_point_id: Optional[str],
        current_annotations: Dict[str, List[Dict[str, Any]]],
        subplot_col: int = 1,
    ) -> bool:
        """Handles graph interactions for 'delete' mode. Modifies current_annotations. Returns True if changed."""
        points_list = current_annotations["points"]
        lines_list = current_annotations["lines"]
        changed = False

        if not self._is_subplot_mode():
            if clicked_annotation_point_id is not None:
                original_len = len(points_list)
                current_annotations["points"] = [
                    p for p in points_list if p["id"] != clicked_annotation_point_id
                ]
                if len(current_annotations["points"]) < original_len:
                    changed = True
                    # Also remove lines connected to this point
                    current_annotations["lines"] = [
                        line for line in lines_list
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
        
        def _pcol(pid: str) -> int:
            for p in points_list:
                if p["id"] == pid:
                    return int(p.get("subplot_col", 1))
            return 1
            
        if clicked_annotation_point_id is not None:
            if _pcol(clicked_annotation_point_id) != subplot_col:
                return False
            original_len = len(points_list)
            current_annotations["points"] = [
                p for p in points_list if p["id"] != clicked_annotation_point_id
            ]
            if len(current_annotations["points"]) < original_len:
                changed = True
                current_annotations["lines"] = [
                    ln for ln in lines_list
                    if not (
                        (ln["start_point_id"] == clicked_annotation_point_id
                        or ln["end_point_id"] == clicked_annotation_point_id)
                        and int(ln.get("subplot_col", _pcol(ln.get("start_point_id","")))) == subplot_col
                    )
                ]
            return changed
            
        pts_subset = [p for p in points_list if int(p.get("subplot_col", 1)) == subplot_col]
        pt_ids_subset = {p["id"] for p in pts_subset}
        lines_subset = [
            ln for ln in lines_list
            if int(ln.get("subplot_col", _pcol(ln.get("start_point_id","")))) == subplot_col
            and ln.get("start_point_id") in pt_ids_subset
            and ln.get("end_point_id") in pt_ids_subset
        ]
        if not lines_subset:
            return False
        subset_ann = {"points": pts_subset, "lines": lines_subset}
        line_to_delete_id = find_closest_line_id(
            x, y, subset_ann, self._absolute_click_tolerance
        )
        if line_to_delete_id is not None:
            original_len = len(lines_list)
            current_annotations["lines"] = [
                ln for ln in lines_list if ln["id"] != line_to_delete_id
            ]
            if len(current_annotations["lines"]) < original_len:
                changed = True
        return changed

    def extract_click_data(self, point_info):
        """Extract click data with minimal complexity - revert to original logic"""
        x, y = point_info["x"], point_info["y"]
        logger.debug(f"  clickData point_info: {point_info}")
        name = str(point_info.get("name", ""))
        logger.debug(f"  curve_name: {name}")
        cd = point_info.get("customdata", None)
        logger.debug(f"  custom_data_val: {cd}")
        subplot_col = 1
        if isinstance(cd, (int, float)):
            subplot_col = int(cd)
        elif isinstance(cd, list) and cd and isinstance(cd[0], (int, float)):
            subplot_col = int(cd[0])
        else:
            xaxis_id = point_info.get("xaxis")
            if isinstance(xaxis_id, str) and xaxis_id.startswith("x"):
                try:
                    subplot_col = 1 if xaxis_id == "x" else int(xaxis_id[1:])
                except Exception:
                    subplot_col = 1

        clicked_annotation_point_id = None
        if name.startswith("annotations_point"):
            if isinstance(cd, list) and cd and isinstance(cd[0], str):
                clicked_annotation_point_id = cd[0]
            elif isinstance(cd, str):
                clicked_annotation_point_id = cd
            
        return x, y, subplot_col, clicked_annotation_point_id
        
    def _handle_translate_all_mode_interaction(
            self,
            x: float,
            y: float,
    ) -> None:
        """Handles graph interactions for 'translate all' mode. """
        # After first click: True (turn translation mode on); after second click: False (turn translation mode off)
        self._translate_all['translate'] = not self._translate_all['translate']

        # Add the clicked point when translation mode is turned on
        if self._translate_all['translate'] == True:
            self._translate_all['clicked_point_x'] = x
            self._translate_all['clicked_point_y'] = y
        # Remove the clicked point when translation mode is turned off
        else: 
            self._translate_all['clicked_point_x'] = None
            self._translate_all['clicked_point_y'] = None
        logging.debug(f'_translate_all: {self._translate_all}')
        return

    def _handle_hover_interaction(
        self,
        hover_data: Dict[str, Any],
        mode: str,
        current_annotations: Dict[str, List[Dict[str, Any]]],
    ) -> bool:
        """Handles graph hover for 'point' move. Modifies current_annotations. Returns True if changed."""
        if not(
            (
                mode == "point"
                and self._selected_point_to_move.get("is_moving")
                and self._selected_point_to_move.get("point_id") is not None
            )
            or
            (
                mode=="translate-all"
                and self._translate_all["translate"] == True
            )
        ):
            return False

        points_list = current_annotations["points"]
        hover_point_info = hover_data["points"][0]
        changed = False

        if mode == "point":
            point_id_to_move = self._selected_point_to_move["point_id"]

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
        
        elif mode == "translate-all":
            # Compute the direction vector (hover point - clicked point)
            x_hover, y_hover = hover_point_info["x"], hover_point_info["y"]
            x_clicked, y_clicked = self._translate_all["clicked_point_x"], self._translate_all["clicked_point_y"]
            dx, dy = x_hover - x_clicked, y_hover - y_clicked

            # Update the current annotations: Current coordinates + direction vector
            for point in points_list:
                point["x"], point["y"] = point["x"] + dx, point["y"] + dy
                changed = True

            # Update the clicked point
            x_clicked, y_clicked = x_clicked + dx, y_clicked + dy
            self._translate_all["clicked_point_x"], self._translate_all["clicked_point_y"] = x_clicked, y_clicked

        return changed

    def _register_graph_interaction_callback(
        self,
        app: Dash,
        shared_viewer_graph_id: Dict[str, str],
        viewer_data_store_id: Dict[str, str],
        viewer_ui_state_store_id: Dict[str, Any],
    ) -> None:
        """Callback to handle user interactions with the graph (clicks, hovers)."""

        @app.callback(
            Output(viewer_data_store_id, "data", allow_duplicate=True),
            Output(viewer_ui_state_store_id, "data", allow_duplicate=True),
            Input(shared_viewer_graph_id, "clickData"),
            Input(shared_viewer_graph_id, "hoverData"),
            Input(self._get_id(self._SHOW_LABELS_CHECKLIST_SUFFIX), "value"),
            State(self._get_id(self._MODE_SELECTOR_SUFFIX), "value"),
            State(viewer_data_store_id, "data"),  # Get current key and version
            prevent_initial_call=True,
        )
        def _handle_graph_interactions(
            click_data: Optional[Dict[str, Any]],
            hover_data: Optional[Dict[str, Any]],
            labels_list: List[str],
            mode: str,
            current_viewer_data_ref: Optional[Dict[str, str]],
        ) -> Tuple[Dict[str, Any],Dict[str, Any]]:
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
            logger.debug(f"triggered_prop_id (normalized): {interaction_type}")

            if interaction_type == "clickData":
                if click_data and click_data.get("points"):
                    pt = click_data["points"][0]
                    sig = (pt.get("curveNumber"), pt.get("pointNumber"), pt.get("x"), pt.get("y"))
                    if getattr(self, "_last_click_sig", None) == sig:
                        raise PreventUpdate
                    self._last_click_sig = sig
            elif interaction_type == "hoverData":
                pass
            else:
                raise PreventUpdate

            if (
                interaction_type == "clickData"
                and click_data
                and click_data.get("points")
            ):
                point_info = click_data["points"][0]
                x, y, subplot_col, clicked_annotation_point_id = self.extract_click_data(point_info)   

                if mode == "line" and clicked_annotation_point_id is None:
                    pts = annotations_copy.get("points", [])
                    if self._is_subplot_mode():
                        pts = [p for p in pts if int(p.get("subplot_col", 1)) == int(subplot_col)]
                    tol = self._absolute_click_tolerance
                    best_id, best_d2 = None, None
                    for p in pts:
                        dx, dy = float(p["x"]) - x, float(p["y"]) - y
                        d2 = dx*dx + dy*dy
                        if d2 <= tol*tol and (best_d2 is None or d2 < best_d2):
                            best_id, best_d2 = p["id"], d2
                    clicked_annotation_point_id = best_id 
                if mode == "point":
                    interaction_changed_data = self._handle_point_mode_interaction(
                        x, y, clicked_annotation_point_id, annotations_copy, subplot_col
                    )
                elif mode == "line":
                    interaction_changed_data = self._handle_line_mode_interaction(
                        clicked_annotation_point_id, annotations_copy, subplot_col
                    )
                elif mode == "delete":
                    interaction_changed_data = self._handle_delete_mode_interaction(
                        x, y, clicked_annotation_point_id, annotations_copy, subplot_col
                    )
                elif mode == "translate-all":
                    self._handle_translate_all_mode_interaction(
                        x,y
                    )
            elif (
                interaction_type == "hoverData"
                and hover_data
                and hover_data.get("points")
            ):
                interaction_changed_data = self._handle_hover_interaction(
                    hover_data, mode, annotations_copy
                )
            elif (
                interaction_type == "value"  # Checklist for showing labels
            ):
                self._show_labels = labels_list
                logging.info(f"Callback triggered by checklist. Show labels: {self._show_labels}")

            # Potential improvement: Only update viewer_ui_state_store_payload, if there are changes.
            viewer_ui_state_store_payload = {
                "selected_point_to_move": self._selected_point_to_move["point_id"],
                "selected_point_for_line": self._selected_indices_for_line,
                "selected_points_for_line": self._selected_indices_for_line,
                "show_labels": self._show_labels,
            }

            if interaction_changed_data:
                new_static_data_object = {
                    "base_image_data": static_data_object["base_image_data"],
                    "annotations": annotations_copy,
                }
                new_version = data_registry.set_data(
                    data_registry.STATIC_DATA_KEY, new_static_data_object
                )
                self._next_point_id_counter = get_next_index(
                    annotations_copy.get("points", []), "p"
                )
                self._next_line_id_counter = get_next_index(
                    annotations_copy.get("lines", []), "l"
                )
                logger.debug(
                    f"{self.component_id}: Annotations updated. New version: {new_version}"
                )
                return {"key": data_registry.STATIC_DATA_KEY, "version": new_version}, viewer_ui_state_store_payload

            else:
                logger.debug("No changes to annotations") #, raising PreventUpdate")
                #raise PreventUpdate
                return dash.no_update, viewer_ui_state_store_payload

    def _register_clear_annotations_callback(
        self, 
        app: Dash, 
        viewer_data_store_id: Dict[str, str], 
        viewer_ui_state_store_id: Dict[str, Any],
    ) -> None:
        """Callback to clear all current annotations."""

        @app.callback(
            Output(viewer_data_store_id, "data", allow_duplicate=True),
            Output(viewer_ui_state_store_id, "data", allow_duplicate=True),
            Input(self._get_id(self._CLEAR_BUTTON_SUFFIX), "n_clicks"),
            State(viewer_data_store_id, "data"),
            prevent_initial_call=True,
        )
        def _clear_annotations(
            _n_clicks: int, current_viewer_data_ref: Optional[Dict[str, str]]
        ) -> Tuple[Dict[str, Any],Dict[str, Any]]:
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
            viewer_ui_state_store_payload = {
                "selected_point_to_move": self._selected_point_to_move["point_id"],
                "selected_point_for_line": self._selected_indices_for_line,
                "show_labels": self._show_labels,
            }

            logger.info(
                f"{self.component_id}: Cleared annotations. New version: {new_version}"
            )
            return {"key": data_registry.STATIC_DATA_KEY, "version": new_version}, viewer_ui_state_store_payload

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

    def _register_load_from_disk_callback(
        self,
        app: Dash,
        viewer_data_store_id: Dict[str, str],
        orchestrator_stores: Dict[str, Any],
        viewer_ui_state_store_id: Dict[str, Any],
    ) -> None:
        from qua_dashboards.video_mode.video_mode_component import VideoModeComponent

        main_status_alert_id = orchestrator_stores.get(
            VideoModeComponent._MAIN_STATUS_ALERT_ID_SUFFIX
        )

        @app.callback(
            Output(viewer_data_store_id, "data", allow_duplicate=True),
            Output(viewer_ui_state_store_id, "data", allow_duplicate=True),
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
            viewer_ui_state_store_payload = {
                "selected_point_to_move": self._selected_point_to_move["point_id"],
                "selected_point_for_line": self._selected_indices_for_line,
                "show_labels": self._show_labels,
            }

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
                viewer_ui_state_store_payload,
                dbc.Alert(
                    f"Successfully loaded data for idx '{idx}'.",
                    color="success",
                    dismissable=True,
                ),
            )

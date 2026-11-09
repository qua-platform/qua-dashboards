import logging
import uuid
from typing import Any, Dict, Union, Optional
import numpy as np
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
    _GATE_AXIS_REFRESH_INTERVAL_SUFFIX = "gate-axis-refresh-interval"
    _NUM_PULSE_OFFSET_SLOTS = 5

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
        gate_set = getattr(self._data_acquirer_instance, "gate_set", None)
        self._known_gate_names = (
            tuple(gate_set.valid_channel_names)
            if gate_set is not None and hasattr(gate_set, "valid_channel_names")
            else tuple()
        )
        self._last_overlay_text: Optional[str] = None
        self._last_overlay_version: Optional[str] = None
        self._last_gridlines_state: Optional[str] = None
        logger.info(
            f"LiveViewTabController '{self.component_id}' initialized with "
            f"Data Acquirer '{self._data_acquirer_instance.component_id}'."
        )

    def _scan_supports_fast_axis_direction(self) -> bool:
        """Return True when the currently selected scan mode has a directional fast axis."""
        scan_mode = getattr(self._data_acquirer_instance, "scan_mode", None)
        return scan_mode is not None and hasattr(scan_mode, "fast_axis_direction")

    @staticmethod
    def _component_contains_id_fragment(component: Any, fragment: str) -> bool:
        """Recursively check whether a Dash component tree contains an ID fragment."""
        if component is None:
            return False

        component_id = getattr(component, "id", None)
        if component_id is not None and fragment in str(component_id):
            return True

        children = getattr(component, "children", None)
        if children is None:
            return False
        if isinstance(children, (list, tuple)):
            return any(
                LiveViewTabController._component_contains_id_fragment(child, fragment)
                for child in children
            )
        return LiveViewTabController._component_contains_id_fragment(children, fragment)

    def _build_swap_xy_row(self) -> dbc.Row:
        """Build a compact Swap X/Y button aligned below the Y Points control."""
        return dbc.Row(
            [
                dbc.Col(html.Div(), width=2),
                dbc.Col(html.Div(), width=5),
                dbc.Col(
                    html.Div(
                        dbc.Button(
                            "⇄ Swap",
                            id=self._get_id("swap-xy-button"),
                            color="secondary",
                            outline=True,
                            size="sm",
                            style={
                                "width": "50%",
                                "minWidth": "78px",
                                "height": "32px",
                                "whiteSpace": "nowrap",
                            },
                        ),
                        className="d-flex justify-content-end",
                    ),
                    width=5,
                ),
            ],
            className="g-2 mb-2 align-items-center",
        )

    def _build_fast_axis_direction_row(self) -> dbc.Row:
        """Build the fast-axis direction control inside Acquirer Parameters."""
        scan_mode = getattr(self._data_acquirer_instance, "scan_mode", None)
        direction = str(
            getattr(scan_mode, "fast_axis_direction", "forward")
        ).lower()
        is_reverse = direction == "reverse"

        return dbc.Row(
            [
                dbc.Col(
                    "Fast-axis direction",
                    width=4,
                    className="d-flex align-items-center",
                ),
                dbc.Col(
                    dbc.Button(
                        "← Reverse" if is_reverse else "→ Forward",
                        id=self._get_id("fast-axis-direction-button"),
                        color="warning" if is_reverse else "secondary",
                        outline=True,
                        size="sm",
                        style={
                            "minWidth": "125px",
                            "height": "36px",
                            "whiteSpace": "nowrap",
                        },
                    ),
                    width="auto",
                ),
            ],
            id=self._get_id("fast-axis-direction-row"),
            className="g-2 mb-2 align-items-center",
            style={} if self._scan_supports_fast_axis_direction() else {"display": "none"},
        )

    def _build_acquirer_controls(self):
        """Order Live View controls as axes -> direction -> software averages -> others."""
        if not self._data_acquirer_instance:
            return [html.P("Data acquirer components could not be loaded.")]

        raw_controls = list(
            self._data_acquirer_instance.get_dash_components(
                include_subcomponents=True,
                include_inner_loop_controls=self._show_inner_loop_controls,
            )
        )

        axis_controls = []
        software_average_controls = []
        other_controls = []

        for control in raw_controls:
            if self._component_contains_id_fragment(control, "gate-select-x"):
                axis_controls.append(control)
            elif self._component_contains_id_fragment(control, "num_software_averages"):
                software_average_controls.append(control)
            else:
                other_controls.append(control)

        return (
            axis_controls
            + [self._build_swap_xy_row()]
            + [self._build_fast_axis_direction_row()]
            + software_average_controls
            + other_controls
        )

    def _get_pulse_offset_gate_names(self):
        """Return physical and virtual gate names while preserving GateSet ordering."""
        gate_set = getattr(self._data_acquirer_instance, "gate_set", None)
        if gate_set is None:
            return [], []

        physical_gate_names = list(getattr(gate_set, "channels", {}).keys())
        valid_names = list(getattr(gate_set, "valid_channel_names", []) or [])
        physical_name_set = set(physical_gate_names)
        virtual_gate_names = [
            name for name in valid_names if name not in physical_name_set
        ]
        return physical_gate_names, virtual_gate_names

    def _build_pulse_offsets_section(self):
        """Build five optional physical/virtual-gate pulse-offset controls."""
        inner = getattr(self._data_acquirer_instance, "inner_loop_action", None)
        current_offsets = dict(getattr(inner, "pulse_offsets", {}) or {})
        current_items = list(current_offsets.items())[: self._NUM_PULSE_OFFSET_SLOTS]

        physical_gate_names, virtual_gate_names = self._get_pulse_offset_gate_names()
        selectable_gate_names = physical_gate_names + virtual_gate_names

        gate_options = [{"label": "None", "value": ""}]
        if physical_gate_names:
            gate_options.append(
                {"label": "Physical", "value": "__physical_header__", "disabled": True}
            )
            gate_options.extend(
                {"label": name, "value": name} for name in physical_gate_names
            )
        if virtual_gate_names:
            gate_options.append(
                {"label": "Virtual", "value": "__virtual_header__", "disabled": True}
            )
            gate_options.extend(
                {"label": name, "value": name} for name in virtual_gate_names
            )

        rows = []
        for slot in range(self._NUM_PULSE_OFFSET_SLOTS):
            if slot < len(current_items):
                gate_name, offset_value = current_items[slot]
            else:
                gate_name, offset_value = "", 0.0

            rows.append(
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Dropdown(
                                id=self._get_id(f"pulse-offset-gate-{slot}"),
                                options=gate_options,
                                value=gate_name if gate_name in selectable_gate_names else "",
                                clearable=False,
                                searchable=True,
                                placeholder="Gate",
                                style={
                                    "color": "#212529",
                                    "width": "100px",
                                    "height": "38px",
                                },
                            ),
                            width="auto",
                            className="pe-0",
                        ),
                        dbc.Col(
                            dbc.InputGroup(
                                [
                                    dbc.Input(
                                        id=self._get_id(f"pulse-offset-value-{slot}"),
                                        type="number",
                                        value=float(offset_value),
                                        step="any",
                                        debounce=True,
                                        disabled=not bool(gate_name),
                                        style={"width": "78px"},
                                    ),
                                    dbc.InputGroupText("V"),
                                ],
                                size="sm",
                                style={"width": "112px",
                                       "height": "38px"
                                },
                            ),
                            width="auto",
                        ),
                    ],
                    className="g-2 mb-2 align-items-center justify-content-start",
                )
            )

        return html.Div(
            [
                html.H6("Pulse offsets", className="text-light mb-2"),
                html.Div(rows),
            ],
            className="mb-2",
        )

    def get_layout(self) -> dbc.Card:
        """Generates the Dash layout for the Live View control panel."""
        logger.debug(
            f"Generating layout for LiveViewTabController '{self.component_id}'"
        )
        acquisition_section = html.Div(
            [
                html.H6("Acquisition", className="text-light mb-2"),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button(
                                "Start",
                                id=self._get_id(self._TOGGLE_ACQ_BUTTON_ID_SUFFIX),
                                color="success",
                                style={
                                    "width": "110px",
                                    "height": "38px",
                                },
                            ),
                            width="auto",
                        ),
                        dbc.Col(html.Div(), width=True),
                        dbc.Col(
                            dbc.Button(
                                "Set DC point",
                                id=self._get_id("set-dc-center-button"),
                                color="primary",
                                disabled=True,
                                style={
                                    "width": "145px",
                                    "height": "38px",
                                    "whiteSpace": "nowrap",
                                },
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    html.Div(
                                        "Status:",
                                        style={
                                            "fontSize": "0.8rem",
                                            "marginBottom": "3px",
                                            "textAlign": "center",
                                        },
                                    ),
                                    dbc.Badge(
                                        "STOPPED",
                                        id=self._get_id(self._ACQUIRER_STATUS_INDICATOR_ID_SUFFIX),
                                        color="secondary",
                                        className="p-2 d-flex align-items-center justify-content-center",
                                        style={
                                            "fontSize": "0.9rem",
                                            "width": "120px",
                                            "height": "38px",
                                            "textAlign": "center",
                                        },
                                    ),
                                ]
                            ),
                            width="auto",
                        ),
                    ],
                    className="g-2 align-items-end",
                ),
                dcc.Store(
                    id=self._get_id("set-dc-center-armed"),
                    data=False,
                ),
            ],
            className="mb-2",
        )
        acquirer_specific_controls = self._build_acquirer_controls()
        acquirer_section = html.Div(
            [
                html.H6("Acquirer Parameters", className="text-light mb-2"),
                html.Div(
                    id=self._get_id(self._ACQUIRER_CONTROLS_DIV_ID_SUFFIX),
                    children=acquirer_specific_controls,
                    className="p-3 border rounded",
                ),
            ],
            className="mb-2",
        )
        pulse_offsets_section = self._build_pulse_offsets_section()
        display_section = html.Div(
            [
                html.H6("Display helper", className="text-light mb-2"),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Checklist(
                                id=self._get_id("center-marker-toggle"),
                                options=[{
                                    "label": html.Span(["✛ ", "Crosshair"]),
                                    "value": "on",
                                }],
                                value=[],
                                switch=True,
                                inline=True,
                                className="mb-0",
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            dbc.Checklist(
                                id=self._get_id("mouse-crosshairs-toggle"),
                                options=[{
                                    "label": html.Span(["✛🖱 ", "Mouse"]),
                                    "value": "on",
                                }],
                                value=[],
                                switch=True,
                                inline=True,
                                className="mb-0",
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            dbc.Checklist(
                                id=self._get_id("show-full-grid-toggle"),
                                options=[{
                                    "label": html.Span(["▦ ", "Grid"]),
                                    "value": "on",
                                }],
                                value=[],
                                switch=True,
                                inline=True,
                                className="mb-0",
                            ),
                            width="auto",
                        ),
                    ],
                    className="g-3 align-items-center flex-nowrap",
                ),
                dbc.Row(
                    [
                        dbc.Col("Grid opacity", width=4),
                        dbc.Col(
                            dcc.Slider(
                                id=self._get_id("grid_opacity"),
                                min=0,
                                max=100,
                                step=5,
                                value=40,
                                marks={0: "0%", 100: "100%"},
                                tooltip={"always_visible": False, "placement": "bottom"},
                                updatemode="drag",
                            ),
                            width=8,
                        ),
                    ],
                    className="g-2 align-items-center mt-1",
                ),
            ],
            className="mb-2",
        )
        saving_section = getattr(self, "saving_section", None)
        card_children = [
            acquisition_section,
            html.Hr(className="my-3"),
            acquirer_section,
            html.Hr(className="my-3"),
            pulse_offsets_section,
            html.Hr(className="my-3"),
            display_section,
        ]
        if saving_section is not None:
            card_children.extend([
                html.Hr(className="my-3"),
                saving_section,
            ])
        if self._supports_dynamic_gate_axis_refresh():
            card_children.append(
                dcc.Interval(
                    id=self._get_id(self._GATE_AXIS_REFRESH_INTERVAL_SUFFIX),
                    interval=1000,
                    n_intervals=0,
                )
            )

        card_children.append(
            html.Div(
                id=self._get_id(self._DUMMY_OUTPUT_ACQUIRER_UPDATE_SUFFIX),
                style={"display": "none"},
            )
        )
        return dbc.Card(
            dbc.CardBody(card_children, className="p-3"),
            color="dark",
            inverse=True,
            className="tab-card-dark",
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
        shared_viewer_graph_id: str = None,
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
        if self._supports_dynamic_gate_axis_refresh():
            self._register_dynamic_gate_axis_refresh_callback(app)
        self._register_set_dc_button_callback(app)
        self._register_gridlines_callback(app, shared_viewer_store_ids)
        self._register_readoutparams_callback(app)
        self._register_mode_callback(app)
        self._register_fast_axis_direction_callback(app)
        self._register_swap_xy_callback(app)
        self._register_pulse_offsets_callback(app)
        self._register_fast_axis_direction_visibility_callback(
            app, orchestrator_stores.get("control-tabs")
        )
        if shared_viewer_graph_id is not None:
            self._register_click_to_centre_callback(app, shared_viewer_graph_id, tabs_id=orchestrator_stores.get("control-tabs"),)

    def _register_readoutparams_callback(self, app):
        @app.callback(
            Output(f"{self._data_acquirer_instance.inner_loop_action.component_id}-readout-params-container", "children"),
            Input(self._data_acquirer_instance._get_id("readouts"), "value"),
            prevent_initial_call=True,
        )

        def _refresh_readout_params(selected_names):
            names = selected_names or []
            if not isinstance(names, list):
                names = [names]
            self._data_acquirer_instance.selected_readout_channels = [
                self._data_acquirer_instance.available_readout_channels[n]
                for n in names
                if n in self._data_acquirer_instance.available_readout_channels
            ]
            self._data_acquirer_instance.inner_loop_action.selected_readout_channels = self._data_acquirer_instance.selected_readout_channels
            return self._data_acquirer_instance.inner_loop_action.build_readout_controls()

    def _register_click_to_centre_callback(
            self,
            app: Dash,
            shared_viewer_graph_id: str,
            *,
            tabs_id: str,
    ) -> None:
        @app.callback(
            Output(
                self._get_id(self._DUMMY_OUTPUT_ACQUIRER_UPDATE_SUFFIX),
                "children",
                allow_duplicate=True,
            ),
            Output(
                self._get_id("set-dc-center-armed"),
                "data",
                allow_duplicate=True,
            ),
            Output(
                self._get_id("set-dc-center-button"),
                "children",
                allow_duplicate=True,
            ),
            Output(
                self._get_id("set-dc-center-button"),
                "color",
                allow_duplicate=True,
            ),
            Output(self._get_id(self._TOGGLE_ACQ_BUTTON_ID_SUFFIX), "children", allow_duplicate=True),
            Output(self._get_id(self._TOGGLE_ACQ_BUTTON_ID_SUFFIX), "color", allow_duplicate=True),
            Output(self._get_id(self._TOGGLE_ACQ_BUTTON_ID_SUFFIX), "disabled", allow_duplicate=True),
            Output(self._get_id(self._ACQUIRER_STATUS_INDICATOR_ID_SUFFIX), "children", allow_duplicate=True),
            Output(self._get_id(self._ACQUIRER_STATUS_INDICATOR_ID_SUFFIX), "color", allow_duplicate=True),
            Input(shared_viewer_graph_id, "clickData"),
            State(self._get_id("set-dc-center-armed"), "data"),
            State(tabs_id, "value"),
            prevent_initial_call=True,
        )

        def _centre_on_click(click_data, armed, active_tab_value):
            if not armed:
                return (
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                )
            if active_tab_value != "live-view-tab":
                return (
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                )
            if not click_data or "points" not in click_data:
                return (
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                )
            pt = click_data["points"][0]
            x_val = pt.get("x")
            y_val = pt.get("y")
            da = self._data_acquirer_instance
            voltage_control = getattr(da, "voltage_control_component", None)
            if voltage_control is None or voltage_control.dc_set is None:
                return (
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                )
            dc_set = voltage_control.dc_set

            with voltage_control.voltage_lock:
                current = dc_set.sync_from_parameter_cache()
                targets = {}
                if x_val is not None and da.x_axis_name in dc_set.valid_channel_names:
                    targets[da.x_axis_name] = (
                        current.get(da.x_axis_name, 0.0)
                        + float(x_val)
                    )
                if (
                    y_val is not None
                    and da.y_axis_name is not None
                    and da.y_axis_name in dc_set.valid_channel_names
                ):
                    targets[da.y_axis_name] = (
                        current.get(da.y_axis_name, 0.0)
                        + float(y_val)
                    )
                if targets:
                    # print("Current DC:", current)
                    # print("Clicked delta:", {"x": x_val, "y": y_val})
                    # print("Targets:", targets)
                    dc_set.set_voltages(targets, requery=False, resync=False)
                    # print("Updated DC:", dc_set._current_levels)
            da.start_acquisition()
            return (
                "",
                False,
                "Set DC point",
                "primary",
                "Stop",
                "danger",
                False,
                "RUNNING",
                "success",
            )

    def _register_gridlines_callback(
            self,
            app: Dash,
            shared_viewer_store_ids: Dict[str, Any]
        ) -> None:
        @app.callback(
            Output(shared_viewer_store_ids["layout_config_store"], "data", allow_duplicate = True),
            Input(self._get_id("center-marker-toggle"), "value"),
            Input(self._get_id("mouse-crosshairs-toggle"), "value"),
            Input(self._get_id("show-full-grid-toggle"), "value"),
            Input(self._get_id("grid_opacity"), "value"),
            State(shared_viewer_store_ids["layout_config_store"], "data"),
            prevent_initial_call=True,
        )

        def _toggle_gridlines(toggle_val, mouse_val, full_val, opacity, existing_layout):
            layout = existing_layout or {}
            show = "on" in (toggle_val or [])
            show_full = "on" in (full_val or [])
            show_mouse = "on" in (mouse_val or [])
            shapes = []
            alpha = float(opacity)/100
            n_subplots = max(1, len(self._data_acquirer_instance.selected_readout_channels))
            def axis_layout_name(axis: str, i: int) -> str:
                return axis if i == 1 else f"{axis}{i}"
            da = self._data_acquirer_instance
            is_1d = da.y_axis_name is None
            cache_key = (
                show,
                show_mouse,
                show_full,
                opacity,
                tuple(self._data_acquirer_instance.x_axis.sweep_values_with_offset),
                tuple(self._data_acquirer_instance.y_axis.sweep_values_with_offset),
                len(self._data_acquirer_instance.selected_readout_channels),
            )
            if cache_key == self._last_gridlines_state:
                return dash.no_update
            self._last_gridlines_state = cache_key
            if show or show_full:
                def ax_suffix(i):
                    return "" if i == 1 else str(i)
                xr = list(self._data_acquirer_instance.x_axis.sweep_values_with_offset)
                xs = np.linspace(xr[0], xr[-1], 15).tolist()
                if not is_1d:
                    yr = list(self._data_acquirer_instance.y_axis.sweep_values_with_offset)
                    ys = np.linspace(yr[0], yr[-1], 15).tolist()
                for i in range(1, n_subplots + 1):
                    sfx = ax_suffix(i)
                    xref = f"x{sfx}"
                    grid_color = f"rgba(0,0,0,{alpha})"
                    grid_color_1d = f"rgba(255,255,255,{alpha})"
                    if show:
                        if is_1d:
                            shapes.append({
                                "type": "line",
                                "xref": f"{xref} domain",
                                "yref": "paper",
                                "x0": 0.5,
                                "x1": 0.5,
                                "y0": 0,
                                "y1": 1,
                                "line": {"width": 5, "color": grid_color_1d},
                                "layer": "above",
                                "name": "crosshair-x",
                            })
                        else:
                            yref = f"y{sfx}"
                            shapes.append({
                                "type": "line",
                                "xref": f"{xref} domain",
                                "yref": f"{yref} domain",
                                "x0": 0.5,
                                "x1": 0.5,
                                "y0": 0,
                                "y1": 1,
                                "line": {"width": 5, "color": grid_color},
                                "layer": "above",
                                "name": "crosshair-x",
                            })
                            shapes.append({
                                "type": "line",
                                "xref": f"{xref} domain",
                                "yref": f"{yref} domain",
                                "x0": 0,
                                "x1": 1,
                                "y0": 0.5,
                                "y1": 0.5,
                                "line": {"width": 5, "color": grid_color},
                                "layer": "above",
                                "name": "crosshair-y",
                            })
                    if show_full:
                        if is_1d:
                            for xv in xs:
                                shapes.append({
                                    "type": "line",
                                    "xref": xref,
                                    "yref": "paper",
                                    "x0": xv, "x1": xv,
                                    "y0": 0, "y1": 1,
                                    "line": {"width": 2, "color": grid_color_1d},
                                    "layer": "above",
                                    "name": "grid-x",
                                })
                        else:
                            yref = f"y{sfx}"
                            for xv in xs:
                                shapes.append({
                                    "type": "line",
                                    "xref": xref, "yref": yref,
                                    "x0": xv, "x1": xv, "y0": min(ys), "y1": max(ys),
                                    "line": {"width": 2, "color": grid_color},
                                    "layer": "above",
                                    "name": "grid-x",
                                })
                            for yv in ys:
                                shapes.append({
                                    "type": "line",
                                    "xref": xref, "yref": yref,
                                    "x0": min(xs), "x1": max(xs), "y0": yv, "y1": yv,
                                    "line": {"width": 2, "color": grid_color},
                                    "layer": "above",
                                    "name": "grid-y",
                                })
            layout["shapes"] = shapes
            for i in range(1, n_subplots + 1):
                xaxis = axis_layout_name("xaxis", i)
                layout[xaxis] = {
                    **layout.get(xaxis, {}),
                    "showspikes": show_mouse,
                    "spikesnap": "cursor",
                    "spikemode": "across",
                    "spikethickness": 1,
                }
                if not is_1d:
                    yaxis = axis_layout_name("yaxis", i)
                    layout[yaxis] = {
                        **layout.get(yaxis, {}),
                        "showspikes": show_mouse,
                        "spikesnap": "cursor",
                        "spikemode": "across",
                        "spikethickness": 1,
                    }
            return layout

    def _supports_dynamic_gate_axis_refresh(self) -> bool:
        """Return True when the data acquirer exposes dynamic gate axes."""
        da = self._data_acquirer_instance
        return (
            hasattr(da, "gate_set")
            and hasattr(da, "ensure_axis")
            and hasattr(da, "_build_dropdown_options")
            and hasattr(da, "_display_x_sweep_axes")
            and hasattr(da, "_display_y_sweep_axes")
        )

    def _register_dynamic_gate_axis_refresh_callback(self, app: Dash) -> None:
        """Keep X/Y gate dropdowns synchronized with a mutable GateSet."""
        da = self._data_acquirer_instance

        @app.callback(
            Output(da._get_id("gate-select-x"), "options"),
            Output(da._get_id("gate-select-y"), "options"),
            Output(da._get_id("gate-select-x"), "value"),
            Output(da._get_id("gate-select-y"), "value"),
            Input(self._get_id(self._GATE_AXIS_REFRESH_INTERVAL_SUFFIX), "n_intervals"),
            prevent_initial_call=True,
        )
        def _refresh_gate_axes(_n_intervals):
            current_gate_names = tuple(da.gate_set.valid_channel_names)

            if current_gate_names == self._known_gate_names:
                raise dash.exceptions.PreventUpdate

            if hasattr(da, "mark_virtual_layer_changed"):
                da.mark_virtual_layer_changed(affects_config=False)

            da.ensure_axis()
            self._known_gate_names = current_gate_names

            return (
                da._build_dropdown_options(da._display_x_sweep_axes),
                da._build_dropdown_options(da._display_y_sweep_axes),
                da.x_axis_name,
                da.y_axis_name,
            )

    def _pause_for_axis_change(self) -> bool:
        """Pause acquisition before a scan change that requires explicit restart.

        Returns True when acquisition was running and was stopped by this call.
        This is used for coordinate-system changes and for span/points updates.
        Axis offset keeps its existing live-update behaviour.
        """
        da = self._data_acquirer_instance
        state = da.get_latest_data() or {}
        was_running = str(state.get("status", "")).lower() == "running"
        if was_running:
            logger.info(
                "Pausing acquisition before changing the X/Y scan coordinate system."
            )
            da.stop_acquisition()
        return was_running

    def _register_gate_selection_callback(
            self, app:Dash
    ) -> None:
        """Pause a running scan when the selected X or Y axis changes."""
        @app.callback(
            Output(self._get_id(self._DUMMY_OUTPUT_ACQUIRER_UPDATE_SUFFIX), "children", allow_duplicate=True),
            Output(self._get_id(self._ACQUIRER_CONTROLS_DIV_ID_SUFFIX), "children", allow_duplicate=True),
            Output(self._get_id(self._TOGGLE_ACQ_BUTTON_ID_SUFFIX), "children", allow_duplicate=True),
            Output(self._get_id(self._TOGGLE_ACQ_BUTTON_ID_SUFFIX), "color", allow_duplicate=True),
            Output(self._get_id(self._ACQUIRER_STATUS_INDICATOR_ID_SUFFIX), "children", allow_duplicate=True),
            Output(self._get_id(self._ACQUIRER_STATUS_INDICATOR_ID_SUFFIX), "color", allow_duplicate=True),
            Output(self._get_id("set-dc-center-button"), "disabled", allow_duplicate=True),
            Input(self._data_acquirer_instance._get_id("gate-select-x"), "value"),
            Input(self._data_acquirer_instance._get_id("gate-select-y"), "value"),
            prevent_initial_call = True
        )

        def on_gate_select(x_gate, y_gate):
            da = self._data_acquirer_instance

            # Dynamic control rebuilds can fire the dropdown callbacks even when
            # the effective selection did not change. Do not pause in that case.
            if x_gate == da.x_axis_name and y_gate == da.y_axis_name:
                raise dash.exceptions.PreventUpdate

            logger.info(f"New Gate Selected: x_axis {x_gate}, y_axis {y_gate}")
            was_running = self._pause_for_axis_change()

            params = {
                da.component_id: {
                    "gate-select-x": x_gate,
                    "gate-select-y": y_gate
                }
            }
            da.update_parameters(params)
            children = self._build_acquirer_controls()

            if was_running:
                return "", children, "Start", "success", "STOPPED", "secondary", True

            return (
                "",
                children,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

    def _register_mode_callback(
            self, app: Dash
    ) -> None:
        """Pause a running scan when the X or Y sweep mode changes."""
        @app.callback(
            Output(self._get_id(self._DUMMY_OUTPUT_ACQUIRER_UPDATE_SUFFIX), "children", allow_duplicate=True),
            Output(self._get_id(self._ACQUIRER_CONTROLS_DIV_ID_SUFFIX), "children", allow_duplicate=True),
            Output(self._get_id(self._TOGGLE_ACQ_BUTTON_ID_SUFFIX), "children", allow_duplicate=True),
            Output(self._get_id(self._TOGGLE_ACQ_BUTTON_ID_SUFFIX), "color", allow_duplicate=True),
            Output(self._get_id(self._ACQUIRER_STATUS_INDICATOR_ID_SUFFIX), "children", allow_duplicate=True),
            Output(self._get_id(self._ACQUIRER_STATUS_INDICATOR_ID_SUFFIX), "color", allow_duplicate=True),
            Output(self._get_id("set-dc-center-button"), "disabled", allow_duplicate=True),
            Input(self._data_acquirer_instance._get_id("x-mode"), "value"),
            Input(self._data_acquirer_instance._get_id("y-mode"), "value"),
            prevent_initial_call = True
        )

        def on_mode_select(x_mode, y_mode):
            da = self._data_acquirer_instance

            if x_mode == da.x_mode and y_mode == da.y_mode:
                raise dash.exceptions.PreventUpdate

            was_running = self._pause_for_axis_change()

            da.x_mode = x_mode
            da.y_mode = y_mode
            x_names = [ax.name for ax in da._display_x_sweep_axes]
            y_names = [ax.name for ax in da._display_y_sweep_axes]
            if da.x_axis_name not in x_names:
                da.x_axis_name = x_names[0] if x_names else None
            if da.y_axis_name not in y_names:
                da.y_axis_name = None
            params = {
                da.component_id: {
                    "x-mode": x_mode,
                    "y-mode": y_mode,
                    "gate-select-x": da.x_axis_name,
                    "gate-select-y": da.y_axis_name,
                }
            }
            da.update_parameters(params)
            children = self._build_acquirer_controls()

            if was_running:
                return "", children, "Start", "success", "STOPPED", "secondary", True

            return (
                "",
                children,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

    def _register_swap_xy_callback(self, app: Dash) -> None:
        """Swap the complete X/Y sweep selections and pause a running acquisition.

        Span and points are properties of the selected sweep-axis objects, so swapping
        the X/Y mode + axis selections also brings each axis' remembered span/points
        with it.
        """
        @app.callback(
            Output(self._get_id(self._ACQUIRER_CONTROLS_DIV_ID_SUFFIX), "children", allow_duplicate=True),
            Output(self._get_id(self._TOGGLE_ACQ_BUTTON_ID_SUFFIX), "children", allow_duplicate=True),
            Output(self._get_id(self._TOGGLE_ACQ_BUTTON_ID_SUFFIX), "color", allow_duplicate=True),
            Output(self._get_id(self._ACQUIRER_STATUS_INDICATOR_ID_SUFFIX), "children", allow_duplicate=True),
            Output(self._get_id(self._ACQUIRER_STATUS_INDICATOR_ID_SUFFIX), "color", allow_duplicate=True),
            Output(self._get_id("set-dc-center-button"), "disabled", allow_duplicate=True),
            Input(self._get_id("swap-xy-button"), "n_clicks"),
            prevent_initial_call=True,
        )
        def _swap_xy(n_clicks):
            if not n_clicks:
                raise dash.exceptions.PreventUpdate

            da = self._data_acquirer_instance
            old_x_mode = getattr(da, "x_mode", None)
            old_y_mode = getattr(da, "y_mode", None)
            old_x_name = getattr(da, "x_axis_name", None)
            old_y_name = getattr(da, "y_axis_name", None)

            # A 1D scan has no Y axis to swap with.
            if old_y_name is None:
                logger.info("Swap X/Y ignored because the current scan is 1D.")
                raise dash.exceptions.PreventUpdate

            was_running = self._pause_for_axis_change()

            params = {
                da.component_id: {
                    "x-mode": old_y_mode,
                    "y-mode": old_x_mode,
                    "gate-select-x": old_y_name,
                    "gate-select-y": old_x_name,
                }
            }
            da.update_parameters(params)

            if hasattr(da, "_scan_idx_cache"):
                da._scan_idx_cache = None

            new_children = self._build_acquirer_controls()

            if was_running:
                return (
                    new_children,
                    "Start",
                    "success",
                    "STOPPED",
                    "secondary",
                    True,
                )

            return (
                new_children,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

    def _register_pulse_offsets_callback(self, app: Dash) -> None:
        """Update fixed virtual-gate offsets and pause before recompiling."""
        gate_inputs = [
            Input(self._get_id(f"pulse-offset-gate-{slot}"), "value")
            for slot in range(self._NUM_PULSE_OFFSET_SLOTS)
        ]
        value_inputs = [
            Input(self._get_id(f"pulse-offset-value-{slot}"), "value")
            for slot in range(self._NUM_PULSE_OFFSET_SLOTS)
        ]
        disabled_outputs = [
            Output(self._get_id(f"pulse-offset-value-{slot}"), "disabled")
            for slot in range(self._NUM_PULSE_OFFSET_SLOTS)
        ]

        @app.callback(
            *disabled_outputs,
            Output(self._get_id(self._TOGGLE_ACQ_BUTTON_ID_SUFFIX), "children", allow_duplicate=True),
            Output(self._get_id(self._TOGGLE_ACQ_BUTTON_ID_SUFFIX), "color", allow_duplicate=True),
            Output(self._get_id(self._ACQUIRER_STATUS_INDICATOR_ID_SUFFIX), "children", allow_duplicate=True),
            Output(self._get_id(self._ACQUIRER_STATUS_INDICATOR_ID_SUFFIX), "color", allow_duplicate=True),
            Output(self._get_id("set-dc-center-button"), "disabled", allow_duplicate=True),
            *gate_inputs,
            *value_inputs,
            prevent_initial_call=True,
        )
        def _update_pulse_offsets(*args):
            n = self._NUM_PULSE_OFFSET_SLOTS
            gates = args[:n]
            values = args[n:2 * n]

            offsets = {}
            for gate_name, value in zip(gates, values):
                if not gate_name:
                    continue
                numeric_value = 0.0 if value in (None, "") else float(value)
                # If a gate is selected in more than one slot, contributions add.
                offsets[gate_name] = offsets.get(gate_name, 0.0) + numeric_value

            inner = getattr(self._data_acquirer_instance, "inner_loop_action", None)
            if inner is None or not hasattr(inner, "pulse_offsets"):
                logger.warning("Pulse-offset update ignored: inner loop has no pulse_offsets attribute.")
                raise dash.exceptions.PreventUpdate

            disabled = [not bool(gate_name) for gate_name in gates]
            current = dict(getattr(inner, "pulse_offsets", {}) or {})
            if current == offsets:
                return (*disabled, *(dash.no_update for _ in range(5)))

            was_running = self._pause_for_axis_change()
            inner.pulse_offsets = offsets

            da = self._data_acquirer_instance
            if hasattr(da, "_compilation_flags"):
                da._compilation_flags |= ModifiedFlags.PROGRAM_MODIFIED
            if hasattr(da, "_clear_queue"):
                da._clear_queue()

            logger.info("Updated pulse offsets: %s", offsets)

            if was_running:
                return (*disabled, "Start", "success", "STOPPED", "secondary", True)
            return (*disabled, *(dash.no_update for _ in range(5)))

    def _register_fast_axis_direction_visibility_callback(
        self, app: Dash, tabs_id
    ) -> None:
        """Show direction only for scan modes that expose fast_axis_direction."""
        if tabs_id is None:
            return

        @app.callback(
            Output(self._get_id("fast-axis-direction-row"), "style"),
            Input(tabs_id, "value"),
            prevent_initial_call=False,
        )
        def _sync_fast_axis_direction_visibility(_active_tab):
            if self._scan_supports_fast_axis_direction():
                return {}
            return {"display": "none"}

    def _register_fast_axis_direction_callback(self, app: Dash) -> None:
        """Toggle the X fast-axis acquisition direction for raster-like scans.

        The direction is applied to every registered scan mode that exposes a
        ``fast_axis_direction`` attribute so Raster and Switch Raster share the
        same user setting. Changing direction pauses a running acquisition and
        marks the QUA program for recompilation.
        """
        @app.callback(
            Output(self._get_id("fast-axis-direction-button"), "children"),
            Output(self._get_id("fast-axis-direction-button"), "color"),
            Output(self._get_id(self._TOGGLE_ACQ_BUTTON_ID_SUFFIX), "children", allow_duplicate=True),
            Output(self._get_id(self._TOGGLE_ACQ_BUTTON_ID_SUFFIX), "color", allow_duplicate=True),
            Output(self._get_id(self._ACQUIRER_STATUS_INDICATOR_ID_SUFFIX), "children", allow_duplicate=True),
            Output(self._get_id(self._ACQUIRER_STATUS_INDICATOR_ID_SUFFIX), "color", allow_duplicate=True),
            Output(self._get_id("set-dc-center-button"), "disabled", allow_duplicate=True),
            Input(self._get_id("fast-axis-direction-button"), "n_clicks"),
            prevent_initial_call=True,
        )
        def _toggle_fast_axis_direction(n_clicks):
            if not n_clicks:
                raise dash.exceptions.PreventUpdate

            da = self._data_acquirer_instance
            current_mode = getattr(da, "scan_mode", None)
            if current_mode is None or not hasattr(current_mode, "fast_axis_direction"):
                logger.info(
                    "Fast-axis direction toggle ignored: current scan mode does not "
                    "support a directional fast axis."
                )
                raise dash.exceptions.PreventUpdate

            current = str(getattr(current_mode, "fast_axis_direction", "forward")).lower()
            new_direction = "reverse" if current == "forward" else "forward"

            was_running = self._pause_for_axis_change()

            # Keep Raster and Switch Raster consistent when the user later
            # changes scan mode in Settings. Other scan modes are untouched.
            for mode in getattr(da, "scan_modes", {}).values():
                if hasattr(mode, "fast_axis_direction"):
                    mode.fast_axis_direction = new_direction

            # Direction changes both acquisition order and sample-to-pixel
            # indexing, so invalidate the scan-index cache and recompile QUA.
            if hasattr(da, "_scan_idx_cache"):
                da._scan_idx_cache = None
            if hasattr(da, "_compilation_flags"):
                da._compilation_flags |= ModifiedFlags.PROGRAM_MODIFIED

            if new_direction == "reverse":
                label, color = "← Reverse", "warning"
            else:
                label, color = "→ Forward", "secondary"

            if was_running:
                return label, color, "Start", "success", "STOPPED", "secondary", True

            return (
                label,
                color,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

    def _register_set_dc_button_callback(self, app: Dash) -> None:
        @app.callback(
            Output(self._get_id("set-dc-center-armed"), "data"),
            Output(self._get_id("set-dc-center-button"), "children"),
            Output(self._get_id("set-dc-center-button"), "color"),
            Output(self._get_id(self._TOGGLE_ACQ_BUTTON_ID_SUFFIX), "children", allow_duplicate=True),
            Output(self._get_id(self._TOGGLE_ACQ_BUTTON_ID_SUFFIX), "color", allow_duplicate=True),
            Output(self._get_id(self._TOGGLE_ACQ_BUTTON_ID_SUFFIX), "disabled", allow_duplicate=True),
            Output(self._get_id(self._ACQUIRER_STATUS_INDICATOR_ID_SUFFIX), "children", allow_duplicate=True),
            Output(self._get_id(self._ACQUIRER_STATUS_INDICATOR_ID_SUFFIX), "color", allow_duplicate=True),
            Input(self._get_id("set-dc-center-button"), "n_clicks"),
            State(self._get_id("set-dc-center-armed"), "data"),
            prevent_initial_call=True,
        )
        def _toggle_set_dc(_n_clicks, armed):
            da = self._data_acquirer_instance

            if armed:
                da.start_acquisition()
                return (
                    False,
                    "Set DC point",
                    "primary",
                    "Stop",
                    "danger",
                    False,
                    "RUNNING",
                    "success",
                )

            state = da.get_latest_data()
            status = state.get("status", "unknown").upper()
            if status != "RUNNING":
                raise dash.exceptions.PreventUpdate

            da.stop_acquisition()
            return (
                True,
                "Cancel",
                "danger",
                "Paused",
                "secondary",
                True,
                "Pick DC point",
                "secondary",
            )

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
            Output(self._get_id("set-dc-center-button"), "disabled"),
            Output(self._get_id(self._TOGGLE_ACQ_BUTTON_ID_SUFFIX), "disabled"),
            Input(self._get_id(self._TOGGLE_ACQ_BUTTON_ID_SUFFIX), "n_clicks"),
            Input(main_status_alert_id, "children"),
            prevent_initial_call=True,
        )

        def handle_acquisition_control_and_status_update(
            _toggle_clicks: Any, _status_alert_trigger: Any
        ) -> tuple[str, str, str, str, bool, bool]:
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
            if (
                not is_button_click
                and _status_alert_trigger
                and "STOPPED" in str(_status_alert_trigger).upper()
            ):
                return (
                    "Start",
                    "success",
                    "STOPPED",
                    "secondary",
                    True,
                    False,
                )
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
                    button_text, button_color = "Start", "success"
                    status_text, status_color = "STOPPED", "secondary"
                else:  # Was STOPPED, ERROR, or UNKNOWN
                    logger.info(
                        f"Attempting to start acquisition for "
                        f"'{self._data_acquirer_instance.component_id}'"
                    )
                    if current_status == "ERROR":
                        self._data_acquirer_instance.reset()
                    self._data_acquirer_instance.start_acquisition()
                    button_text, button_color = "Stop", "danger"
                    status_text, status_color = "RUNNING", "success"
            else:
                if current_status != "STOPPED":
                    logger.debug(
                        f"LiveViewTab: Status update triggered externally. "
                        f"Acquirer status: {current_status}"
                    )
                if current_status == "RUNNING":
                    button_text, button_color = "Stop", "danger"
                    status_text, status_color = "RUNNING", "success"
                elif current_status == "STOPPED":
                    button_text, button_color = "Start", "success"
                    status_text, status_color = "STOPPED", "secondary"
                elif current_status == "ERROR":
                    button_text, button_color = (
                        "Start",
                        "success",
                    )
                    status_text = (
                        f"ERROR{(': ' + str(error_details)) if error_details else ''}"
                    )
                    status_text = status_text[:100]
                    status_color = "danger"
                else:  # Unknown or other states
                    button_text, button_color = "Start", "warning"
                    status_text, status_color = current_status, "warning"
            set_dc_disabled = status_text != "RUNNING"
            return button_text, button_color, status_text, status_color, set_dc_disabled, False

    def _register_parameter_update_callback(self, app: Dash) -> None:
        """Update acquirer parameters and pause for scan span/point changes.

        Span and point-count changes alter the physical scan trajectory / frame
        shape. If acquisition is running, stop it before applying either change
        and require the user to press Start explicitly. Other numeric parameters
        (for example axis offset) retain their existing live-update behaviour.
        """
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
            Output(self._get_id(self._ACQUIRER_CONTROLS_DIV_ID_SUFFIX), "children", allow_duplicate=True),
            Output(self._get_id(self._TOGGLE_ACQ_BUTTON_ID_SUFFIX), "children", allow_duplicate=True),
            Output(self._get_id(self._TOGGLE_ACQ_BUTTON_ID_SUFFIX), "color", allow_duplicate=True),
            Output(self._get_id(self._ACQUIRER_STATUS_INDICATOR_ID_SUFFIX), "children", allow_duplicate=True),
            Output(self._get_id(self._ACQUIRER_STATUS_INDICATOR_ID_SUFFIX), "color", allow_duplicate=True),
            Output(self._get_id("set-dc-center-button"), "disabled", allow_duplicate=True),
            Input({"type": "number-input", "index": ALL}, "value"),
            Input({"type": "toggle", "index": ALL}, "value"),
            *static_inputs,
            State({"type": "number-input", "index": ALL}, "id"),
            State({"type": "toggle", "index": ALL}, "id"),
            *static_states,
            prevent_initial_call=True,
        )

        def handle_hybrid_parameter_update(*args):
            """Handle both pattern-matched axis parameters and static component parameters."""
            num_pattern_inputs = 2
            num_static_inputs = len(static_inputs)
            num_pattern_states = 2
            num_static_states = len(static_states)
            pattern_values = args[:num_pattern_inputs]
            static_values = args[num_pattern_inputs:num_pattern_inputs + num_static_inputs]
            pattern_states = args[num_pattern_inputs + num_static_inputs:
                                num_pattern_inputs + num_static_inputs + num_pattern_states]
            static_states_data = args[num_pattern_inputs + num_static_inputs + num_pattern_states:]
            parameters_to_update = {}

            # Only span / points should force a running acquisition to stop.
            # ``ctx.triggered_id`` is the pattern-matched ID of the control that
            # actually changed, e.g. {"type": "number-input",
            # "index": "<axis-component-id>::span"}.
            triggered_param_name = None
            triggered_id = ctx.triggered_id
            if isinstance(triggered_id, dict):
                triggered_index = triggered_id.get("index", "")
                _, triggered_param_name = self._parse_param_id(triggered_index)

            pause_for_scan_geometry_change = triggered_param_name in {"span", "points"}

            toggle_values = pattern_values[1]
            toggle_ids = pattern_states[1]
            rebuild = False
            if toggle_values and toggle_ids:
                for value, id_dict in zip(toggle_values, toggle_ids):
                    if not isinstance(id_dict, dict):
                        continue
                    comp_id, param = self._parse_param_id(id_dict.get("index", ""))
                    if comp_id and param:
                        parameters_to_update.setdefault(comp_id, {})[param] = value
                        if param.endswith("dbm-toggle"):
                            rebuild = True
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
            was_running = False
            if parameters_to_update:
                if pause_for_scan_geometry_change:
                    was_running = self._pause_for_axis_change()
                self._data_acquirer_instance.update_parameters(parameters_to_update)

            new_children = self._build_acquirer_controls() if rebuild else dash.no_update

            if was_running:
                logger.info(
                    "Acquisition paused after changing scan %s; press Start to resume.",
                    triggered_param_name,
                )
                return (
                    dash.no_update,
                    new_children,
                    "Start",
                    "success",
                    "STOPPED",
                    "secondary",
                    True,
                )

            return (
                dash.no_update,
                new_children,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

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

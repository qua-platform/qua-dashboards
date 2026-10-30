from dash.development.base_component import Component
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence
from dash import Dash, Input, Output, dcc, html, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import logging
import threading
import xarray as xr

from qua_dashboards.core import BaseComponent
from qua_dashboards.video_mode import data_registry
from qua_dashboards.video_mode.utils.plot_utils import AVAILABLE_COLORS
from qua_dashboards.post_processing.processed_viewer import ProcessedViewer

from qua_dashboards.video_mode.data_acquirers import BaseDataAcquirer

logger = logging.getLogger(__name__)

__all__ = ["PostProcessor"]

PostProcessingFunction = Callable[[xr.DataArray], xr.DataArray]


def x_derivative(da: xr.DataArray) -> xr.DataArray:
    """Derivative along the x axis, which is the last dimension in the plotting convention."""
    return da.differentiate(da.dims[-1])


def y_derivative(da: xr.DataArray) -> xr.DataArray:
    """Derivative along the y axis, which is the second-to-last dimension in the plotting convention."""
    if da.ndim < 2:
        raise ValueError("y_derivative requires 2D data.")
    return da.differentiate(da.dims[-2])


DEFAULT_POST_PROCESSING_FUNCTIONS: Dict[str, PostProcessingFunction] = {
    "Raw_data": lambda da: da,
    "x_derivative": x_derivative,
    "y_derivative": y_derivative,
}


class PostProcessor(BaseComponent):
    """
    A post processing component, with its own shared viewer. It takes 2D maps from the data acquirer
    and plots them after applying the selected post processing functions. It is independent of the
    VideoModeComponent.
    """

    # Store holding {"key": processed_data_key, "version": ...}, updated on each new processed frame
    VIEWER_DATA_STORE_SUFFIX = "viewer-data-store"
    VIEWER_COLOR_SCALE_STORE_SUFFIX = "viewer-color-scale-store"

    _MAIN_STATUS_ALERT_ID_SUFFIX = "main-status-alert"
    _PLOTTING_MODE_SWITCH_ID_SUFFIX = "plotting-mode-switch"
    _FUNCTION_SELECTOR_ID_SUFFIX = "function-selector"
    _UPDATE_INTERVAL_ID_SUFFIX = "update-interval"
    _IMPORT_FRAME_BUTTON_ID_SUFFIX = "import-frame-button"
    _IMPORT_FRAME_CONTAINER_ID_SUFFIX = "import-frame-container"

    def __init__(
        self,
        data_acquirer: BaseDataAcquirer,
        post_processing_functions: Optional[Dict[str, PostProcessingFunction]] = None,
        selected_functions: Optional[Sequence[str]] = None,
        plotting_mode: Literal["latest_frame", "live"] = "latest_frame",
        color_scheme: str = "plasma",
        update_interval_ms: int = 500,
        layout_columns: int = 12,
        layout_rows: int = 10,
        component_id: str = "post-processor",
        **kwargs: Any,
    ):
        """
        Args:
            data_acquirer: The data acquirer whose frames are post processed.
            post_processing_functions: Post processing functions by name. Each takes and returns
                an xarray.DataArray. Defaults to DEFAULT_POST_PROCESSING_FUNCTIONS.
            selected_functions: Names of the initially selected functions.
            plotting_mode: "live" processes frames as they are acquired, "latest_frame" processes
                the latest frame only.
            color_scheme: The plotly color scheme of the processing viewer.
            update_interval_ms: How often to check the data acquirer for a new frame in live mode.
            layout_columns: Number of dashboard grid columns, matching the VideoModeComponent default.
            layout_rows: Number of dashboard grid rows, matching the VideoModeComponent default.
            component_id: Unique ID for this component instance.
        """
        super().__init__(component_id=component_id, **kwargs)
        self.layout_columns = layout_columns
        self.layout_rows = layout_rows
        self.data_acquirer = data_acquirer

        self.post_processing_functions: Dict[str, PostProcessingFunction] = (
            dict(post_processing_functions)
            if post_processing_functions
            else dict(DEFAULT_POST_PROCESSING_FUNCTIONS)
        )
        unknown = [n for n in (selected_functions or []) if n not in self.post_processing_functions]
        if unknown:
            raise ValueError(
                f"Unknown post processing functions: {unknown}. "
                f"Available: {list(self.post_processing_functions)}"
            )
        self.selected_functions: List[str] = list(
            selected_functions or list(self.post_processing_functions)[:1]
        )

        if plotting_mode not in ("latest_frame", "live"):
            raise ValueError(f"plotting_mode must be 'latest_frame' or 'live'. Received: {plotting_mode}")
        self.plotting_mode = plotting_mode

        if color_scheme not in AVAILABLE_COLORS:
            raise ValueError(f"Color scheme not in list of valid schemes. Received: {color_scheme}. Available: {AVAILABLE_COLORS}")
        self.color_scheme = color_scheme
        self.update_interval_ms = update_interval_ms

        # Registry key under which processed results are stored for the processing viewer
        self.processed_data_key = f"{self.component_id}-processed-data"

        self.processing_viewer: ProcessedViewer = ProcessedViewer(
            component_id=f"{self.component_id}-processed-viewer",
            data_key=self.processed_data_key,
        )

        self._last_processed_seq: Optional[int] = None
        self._imported_frame: Optional[xr.DataArray] = None
        self._processing_lock = threading.Lock()

    def _get_store_id(self, suffix: str) -> Dict[str, str]:
        """Generates a namespaced Dash component ID for internal stores."""
        return self._get_id(suffix)

    def process(self, data: xr.DataArray) -> xr.DataArray:
        """Applies each selected post processing function to a frame from the data acquirer.

        Multiple results are concatenated along a "readout" dimension labelled by function name,
        which the plot utilities render as one subplot per label. Frames that already have a
        "readout" dimension are processed per readout.
        """
        if not self.selected_functions:
            return data

        if "readout" in data.dims:
            frames = {
                str(r): data.isel(readout=i).drop_vars("readout", errors="ignore")
                for i, r in enumerate(data.coords["readout"].values)
            }
        else:
            frames = {None: data}

        panels, labels = [], []
        for name in self.selected_functions:
            function = self.post_processing_functions[name]
            for readout, frame in frames.items():
                panels.append(function(frame))
                labels.append(name if readout is None else f"{name} ({readout})")

        if len(panels) == 1:
            return panels[0]
        return xr.concat(panels, dim="readout").assign_coords(readout=labels)

    def _get_latest_frame(self) -> tuple[Optional[xr.DataArray], Optional[int]]:
        """Returns the latest frame from the data acquirer and its sequence number."""
        response = self.data_acquirer.get_latest_data()
        if response.get("error") is not None:
            raise RuntimeError(f"Data acquirer error: {response['error']}")
        # Acquirers returning xarray data do not always include "seq", so fall back to the counter.
        seq = response.get("seq", getattr(self.data_acquirer, "_latest_seq", None))
        return response.get("data"), seq

    def _process_and_store(self, data: xr.DataArray) -> Dict[str, Any]:
        """Processes a frame, stores it in the data registry and returns the viewer data reference."""
        processed = self.process(data)
        version = data_registry.set_data(
            self.processed_data_key,
            {"base_image_data": processed, "annotations": {"points": [], "lines": []}},
        )
        return {"key": self.processed_data_key, "version": version}

    def _error_alert(self, e: Exception) -> dbc.Alert:
        logger.error(f"PostProcessor '{self.component_id}' processing error: {e}", exc_info=True)
        return dbc.Alert(f"Post processing error: {str(e)[:200]}", color="danger", dismissable=True)

    @staticmethod
    def _import_button_style(is_live: bool) -> Dict[str, str]:
        return {"display": "none"} if is_live else {}

    def get_layout(
        self
    ) -> Component:
        """Generates the Dash layout for the PostProcessor."""
        logger.debug(f"Generating layout for PostProcessor '{self.component_id}'")

        left_panel = dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Post Processing", className="card-title text-light"),
                    dbc.Row(
                        [
                            dbc.Col(dbc.Label("Latest frame", className="mb-0"), width="auto"),
                            dbc.Col(
                                dbc.Switch(
                                    id=self._get_id(self._PLOTTING_MODE_SWITCH_ID_SUFFIX),
                                    value=self.plotting_mode == "live",
                                    className="mb-0",
                                ),
                                width="auto",
                            ),
                            dbc.Col(dbc.Label("Live", className="mb-0"), width="auto"),
                        ],
                        className="mb-2 g-2 align-items-center",
                    ),
                    html.Div(
                        dbc.Button(
                            "Import live frame",
                            id=self._get_id(self._IMPORT_FRAME_BUTTON_ID_SUFFIX),
                            color="secondary",
                            size="sm",
                        ),
                        id=self._get_id(self._IMPORT_FRAME_CONTAINER_ID_SUFFIX),
                        style=self._import_button_style(self.plotting_mode == "live"),
                        className="mb-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Label("Post-Processing Functions", width="auto", className="col-form-label"),
                            dbc.Col(
                                dcc.Dropdown(
                                    id=self._get_id(self._FUNCTION_SELECTOR_ID_SUFFIX),
                                    options=[
                                        {"label": name, "value": name}
                                        for name in self.post_processing_functions
                                    ],
                                    value=self.selected_functions,
                                    multi=True,
                                    clearable=False,
                                    style={"color": "black"},
                                ),
                                width=True,
                            ),
                        ],
                        className="mb-2 align-items-center",
                    ),
                ]
            ),
            color="dark",
            inverse=True,
            className="tab-card-dark",
        )
        right_panel = self.processing_viewer.get_layout()

        shared_stores_for_layout = [
            dcc.Store(
                id=self._get_store_id(self.VIEWER_DATA_STORE_SUFFIX),
                data={"key": self.processed_data_key, "version": None},
            ),
            dcc.Store(
                id=self._get_store_id(self.VIEWER_COLOR_SCALE_STORE_SUFFIX),
                data=self.color_scheme,
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
                            left_panel,
                            width=12,
                            lg=4,
                            className="mb-3 mb-lg-0",
                            style={
                                "maxHeight": "calc(100vh - 70px)",
                                "overflowY": "auto",
                            },
                        ),
                        dbc.Col(
                            right_panel,
                            width=12,
                            lg=8,
                            style={
                                # Explicit height, since the short left panel would otherwise
                                # collapse the graph, which only fills its parent.
                                "height": "calc(100vh - 70px)",
                                "overflowY": "hidden",  # Graph usually handles its own scroll
                            },
                        ),
                    ],
                    className="g-3",  # Gutters
                ),
                dcc.Interval(
                    id=self._get_id(self._UPDATE_INTERVAL_ID_SUFFIX),
                    interval=self.update_interval_ms,
                    n_intervals=0,
                ),
                *shared_stores_for_layout,
            ],
            style={"paddingTop": "10px"},
        )
        return main_layout

    def register_callbacks(self, app: Dash, **kwargs: Any) -> None:
        """Registers all necessary Dash callbacks for this component."""
        logger.info(
            f"Registering callbacks for PostProcessor '{self.component_id}'"
        )

        self.processing_viewer.register_callbacks(
            app,
            viewer_data_store_id=self._get_store_id(self.VIEWER_DATA_STORE_SUFFIX),
            color_scale_store_id=self._get_store_id(self.VIEWER_COLOR_SCALE_STORE_SUFFIX),
        )

        @app.callback(
            Output(self._get_store_id(self.VIEWER_DATA_STORE_SUFFIX), "data", allow_duplicate=True),
            Output(self._get_id(self._MAIN_STATUS_ALERT_ID_SUFFIX), "children", allow_duplicate=True),
            Input(self._get_id(self._UPDATE_INTERVAL_ID_SUFFIX), "n_intervals"),
            prevent_initial_call=True,
        )
        def _process_live_frame(_n_intervals):
            if self.plotting_mode != "live":
                raise PreventUpdate
            # Skip this tick if the previous frame is still being processed.
            if not self._processing_lock.acquire(blocking=False):
                raise PreventUpdate
            try:
                data, seq = self._get_latest_frame()
                if data is None or (seq is not None and seq == self._last_processed_seq):
                    raise PreventUpdate
                viewer_data_ref = self._process_and_store(data)
                self._last_processed_seq = seq
                return viewer_data_ref, ""
            except PreventUpdate:
                raise
            except Exception as e:
                return no_update, self._error_alert(e)
            finally:
                self._processing_lock.release()

        @app.callback(
            Output(self._get_store_id(self.VIEWER_DATA_STORE_SUFFIX), "data", allow_duplicate=True),
            Output(self._get_id(self._MAIN_STATUS_ALERT_ID_SUFFIX), "children", allow_duplicate=True),
            Input(self._get_id(self._IMPORT_FRAME_BUTTON_ID_SUFFIX), "n_clicks"),
            prevent_initial_call=True,
        )
        def _import_live_frame(_n_clicks):
            with self._processing_lock:
                try:
                    data, _ = self._get_latest_frame()
                    if data is None:
                        return no_update, dbc.Alert(
                            "No frame available yet. Start the acquisition first.",
                            color="warning",
                            dismissable=True,
                        )
                    self._imported_frame = data
                    return self._process_and_store(data), ""
                except Exception as e:
                    return no_update, self._error_alert(e)

        @app.callback(
            Output(self._get_id(self._IMPORT_FRAME_CONTAINER_ID_SUFFIX), "style"),
            Output(self._get_id(self._MAIN_STATUS_ALERT_ID_SUFFIX), "children", allow_duplicate=True),
            Input(self._get_id(self._PLOTTING_MODE_SWITCH_ID_SUFFIX), "value"),
            prevent_initial_call=True,
        )
        def _set_plotting_mode(is_live):
            self.plotting_mode = "live" if is_live else "latest_frame"
            logger.info(f"PostProcessor '{self.component_id}' plotting mode: {self.plotting_mode}")
            return self._import_button_style(is_live), ""

        @app.callback(
            Output(self._get_store_id(self.VIEWER_DATA_STORE_SUFFIX), "data", allow_duplicate=True),
            Output(self._get_id(self._MAIN_STATUS_ALERT_ID_SUFFIX), "children", allow_duplicate=True),
            Input(self._get_id(self._FUNCTION_SELECTOR_ID_SUFFIX), "value"),
            prevent_initial_call=True,
        )
        def _set_selected_functions(names):
            self.selected_functions = list(names or [])
            logger.info(f"PostProcessor '{self.component_id}' functions: {self.selected_functions}")
            if self.plotting_mode == "live":
                # Forces the next interval tick to reprocess the current frame, even if paused.
                self._last_processed_seq = None
                return no_update, ""
            if self._imported_frame is None:
                return no_update, ""
            with self._processing_lock:
                try:
                    return self._process_and_store(self._imported_frame), ""
                except Exception as e:
                    return no_update, self._error_alert(e)

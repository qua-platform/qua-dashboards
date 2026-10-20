import logging
from typing import Any, Dict, List, Optional
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

from qua_dashboards.core import BaseComponent
from qua_dashboards.video_mode import data_registry
from qua_dashboards.video_mode.utils.plot_utils import *


logger = logging.getLogger(__name__)

__all__ = ["SharedViewerComponent"]


class SharedViewerComponent(BaseComponent):
    """
    A component responsible for displaying data visualizations.

    It listens to a primary data store for references to data in a
    server-side registry (which can be live data or a static compound object
    containing a base image and annotations) and a layout configuration store.
    """

    _MAIN_GRAPH_ID_SUFFIX = "main-graph"
    _VIEWER_CONTAINER_ID_SUFFIX = "viewer-container"

    def __init__(self, component_id: str, **kwargs: Any) -> None:
        """Initializes the SharedViewerComponent.

        Args:
            component_id: A unique string identifier for this component instance.
            **kwargs: Additional keyword arguments passed to BaseComponent.
        """
        super().__init__(component_id=component_id, **kwargs)
        self._current_figure: go.Figure = (
            self._get_default_figure()
        )  # Start with an empty dark figure
        self._last_layout_update: Optional[Dict] = None
        self._last_layout_shapes: List[Dict] = []
        logger.info(f"SharedViewerComponent '{self.component_id}' initialized.")

    def _get_default_figure(self) -> go.Figure:
        """
        Returns a default empty Plotly figure with the dark template.
        """
        return go.Figure().update_layout(template="plotly_dark")

    @staticmethod
    def _viewer_grid_shape(data_object: Optional[dict]) -> tuple[int, int]:
        """Return (rows, cols) for the current readout grid."""
        base = None
        if isinstance(data_object, dict):
            base = data_object.get("base_image_data")
            if base is None:
                base = data_object.get("data")

        try:
            if base is not None and "readout" in base.dims:
                n = int(base.sizes["readout"])
            else:
                n = 1
        except Exception:
            n = 1

        if n <= 1:
            return 1, 1
        if n == 2:
            return 1, 2
        return (n + 1) // 2, 2

    @classmethod
    def _viewer_container_style(cls, data_object: Optional[dict] = None) -> Dict[str, str]:
        """Fill the space provided by VideoModeComponent."""
        return {
            "width": "100%",
            "height": "100%",
            "minWidth": "0",
            "minHeight": "0",
            "flex": "1 1 auto",
        }

    def get_layout(self) -> html.Div:
        return html.Div(
            id=self._get_id(self._VIEWER_CONTAINER_ID_SUFFIX),
            style=self._viewer_container_style(),
            children=[
                dcc.Graph(
                    id=self._get_id(self._MAIN_GRAPH_ID_SUFFIX),
                    figure=self._current_figure,
                    style={"height": "100%", "width": "100%"},
                    responsive=True,
                    config={
                        "scrollZoom": True,
                        "displaylogo": False,
                        "displayModeBar": False,
                        "doubleClick": False,
                    },
                )
            ],
        )
    @staticmethod
    def _keep_square_axes(fig):
        layout = fig.to_dict().get("layout", {})

        if "yaxis2" in layout:
            return fig

        x_range = None
        y_range = None

        for trace in fig.data:
            if getattr(trace, "type", None) != "heatmap":
                continue

            x_vals = getattr(trace, "x", None)
            y_vals = getattr(trace, "y", None)

            if x_vals is None or y_vals is None or len(x_vals) == 0 or len(y_vals) == 0:
                break

            x_range = [float(min(x_vals)), float(max(x_vals))]
            y_range = [float(min(y_vals)), float(max(y_vals))]
            break

        # Line-profile figures do not contain a heatmap. In that case, leave
        # Plotly autorange untouched; forcing autorange=False without an explicit
        # range produces the oversized/default axis range seen in profile mode.
        if x_range is None or y_range is None:
            return fig

        fig.update_xaxes(
            range=x_range,
            constrain="domain",
            autorange=False,
        )
        fig.update_yaxes(
            range=y_range,
            constrain="domain",
            autorange=False,
        )

        fig.layout.yaxis.scaleanchor = None

        return fig

    def get_graph_id(self) -> Dict[str, str]:
        """
        Returns the Dash ID dictionary for the main graph component.

        This ID is needed by other components (e.g., AnnotationTabController)
        to register callbacks for graph interactions.

        Returns:
            A dictionary suitable for use as a Dash component ID.
        """
        return self._get_id(self._MAIN_GRAPH_ID_SUFFIX)

    def _create_figure_from_live_data(self, data_object: dict) -> go.Figure:
        """
        Creates a Plotly figure from live data (expected to be xr.DataArray).

        Args:
            data_object: The data object fetched from the registry.

        Returns:
            A Plotly Figure object.
        """
        try:
            fig = build_live_figure(data_object)
            fig = self._keep_square_axes(fig)
        except Exception as e:
            logger.error(f"SharedViewer ({self.component_id}): live->fig error: {e}", exc_info=True)
            fig = empty_dark()
        self._current_figure = fig
        return fig


    def _create_figure_from_static_data(self, static_data_object: dict, viewer_ui_state_input: dict) -> go.Figure:
        """
        Creates a Plotly figure from a static data compound object.

        The compound object is expected to contain 'base_image_data'
        and 'annotations'.

        Args:
            static_data_object: The compound static data object from the registry.

        Returns:
            A Plotly Figure object with base image and annotations.
        """
        try:
            fig = build_static_figure(static_data_object, viewer_ui_state_input)
            fig = self._keep_square_axes(fig)
        except Exception as e:
            logger.error(f"SharedViewer ({self.component_id}): static->fig error: {e}", exc_info=True)
            fig = empty_dark()
        self._current_figure = fig
        return fig


    def register_callbacks(
        self,
        app: Dash,
        viewer_data_store_id: Dict[str, str],
        viewer_ui_state_store_id: Dict[str, Any],
        layout_config_store_id: Dict[str, str],
    ) -> None:
        """Registers Dash callbacks for the SharedViewerComponent.

        Args:
            app: The main Dash application instance.
            viewer_data_store_id: The ID of the store containing the reference
                to the primary data (key and version).
            layout_config_store_id: The ID of the store containing layout updates.
        """
        logger.debug(
            f"Registering callbacks for SharedViewerComponent '{self.component_id}'"
        )

        @app.callback(
            Output(self._get_id(self._MAIN_GRAPH_ID_SUFFIX), "figure"),
            Output(self._get_id(self._VIEWER_CONTAINER_ID_SUFFIX), "style"),
            Input(viewer_data_store_id, "data"),
            Input(viewer_ui_state_store_id, "data"),
            Input(layout_config_store_id, "data"),
            State(self._get_id(self._MAIN_GRAPH_ID_SUFFIX), "figure"),
            prevent_initial_call=True,  # Prevent update on initial app load if stores are empty
        )
        def update_shared_viewer_graph(
            viewer_data_ref: Optional[Dict[str, Any]],
            viewer_ui_state_input: Optional[Dict[str, Any]],
            layout_updates_input: Optional[Dict[str, Any]],
            current_fig_state_dict: Optional[Dict[str, Any]],
        ) -> tuple[go.Figure, Dict[str, str]]:
            fig_to_display = self._get_default_figure()  # Default to empty dark figure
            viewer_style = self._viewer_container_style()

            if viewer_data_ref is None or not isinstance(viewer_data_ref, dict):
                logger.debug(
                    f"SharedViewer ({self.component_id}): viewer_data_ref is None or not a dict. "
                    "No primary data to display."
                )
                # Apply layout updates even to an empty figure if provided
                if isinstance(layout_updates_input, dict):
                    fig_to_display.update_layout(layout_updates_input)
                self._current_figure = fig_to_display  # Store the figure
                return fig_to_display, viewer_style

            data_key = viewer_data_ref.get("key")
            # version = viewer_data_ref.get("version") # Version can be used for debugging

            if not data_key:
                logger.warning(
                    f"SharedViewer ({self.component_id}): viewer_data_ref missing 'key'."
                )
            else:
                data_object = data_registry.get_data(data_key)
                if data_object is not None:
                    viewer_style = self._viewer_container_style(data_object)
                if data_object is None:
                    logger.warning(
                        f"SharedViewer ({self.component_id}): Data not found in registry "
                        f"for key '{data_key}'. Displaying empty graph."
                    )
                elif data_key == data_registry.LIVE_DATA_KEY:
                    # logger.debug(
                    #     f"SharedViewer ({self.component_id}): Processing live data "
                    #     f"for key '{data_key}'."
                    # )
                    fig_to_display = self._create_figure_from_live_data(data_object)
                elif data_key == data_registry.STATIC_DATA_KEY:
                    logger.debug(
                        f"SharedViewer ({self.component_id}): Processing static data "
                        f"for key '{data_key}'."
                    )
                    fig_to_display = self._create_figure_from_static_data(data_object,viewer_ui_state_input)
                    fig_to_display.update_layout(showlegend=False)
                else:
                    logger.warning(
                        f"SharedViewer ({self.component_id}): Unrecognized data key "
                        f"'{data_key}'. Displaying empty graph."
                    )

            # Apply layout updates from the dedicated store
            if isinstance(layout_updates_input, dict):
                if "shapes" in layout_updates_input:
                    self._last_layout_shapes = layout_updates_input["shapes"]
                non_shape_updates = {k: v for k, v in layout_updates_input.items() if k != "shapes"}
                if non_shape_updates:
                    logger.debug(
                        f"SharedViewer ({self.component_id}): Applying layout updates: "
                        f"{layout_updates_input}"
                    )
                    fig_to_display.update_layout(non_shape_updates)
                # if layout_updates_input != self._last_layout_update:
                #     fig_to_display.update_layout(layout_updates_input)
                #     self._last_layout_update = layout_updates_input
            if self._last_layout_shapes:
                fig_to_display.update_layout(shapes=self._last_layout_shapes)

            # Commented out because it causes annotation points to become barely visible
            # Preserve zoom/pan state (uirevision)
            # If layout_updates_input itself contains 'uirevision', that will take precedence.
            # Otherwise, we try to preserve the existing uirevision.
            # if not (
            #     isinstance(layout_updates_input, dict)
            #     and "uirevision" in layout_updates_input
            # ):
            #     previous_uirevision = True  # Default to True to preserve state
            #     if current_fig_state_dict and isinstance(
            #         current_fig_state_dict.get("layout"), dict
            #     ):
            #         previous_uirevision = current_fig_state_dict["layout"].get(
            #             "uirevision", True
            #         )
            #     fig_to_display.layout.uirevision = previous_uirevision
            self._current_figure = fig_to_display  # Store the figure
            return fig_to_display, viewer_style

    def get_figures(self) -> List[go.Figure]:
        """
        Returns a list containing the current figure managed by this component.

        This figure is the last one that was processed and output by the
        component's update callback.

        Returns:
            A list containing a single plotly.graph_objects.Figure.
        """
        return [self._current_figure]
import logging
from typing import Any, Dict, Optional

import plotly.graph_objects as go
from dash import Dash, Input, Output

from qua_dashboards.video_mode import data_registry
from qua_dashboards.video_mode.shared_viewer_component import SharedViewerComponent

logger = logging.getLogger(__name__)

__all__ = ["ProcessedViewer"]

class ProcessedViewer(SharedViewerComponent):
    """
    A viewer for post processed data. It reuses the SharedViewerComponent layout and figure
    building, but plots the data stored in the data registry under its own key instead of
    the live or static video mode data.
    """

    def __init__(
        self,
        component_id: str,
        data_key: str,
        **kwargs: Any,
    ):
        """
        Args:
            component_id: A unique string identifier for this component instance.
            data_key: The data registry key holding the processed data to plot, as a dict
                with a "base_image_data" xarray.DataArray.
            **kwargs: Additional keyword arguments passed to SharedViewerComponent.
        """
        super().__init__(component_id=component_id, **kwargs)
        self.data_key = data_key

    def register_callbacks(
        self,
        app: Dash,
        viewer_data_store_id: Dict[str, str],
        color_scale_store_id: Dict[str, str],
        **kwargs: Any,
    ) -> None:
        """Registers the callback redrawing the graph whenever new processed data is stored.

        Args:
            app: The main Dash application instance.
            viewer_data_store_id: The ID of the store holding {"key": data_key, "version": ...},
                updated whenever new processed data is stored in the registry.
            color_scale_store_id: The ID of the store containing the color scale name.
        """

        @app.callback(
            Output(self.get_graph_id(), "figure"),
            Input(viewer_data_store_id, "data"),
            Input(color_scale_store_id, "data"),
            prevent_initial_call=True,
        )
        def update_processed_viewer_graph(
            viewer_data_ref: Optional[Dict[str, Any]],
            color_scale: Optional[str],
        ) -> go.Figure:
            data_object = data_registry.get_data(self.data_key)
            if data_object is None:
                fig = self._get_default_figure()
            else:
                fig = self._create_figure_from_live_data(data_object)

            if color_scale:
                fig.update_traces(
                    colorscale=color_scale,
                    selector=lambda t: t.type == "heatmap"
                    and t.name != "_click_capture_1d",
                )
            self._current_figure = fig
            return fig

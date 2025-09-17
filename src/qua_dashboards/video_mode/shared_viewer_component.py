import logging
from typing import Any, Dict, List, Optional
import numpy as np
import plotly.graph_objects as go
import xarray as xr
from dash import Dash, Input, Output, State, dcc, html

from qua_dashboards.core import BaseComponent
from qua_dashboards.video_mode.utils.dash_utils import xarray_to_plotly
from qua_dashboards.video_mode import data_registry

from qua_dashboards.video_mode.utils.annotation_utils import (
    generate_annotation_traces,
)


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
        logger.info(f"SharedViewerComponent '{self.component_id}' initialized.")

    def _get_default_figure(self) -> go.Figure:
        """
        Returns a default empty Plotly figure with the dark template.
        """
        return go.Figure().update_layout(template="plotly_dark")

    def get_layout(self) -> html.Div:
        """Generates the Dash layout for the SharedViewerComponent.

        Returns:
            An html.Div component containing the dcc.Graph.
        """
        return html.Div(
            style={"height": "100%", "width": "100%"},
            children=[
                dcc.Graph(
                    id=self._get_id(self._MAIN_GRAPH_ID_SUFFIX),
                    figure=self._current_figure,
                    style={"height": "100%", "width": "100%"},
                    config={"scrollZoom": True, "displaylogo": False},
                )
            ],
        )

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
        if not isinstance(data_object, dict):
            logger.warning(
                f"SharedViewer ({self.component_id}): Live data object "
                f"is not a dict (type: {type(data_object)}). "
                "Cannot auto-plot."
            )
            self._current_figure = self._get_default_figure()
            return self._current_figure
        if "base_image_data" not in data_object:
            logger.warning(
                f"SharedViewer ({self.component_id}): Live data object "
                f"missing 'base_image_data' (type: {type(data_object)}). "
                "Cannot auto-plot."
            )
            self._current_figure = self._get_default_figure()
            return self._current_figure
        base_image_data = data_object.get("base_image_data")
        if not isinstance(base_image_data, xr.DataArray):
            logger.warning(
                f"SharedViewer ({self.component_id}): Live data object "
                f"base_image_data is not an xr.DataArray (type: {type(base_image_data)}). "
                "Cannot auto-plot."
            )
            self._current_figure = self._get_default_figure()
            return self._current_figure

        try:
            if base_image_data.ndim == 2 and (1 in base_image_data.shape):
                vals = np.asarray(base_image_data.values).ravel()
                dims = base_image_data.dims
                x_dim = dims[1] if base_image_data.shape[0] == 1 else dims[0]
                coord = base_image_data.coords[x_dim]
                x = np.asarray(coord.values)
                x_label = coord.attrs.get("label") or str(x_dim)
                if coord.attrs.get("units"):
                    x_label = f"{x_label} [{coord.attrs['units']}]"
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=vals, mode="lines"))
                fig.update_layout(template="plotly_dark",
                                  xaxis_title=x_label,
                                  yaxis_title="Value")
                self._current_figure = fig
                return fig
            self._current_figure = xarray_to_plotly(base_image_data)
        except Exception as e:
            logger.error(
                f"SharedViewer ({self.component_id}): Error converting "
                f"live xr.DataArray to Plotly: {e}",
                exc_info=True,
            )
            self._current_figure = self._get_default_figure()
        return self._current_figure

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
        fig = self._get_default_figure()
        profile = static_data_object.get("profile_plot")
        if isinstance(profile, dict):
            s = profile.get("s", [])
            vals = profile.get("vals", [])
            name = profile.get("name", "Line Profile")
            y_label = profile.get("y_label", "Value")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s, y=vals, mode="lines", name=name))
            fig.update_layout(
                template="plotly_dark",
                margin=dict(l=40, r=10, t=10, b=40),
                xaxis_title="Arbitrary Units",
                yaxis_title=y_label,
                showlegend=False,
            )
            self._current_figure = fig
            return fig
        
        if not isinstance(static_data_object, dict):
            logger.warning(
                f"SharedViewer ({self.component_id}): Static data object is not a dict "
                f"(type: {type(static_data_object)})."
            )
            return fig

        base_image_data = static_data_object.get("base_image_data")
        annotations_data = static_data_object.get("annotations")

        if isinstance(base_image_data, xr.DataArray):
            try:
                if base_image_data.ndim == 2 and (1 in base_image_data.shape):
                    vals = np.asarray(base_image_data.values).ravel()
                    dims = base_image_data.dims
                    x_dim = dims[1] if base_image_data.shape[0] == 1 else dims[0]
                    coord = base_image_data.coords[x_dim]
                    x = np.asarray(coord.values)
                    x_label = coord.attrs.get("label") or str(x_dim)
                    if coord.attrs.get("units"):
                        x_label = f"{x_label} [{coord.attrs['units']}]"
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x, y=vals, mode="lines"))
                    fig.update_layout(template="plotly_dark",
                                    xaxis_title=x_label,
                                    yaxis_title="Value")
                    self._current_figure = fig
                    return fig
                fig = xarray_to_plotly(base_image_data)
            except Exception as e:
                logger.error(
                    f"SharedViewer ({self.component_id}): Error converting static "
                    f"base_image_data (xr.DataArray) to Plotly: {e}",
                    exc_info=True,
                )
        elif base_image_data is not None:  # Data present but not xr.DataArray
            logger.warning(
                f"SharedViewer ({self.component_id}): Static base_image_data "
                f"is not an xr.DataArray (type: {type(base_image_data)}). "
                "Displaying empty base."
            )

        if isinstance(annotations_data, dict):
            overlay_traces = generate_annotation_traces(annotations_data, viewer_ui_state_input)
            for trace_dict in overlay_traces:
                if isinstance(trace_dict, dict):
                    fig.add_trace(go.Scatter(**trace_dict))
                else:
                    logger.warning(
                        f"SharedViewer ({self.component_id}): Invalid overlay trace "
                        f"format: {type(trace_dict)}. Expected dict."
                    )
        elif annotations_data is not None:
            logger.warning(
                f"SharedViewer ({self.component_id}): 'annotations' in static data "
                f"is not a dict (type: {type(annotations_data)})."
            )

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
        ) -> go.Figure:
            fig_to_display = self._get_default_figure()  # Default to empty dark figure

            if viewer_data_ref is None or not isinstance(viewer_data_ref, dict):
                logger.debug(
                    f"SharedViewer ({self.component_id}): viewer_data_ref is None or not a dict. "
                    "No primary data to display."
                )
                # Apply layout updates even to an empty figure if provided
                if isinstance(layout_updates_input, dict):
                    fig_to_display.update_layout(layout_updates_input)
                self._current_figure = fig_to_display  # Store the figure
                return fig_to_display

            data_key = viewer_data_ref.get("key")
            # version = viewer_data_ref.get("version") # Version can be used for debugging

            if not data_key:
                logger.warning(
                    f"SharedViewer ({self.component_id}): viewer_data_ref missing 'key'."
                )
            else:
                data_object = data_registry.get_data(data_key)
                if data_object is None:
                    logger.warning(
                        f"SharedViewer ({self.component_id}): Data not found in registry "
                        f"for key '{data_key}'. Displaying empty graph."
                    )
                elif data_key == data_registry.LIVE_DATA_KEY:
                    logger.debug(
                        f"SharedViewer ({self.component_id}): Processing live data "
                        f"for key '{data_key}'."
                    )
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
                logger.debug(
                    f"SharedViewer ({self.component_id}): Applying layout updates: "
                    f"{layout_updates_input}"
                )
                fig_to_display.update_layout(layout_updates_input)

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
            return fig_to_display

    def get_figures(self) -> List[go.Figure]:
        """
        Returns a list containing the current figure managed by this component.

        This figure is the last one that was processed and output by the
        component's update callback.

        Returns:
            A list containing a single plotly.graph_objects.Figure.
        """
        return [self._current_figure]

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import plotly.graph_objects as go
import xarray as xr
from dash import Dash, Input, Output, State, dcc, html

from qua_dashboards.core import BaseComponent
from qua_dashboards.video_mode.utils.dash_utils import xarray_to_plotly
from qua_dashboards.video_mode import data_registry
from plotly.subplots import make_subplots
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
    
    def _make_readout_subplots_with_profile(self, da: xr.DataArray, profile: dict) -> go.Figure:
        assert isinstance(da, xr.DataArray) and da.ndim == 3 and "readout" in da.dims
        y_name, x_name = [d for d in da.dims if d != "readout"]
        rd_vals = list(da.coords["readout"].values)
        labels = [str(v) for v in rd_vals]
        n = len(labels)
        if "subplot_col" in profile:
            target_idx = max(0, int(profile["subplot_col"]) - 1)
        else:
            target_label = str(profile.get("readout", labels[0]))
            try:
                target_idx = labels.index(target_label)
            except ValueError:
                target_idx = 0

        fig = make_subplots(rows=1, cols=n, subplot_titles=labels,
                            horizontal_spacing=min(0.06, 1.0/(max(n-1,1)) - 1e-6))
        yc, xc = da.coords[y_name], da.coords[x_name]
        y_lab = yc.attrs.get("long_name", y_name)
        y_unit = yc.attrs.get("units", "")
        x_lab = xc.attrs.get("long_name", x_name)
        x_unit = xc.attrs.get("units", "")
        y_title = f"{y_lab} ({y_unit})" if y_unit else y_lab
        x_title = f"{x_lab} ({x_unit})" if x_unit else x_lab
        for i in range(n):
            if i == target_idx:
                s = np.asarray(profile.get("s", []))
                vals = np.asarray(profile.get("vals", []))
                fig.add_trace(
                    go.Scatter(x=s, y=vals, mode="lines", name=f"profile_{labels[i]}", showlegend=False),
                    row=1, col=i+1
                )
                fig.update_xaxes(title_text=profile.get("x_label", "Distance"), row=1, col=i+1)
                fig.update_yaxes(title_text=profile.get("y_label", "Value"),    row=1, col=i+1)
            else:
                sub = da.isel(readout=i).reset_coords("readout", drop=True)
                small = xarray_to_plotly(sub)
                if small.data:
                    tr = small.data[0]
                    tr.update(showscale=(i == n - 1))
                    try:
                        z = np.asarray(tr["z"])
                        rows, cols = z.shape
                        tr["customdata"] = [[i + 1] * cols for _ in range(rows)]
                    except Exception:
                        pass
                    fig.add_trace(tr, row=1, col=i+1)
                    fig.update_xaxes(title_text=x_title, row=1, col=i+1)
                if i == 0:
                    fig.update_yaxes(title_text=y_title, row=1, col=1)

        fig.update_layout(template="plotly_dark", showlegend=False, margin=dict(l=60, r=30, t=40, b=40))
        return fig

    def _pick_readout_dim(self, da: xr.DataArray) -> tuple[str, str]:
        dims = list(da.dims)
        if "readout" in dims:
            rd = "readout"
            return rd, [d for d in dims if d != rd][0]

        def _non_numeric(d: str) -> bool:
            c = da.coords.get(d)
            if c is None:
                return False
            return np.asarray(c.values).dtype.kind not in ("i", "u", "f")

        rd = min(dims, key=lambda d: (da.sizes[d], 0 if _non_numeric(d) else 1))
        return rd, [d for d in dims if d != rd][0]

    def _axis_vals_and_label(self, da: xr.DataArray, dim: str) -> tuple[np.ndarray, str]:
        c = da.coords.get(dim)
        x = np.asarray(c.values) if c is not None else np.arange(da.sizes[dim])
        label = (c.attrs.get("label", dim) if c is not None else dim)
        u = (c.attrs.get("units") if c is not None else None)
        if u:
            label = f"{label} [{u}]"
        return x, label

    def _safe_hspace(self, n: int, default: float = 0.06) -> float:
        return min(default, 1.0 / (n - 1) - 1e-6) if n > 1 else default

    def _is_line(self, da: xr.DataArray) -> bool:
        return (
            isinstance(da, xr.DataArray)
            and "readout" not in da.dims
            and (da.ndim == 1 or (da.ndim == 2 and 1 in da.shape))
        )

    def _make_readout_line_subplots(self, da: xr.DataArray) -> go.Figure:
        assert isinstance(da, xr.DataArray) and da.ndim == 2
        rd_dim, x_dim = self._pick_readout_dim(da)
        labels = [str(v) for v in (np.asarray(da.coords[rd_dim].values)
                                if rd_dim in da.coords else range(da.sizes[rd_dim]))]
        n = len(labels)
        x_vals, x_label = self._axis_vals_and_label(da, x_dim)

        if n > 8:
            fig = go.Figure()
            for i, lab in enumerate(labels):
                y = np.asarray(da.isel({rd_dim: i}).values).ravel()
                fig.add_trace(go.Scatter(x=x_vals, y=y, mode="lines", name=lab))
            fig.update_layout(template="plotly_dark", xaxis_title=x_label, yaxis_title="Value", showlegend=True)
            return fig

        if n <= 1:
            y = np.asarray(da.isel({rd_dim: 0}).values).ravel()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_vals, y=y, mode="lines", name=labels[0]))
            fig.update_layout(template="plotly_dark", xaxis_title=x_label, yaxis_title="Value", showlegend=False)
            return fig

        fig = make_subplots(rows=1, cols=n, subplot_titles=labels, horizontal_spacing=self._safe_hspace(n))
        for i, lab in enumerate(labels):
            y = np.asarray(da.isel({rd_dim: i}).values).ravel()
            fig.add_trace(go.Scatter(x=x_vals, y=y, mode="lines", name=lab, showlegend=False), row=1, col=i + 1)
            fig.update_xaxes(title_text=x_label, row=1, col=i + 1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_layout(template="plotly_dark", showlegend=False, margin=dict(l=60, r=30, t=40, b=40))
        return fig

    def _make_readout_subplots(self, da: xr.DataArray) -> go.Figure:
        if not (isinstance(da, xr.DataArray) and da.ndim == 3 and "readout" in da.dims):
            return xarray_to_plotly(da)

        y_name, x_name = [d for d in da.dims if d != "readout"]
        labels = [str(v) for v in da.coords["readout"].values]
        n = len(labels)

        fig = make_subplots(rows=1, cols=n, subplot_titles=labels, horizontal_spacing=self._safe_hspace(n))
        yc, xc = da.coords[y_name], da.coords[x_name]
        y_title = f"{yc.attrs.get('long_name', y_name)} ({yc.attrs.get('units','')})".strip(" ()")
        x_title = f"{xc.attrs.get('long_name', x_name)} ({xc.attrs.get('units','')})".strip(" ()")

        for i in range(n):
            sub = da.isel(readout=i).reset_coords("readout", drop=True)
            small = xarray_to_plotly(sub)
            if not small.data:
                continue
            tr = small.data[0]
            tr.update(showscale=(i == n - 1))
            try:
                z = np.asarray(tr["z"])
                rows, cols = z.shape
                tr["customdata"] = [[i + 1] * cols for _ in range(rows)]
            except Exception:
                pass
            fig.add_trace(tr, row=1, col=i + 1)
        for col in range(1, n + 1):
            fig.update_xaxes(title_text=x_title, row=1, col=col)
        fig.update_yaxes(title_text=y_title, row=1, col=1)
        fig.update_layout(template="plotly_dark", showlegend=False, margin=dict(l=60, r=30, t=40, b=40))
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
        if not isinstance(data_object, dict):
            logger.warning(
                f"SharedViewer ({self.component_id}): Live data object "
                f"is not a dict (type: {type(data_object)}). "
                "Cannot auto-plot."
            )
            self._current_figure = self._get_default_figure()
            return self._current_figure

        if "base_image_data" not in data_object and "data" not in data_object:
            logger.warning(
                f"SharedViewer ({self.component_id}): Live data object missing 'base_image_data'."
            )
            self._current_figure = self._get_default_figure()
            return self._current_figure

        bid = data_object.get("base_image_data")
        base_image_data = bid if bid is not None else data_object.get("data")

        if not isinstance(base_image_data, xr.DataArray):
            logger.warning(
                f"SharedViewer ({self.component_id}): base_image_data is not xr.DataArray "
                f"(type: {type(base_image_data)})."
            )
            self._current_figure = self._get_default_figure()
            return self._current_figure

        try:
            if self._is_line(base_image_data):
                da = base_image_data.squeeze(drop=True)
                if da.ndim == 2:
                    non_single_dims = [d for d in da.dims if da.sizes[d] != 1]
                    if len(non_single_dims) == 1:
                        keep = non_single_dims[0]
                        sel = {d: 0 for d in da.dims if d != keep}
                        da = da.isel(**sel)
                    else:
                        da = xr.DataArray(da.values.ravel(), dims=("idx",), coords={"idx": np.arange(da.size)})

                x_dim = da.dims[0]
                coord = da.coords.get(x_dim, None)
                x = np.asarray(coord.values) if coord is not None else np.arange(da.size)
                x_label = (coord.attrs.get("label") if coord is not None else str(x_dim))
                units = (coord.attrs.get("units") if coord is not None else None)
                if units:
                    x_label = f"{x_label} [{units}]"

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=np.asarray(da.values).ravel(), mode="lines"))
                fig.update_layout(template="plotly_dark", xaxis_title=x_label, yaxis_title="Value")
                self._current_figure = fig
                return fig

            if base_image_data.ndim == 2 and ("readout" in base_image_data.dims):
                fig = self._make_readout_line_subplots(base_image_data)
                self._current_figure = fig
                return fig

            if base_image_data.ndim == 3 and "readout" in base_image_data.dims:
                fig = self._make_readout_subplots(base_image_data)
                self._current_figure = fig
                return fig

            self._current_figure = xarray_to_plotly(base_image_data)
            return self._current_figure

        except Exception as e:
            logger.error(
                f"SharedViewer ({self.component_id}): Error converting live xr.DataArray to Plotly: {e}",
                exc_info=True,
            )
            self._current_figure = self._get_default_figure()
        return self._current_figure


    def _create_figure_from_static_data(self, static_data_object: dict, viewer_ui_state_input: dict) -> go.Figure:
        logger.info("Rendering static figure, annotations present? %s",
            isinstance(static_data_object.get("annotations"), dict))
        ann = static_data_object.get("annotations") or {}
        logger.info("Ann counts: points=%d lines=%d", len(ann.get("points", [])), len(ann.get("lines", [])))

        fig = self._get_default_figure()

        profile = static_data_object.get("profile_plot") if isinstance(static_data_object, dict) else None
        base_image_data = static_data_object.get("base_image_data") if isinstance(static_data_object, dict) else None

        if isinstance(profile, dict) and isinstance(base_image_data, xr.DataArray) and base_image_data.ndim == 3 and "readout" in base_image_data.dims:
            try:
                fig = self._make_readout_subplots_with_profile(base_image_data, profile)
                self._current_figure = fig
                return fig
            except Exception as e:
                logger.warning(f"SharedViewer ({self.component_id}): failed to build mixed subplots with profile: {e}")

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
                xaxis_title=profile.get("x_label", "Distance"),
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
        n_cols = 1
        if isinstance(base_image_data, xr.DataArray) and base_image_data.ndim == 3 and "readout" in base_image_data.dims:
            try:
                n_cols = int(len(base_image_data.coords["readout"]))
            except Exception:
                n_cols = 2

        if isinstance(base_image_data, xr.DataArray):
            try:
                if self._is_line(base_image_data):
                    da = base_image_data.squeeze()
                    x_dim = da.dims[0]
                    coord = da.coords[x_dim] if x_dim in da.coords else None
                    x = (np.asarray(coord.values) if coord is not None else np.arange(da.size))
                    x_label = (coord.attrs.get("label") if coord is not None else str(x_dim))
                    units = (coord.attrs.get("units") if coord is not None else None)
                    if units:
                        x_label = f"{x_label} [{units}]"
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x, y=np.asarray(da.values).ravel(), mode="lines"))
                    fig.update_layout(template="plotly_dark", xaxis_title=x_label, yaxis_title="Value")
                elif base_image_data.ndim == 3 and "readout" in base_image_data.dims:
                    fig = self._make_readout_subplots(base_image_data)
                else:
                    fig = xarray_to_plotly(base_image_data)
            except Exception as e:
                logger.error(
                    f"SharedViewer ({self.component_id}): Error converting static "
                    f"base_image_data (xr.DataArray) to Plotly: {e}",
                    exc_info=True,
                )
                fig = self._get_default_figure()
        elif base_image_data is not None:
            logger.warning(
                f"SharedViewer ({self.component_id}): Static base_image_data is not xr.DataArray "
                f"(type: {type(base_image_data)}). Displaying empty base."
            )

        if isinstance(annotations_data, dict):
            show_point_labels = False
            try:
                show_point_labels = bool(
                    viewer_ui_state_input
                    and "show_labels" in viewer_ui_state_input
                    and "points" in (viewer_ui_state_input.get("show_labels") or [])
                )
            except Exception:
                show_point_labels = False

            points = annotations_data.get("points", []) or []
            lines  = annotations_data.get("lines", []) or []

            def _col_of_point(pid: str, default: int = 1) -> int:
                for p in points:
                    if p.get("id") == pid:
                        return int(p.get("subplot_col", default) or default)
                return default

            # lines
            for ln in lines:
                try:
                    lid = str(ln.get("id"))
                    sid = str(ln.get("start_point_id"))
                    eid = str(ln.get("end_point_id"))
                    col = int(ln.get("subplot_col", _col_of_point(sid)))

                    p1 = next((p for p in points if p.get("id") == sid), None)
                    p2 = next((p for p in points if p.get("id") == eid), None)
                    if not p1 or not p2:
                        continue

                    x = [float(p1["x"]), float(p2["x"])]
                    y = [float(p1["y"]), float(p2["y"])]

                    line_trace = go.Scatter(
                        x=x, y=y,
                        mode="lines",
                        name="annotations_lines",
                        showlegend=False,
                        hoverinfo="skip",
                        line=dict(width=2, color="white"),
                        customdata=[[lid]] * 2,
                    )
                    if n_cols == 1:
                        fig.add_trace(line_trace)
                    else:
                        fig.add_trace(line_trace, row=1, col=col)
                except Exception as _e:
                    logger.warning(f"SharedViewer: failed to draw line {ln}: {_e}")

            by_col: Dict[int, List[Dict[str, Any]]] = {}
            for p in points:
                try:
                    col = int(p.get("subplot_col", 1))
                except Exception:
                    col = 1
                by_col.setdefault(col, []).append(p)

            for col, pts in by_col.items():
                if not pts:
                    continue
                xs = [float(p["x"]) for p in pts]
                ys = [float(p["y"]) for p in pts]
                ids = [str(p["id"]) for p in pts]

                pt_trace = go.Scatter(
                    x=xs, y=ys,
                    mode="markers",
                    name="annotations_points",
                    showlegend=False,
                    hoverinfo="skip",
                    marker=dict(size=10, symbol="circle",color="white", line=dict(width=2)),
                    customdata=[[pid] for pid in ids],
                )
                if n_cols == 1:
                    fig.add_trace(pt_trace)
                else:
                    fig.add_trace(pt_trace, row=1, col=col)

                if show_point_labels:
                    text_trace = go.Scatter(
                        x=xs, y=ys,
                        mode="text",
                        text=ids,
                        textposition="top center",
                        name="annotations_point_labels",
                        showlegend=False,
                        hoverinfo="skip",
                        customdata=[[pid] for pid in ids],
                    )
                    if n_cols == 1:
                        fig.add_trace(text_trace)
                    else:
                        fig.add_trace(text_trace, row=1, col=col)

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
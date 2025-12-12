import xarray as xr
import numpy as np
from typing import Type, TypeVar, Optional
from dash.development.base_component import Component
from dash import html, dcc, Input, Output, State, MATCH, ALL
from plotly import graph_objects as go
from qua_dashboards.logging_config import logger
from qua_dashboards.data_dashboard.plotting import plot_xarray, update_xarray_plot
from .base_data_component import BaseDataComponent

GRAPH_STYLE = {"aspect-ratio": "1 / 1", "max-width": "400px"}
T = TypeVar("T", bound=Component)


class DataArrayComponent(BaseDataComponent):
    """
    A component for displaying xarray DataArrays.

    Supports:
      - 1D arrays: Displayed as line plots using `plot_xarray`.
      - 2D arrays: Displayed as heatmaps or images using
        `update_xarray_plot`.
      - ND arrays (ndim ≥ 3): The last two dimensions are used for the plot;
        every outer dimension gets its own slider control.
    """

    _data_nd = {}
    _nd_callback_registered = False

    @classmethod
    def can_handle(cls, value: any) -> bool:
        """Return True if value is an xarray DataArray with ndim ≥ 1."""
        return isinstance(value, xr.DataArray) and value.ndim >= 1

    @classmethod
    def create_component(
        cls,
        label: str,
        value: xr.DataArray,
        existing_component: Optional[Component] = None,
        root_component_class: Type[T] = html.Div,
    ) -> T:
        """
        Create or update a collapsible component for an xarray DataArray.

        Intelligently decides between full rebuild and data-only update based on
        array structure compatibility. Preserves user interaction state (zoom,
        scroll position) when structure remains the same.
        """
        if not cls._validate_existing_component(
            existing_component, value, root_component_class
        ):
            logger.info(f"Creating new data array component {label}")
            root_component = cls.create_collapsible_root_component(
                label, root_component_class, "xarray_data_array_component"
            )
            label_comp, collapse_comp = root_component.children
            collapse_comp.children = cls._build_ui_for_data_array(label, value)
        else:
            root_component = existing_component
            label_comp, collapse_comp = root_component.children

            # Check if we can do a data-only update (preserve state)
            old_array = cls._data_nd.get(label)
            logger.info(f"[{label}] old_array in cache: {old_array is not None}, ndim: {value.ndim}")

            if old_array is not None:
                structure_compatible = cls._structure_compatible(old_array, value)
                logger.info(
                    f"[{label}] Structure check: old shape={old_array.shape}, "
                    f"new shape={value.shape}, compatible={structure_compatible}"
                )

            if (
                old_array is not None
                and value.ndim > 2
                and cls._structure_compatible(old_array, value)
            ):
                logger.info(f"DATA-ONLY UPDATE: {label} (preserving zoom/scroll state)")
                cls._update_nd_ui_data_only(label, value, collapse_comp.children)
            else:
                # Structure changed or first time: full rebuild
                if old_array is not None:
                    logger.info(f"FULL REBUILD: {label} (structure changed)")
                else:
                    logger.info(f"FULL REBUILD: {label} (no old_array in cache)")
                collapse_comp.children = cls._build_ui_for_data_array(label, value)

        if value.ndim > 2:
            cls._data_nd[label] = value

        return root_component

    @classmethod
    def _build_ui_for_data_array(cls, label: str, value: xr.DataArray) -> list:
        """
        Build UI components for the given DataArray based on its dimensions.

        Returns a list of Dash components.
        """
        if value.ndim == 1:
            fig = plot_xarray(value)
            graph = dcc.Graph(figure=fig, style=GRAPH_STYLE)
            return [graph]
        elif value.ndim == 2:
            fig = go.Figure(layout=dict(margin=dict(l=20, r=20, t=20, b=20)))
            graph = dcc.Graph(figure=fig, style=GRAPH_STYLE)
            update_xarray_plot(fig, value)
            return [graph]
        elif value.ndim > 2:
            return cls._prepare_nd_ui(label, value)
        else:
            logger.error("Unsupported array dimensions: %s", value.ndim)
            return []

    @classmethod
    def _prepare_nd_ui(cls, label: str, value: xr.DataArray) -> list:
        """
        Prepare UI components for ND arrays by creating a graph and sliders.

        Returns a list containing a graph and a control container.
        """
        outer_dims = value.dims[:-2]
        graph_width = GRAPH_STYLE.get("max-width", "400px")
        slider_controls = cls._build_slider_controls(label, value, outer_dims)
        control_container = html.Div(
            children=slider_controls,
            id={"type": "data-array-control-container", "index": label},
            style={"width": graph_width, "marginTop": "10px"},
        )
        slice_dict = {dim: 0 for dim in outer_dims}
        sliced_array = value.isel(**slice_dict)
        fig = go.Figure(layout=dict(margin=dict(l=20, r=20, t=20, b=20)))
        update_xarray_plot(fig, sliced_array)
        graph = dcc.Graph(
            id={"type": "data-array-graph", "index": label},
            figure=fig,
            style=GRAPH_STYLE,
        )
        return [graph, control_container]

    @classmethod
    def _build_slider_controls(
        cls, label: str, value: xr.DataArray, dims: tuple
    ) -> list:
        """
        Build a list of slider control rows for the given dimensions.

        Each slider row corresponds to one outer dimension.
        """
        slider_controls = []
        for dim in dims:
            size = value.sizes[dim]
            if dim in value.coords:
                coord = value.coords[dim].values
                if len(coord) != size:
                    coord = list(range(size))
            else:
                coord = list(range(size))
            marks = {i: str(coord[i]) for i in range(size)}
            slider = dcc.Slider(
                id={"type": "data-array-slicer", "index": label, "dim": dim},
                min=0,
                max=size - 1,
                step=1,
                value=0,
                marks=marks,
                tooltip={"placement": "bottom", "always_visible": False},
                updatemode="drag",
            )
            control_row = html.Div(
                [
                    html.Div(
                        str(dim),
                        className="control-label",
                        style={"textAlign": "right", "paddingRight": "10px"},
                    ),
                    html.Div(
                        slider, className="control-inner", style={"width": "100%"}
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "max-content 1fr",
                    "alignItems": "center",
                    "columnGap": "10px",
                    "width": "100%",
                    "marginBottom": "5px",
                },
            )
            slider_controls.append(control_row)
        return slider_controls

    @staticmethod
    def _validate_existing_component(
        component: Component, value: xr.DataArray, root_component_class: Type[T]
    ) -> bool:
        """
        Validate that an existing component is of the correct type and carries
        the proper custom data-class attribute.
        """
        if not isinstance(component, root_component_class):
            return False
        if getattr(component, "data-class", None) != "xarray_data_array_component":
            return False
        return True

    @staticmethod
    def _structure_compatible(
        old_array: xr.DataArray, new_array: xr.DataArray
    ) -> bool:
        """
        Check if two arrays have compatible structure for data-only update.

        Returns True if they have the same dimensions, shape, and coordinate labels.
        A data-only update can be performed when structure is the same but values differ.
        """
        # Check dimensions match
        if old_array.dims != new_array.dims:
            return False

        # Check shape matches
        if old_array.shape != new_array.shape:
            return False

        # Check coordinate labels match (same keys and values)
        for dim in old_array.dims:
            if dim in old_array.coords and dim in new_array.coords:
                old_coords = old_array.coords[dim].values
                new_coords = new_array.coords[dim].values
                if not (
                    len(old_coords) == len(new_coords)
                    and np.array_equal(old_coords, new_coords)
                ):
                    return False
            elif (dim in old_array.coords) != (dim in new_array.coords):
                # One has the coordinate and the other doesn't
                return False

        return True

    @classmethod
    def _update_nd_ui_data_only(
        cls, label: str, value: xr.DataArray, children: list
    ) -> None:
        """
        Update an existing ND array UI with new data while preserving graph state.

        This function finds the Graph component in the children and updates its data
        while preserving zoom, pan, and other user interaction state by reusing the
        current figure's layout.
        """
        # Find the graph component in children
        graph = None
        logger.info(f"[{label}] Looking for Graph in {len(children)} children")
        for i, child in enumerate(children):
            if isinstance(child, dcc.Graph):
                logger.info(f"[{label}] Found Graph at index {i}")
                if (
                    hasattr(child, "id")
                    and isinstance(child.id, dict)
                    and child.id.get("type") == "data-array-graph"
                ):
                    logger.info(f"[{label}] Graph ID matches")
                    graph = child
                    break

        if graph is None:
            logger.warning(f"[{label}] Could not find graph component")
            return

        # Get current figure state (preserves zoom, pan, camera angles)
        curr_fig = graph.figure
        if curr_fig is None:
            logger.warning(f"[{label}] No figure state found")
            return

        logger.info(f"[{label}] Updating figure data with preserved layout")

        # Get slice dict - we update at the first slice (outer dims = 0)
        # The slider callback will handle slicing for user interactions
        outer_dims = value.dims[:-2]
        slice_dict = {dim: 0 for dim in outer_dims}
        sliced_array = value.isel(**slice_dict)

        # Update figure with preserved layout (preserves zoom, pan, camera)
        fig = go.Figure(layout=curr_fig["layout"])
        update_xarray_plot(fig, sliced_array)

        # Update the graph figure in place
        graph.figure = fig
        logger.info(f"[{label}] Graph figure updated")

    @classmethod
    def register_callbacks(cls, app):
        """
        Register a callback for ND arrays (ndim ≥ 3) that updates the graph based
        on slider values.
        """
        if not cls._nd_callback_registered:

            @app.callback(
                Output({"type": "data-array-graph", "index": MATCH}, "figure"),
                Input(
                    {"type": "data-array-slicer", "index": MATCH, "dim": ALL}, "value"
                ),
                State({"type": "data-array-slicer", "index": MATCH, "dim": ALL}, "id"),
                State({"type": "data-array-graph", "index": MATCH}, "figure"),
                State({"type": "data-array-graph", "index": MATCH}, "id"),
            )
            def update_nd_graph(slider_vals, slider_ids, curr_fig, graph_id):
                label = graph_id["index"]
                full_array = cls._data_nd.get(label)
                if full_array is None:
                    return curr_fig
                slice_dict = {
                    id_obj["dim"]: val for id_obj, val in zip(slider_ids, slider_vals)
                }
                for d in full_array.dims[:-2]:
                    if d not in slice_dict:
                        slice_dict[d] = 0
                sliced_array = full_array.isel(**slice_dict)
                fig = go.Figure(layout=curr_fig["layout"])
                update_xarray_plot(fig, sliced_array)
                return fig

            cls._nd_callback_registered = True

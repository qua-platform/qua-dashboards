import xarray as xr
from typing import Type, TypeVar, Optional
import json
from dash.development.base_component import Component
from dash import html, dcc, Input, Output, State, MATCH, ALL, callback_context
import dash_bootstrap_components as dbc
from plotly import graph_objects as go
from qua_dashboards.logging_config import logger
from qua_dashboards.data_dashboard.plotting import update_xarray_plot
from .base_data_component import BaseDataComponent

GRAPH_STYLE = {"aspect-ratio": "1 / 1", "max-width": "400px"}
T = TypeVar("T", bound=Component)


class DataArrayComponent(BaseDataComponent):
    # Store full ND arrays for use in callbacks.
    _data_3d = {}
    _nd_callback_registered = False

    @classmethod
    def can_handle(cls, value: any) -> bool:
        # Accept 2D or ND (>=2) xarray DataArrays.
        return isinstance(value, xr.DataArray) and value.ndim >= 2

    @classmethod
    def create_component(
        cls,
        label: str,
        value: xr.DataArray,
        existing_component: Optional[Component] = None,
        root_component_class: Type[T] = html.Div,
    ) -> T:
        if not cls._validate_existing_component(
            existing_component, value, root_component_class
        ):
            logger.info(f"Creating new data array component ({label}: {value})")
            root_component = cls.create_collapsible_root_component(
                label, root_component_class, "xarray_data_array_component"
            )
        else:
            value_str = str(value)
            if len(value_str) > 60:
                value_str = value_str[:60] + "..."
            logger.info(f"Using existing data array component ({label}: {value_str})")
            root_component = existing_component

        label_component, collapse_component = root_component.children

        if value.ndim == 2:
            # For 2D arrays, behave as before.
            if not collapse_component.children:
                fig = go.Figure(layout=dict(margin=dict(l=20, r=20, t=20, b=20)))
                graph = dcc.Graph(
                    figure=fig,
                    style=GRAPH_STYLE,
                )
                collapse_component.children = [graph]
            else:
                graph = collapse_component.children[0]

            logger.info("Updating xarray plot for 2D array")
            update_xarray_plot(graph.figure, value)

        elif value.ndim > 2:
            # For ND arrays (ndim>=3), assume the last two dims are for plotting.
            # All outer dimensions (dims[:-2]) get their own slider.
            outer_dims = value.dims[:-2]
            graph_width = GRAPH_STYLE.get("max-width", "400px")

            slider_controls = []
            for dim in outer_dims:
                size = value.sizes[dim]
                # Use coordinate values if available; otherwise use indices.
                if dim in value.coords:
                    coord = value.coords[dim].values
                    # Ensure the length matches the size.
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
                            style={
                                "textAlign": "right",
                                "paddingRight": "10px",
                            },
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

            control_container = html.Div(
                children=slider_controls,
                id={"type": "data-array-control-container", "index": label},
                style={"width": graph_width, "marginTop": "10px"},
            )

            # Create the initial slice: default to index 0 for every outer dimension.
            slice_dict = {dim: 0 for dim in outer_dims}
            sliced_array = value.isel(**slice_dict)
            fig = go.Figure(layout=dict(margin=dict(l=20, r=20, t=20, b=20)))
            update_xarray_plot(fig, sliced_array)
            graph = dcc.Graph(
                id={"type": "data-array-graph", "index": label},
                figure=fig,
                style=GRAPH_STYLE,
            )
            collapse_component.children = [graph, control_container]
            # Store the full ND array so the callback can update slices.
            cls._data_3d[label] = value
            logger.info(
                f"Created ND data array component with slicing controls for dims: {outer_dims}"
            )
        else:
            logger.error(
                "DataArrayComponent only supports 2D or ND arrays with ndim >= 2"
            )

        return root_component

    @staticmethod
    def _validate_existing_component(
        component: Component, value: xr.DataArray, root_component_class: Type[T]
    ) -> bool:
        if not isinstance(component, root_component_class):
            return False
        if not getattr(component, "data-class", None) == "xarray_data_array_component":
            return False
        return True

    @classmethod
    def register_callbacks(cls, app):
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
            def update_nd_graph(slider_values, slider_ids, current_fig, graph_id):
                label = graph_id["index"]
                full_array = cls._data_3d.get(label)
                if full_array is None:
                    return current_fig

                # Build a slice dictionary using the slider IDs and their corresponding values.
                slice_dict = {}
                for id_obj, value in zip(slider_ids, slider_values):
                    # Each id_obj is of the form: {'type': 'data-array-slicer', 'index': label, 'dim': dim_name}
                    slice_dict[id_obj["dim"]] = value

                # For any outer dimension not present (should not happen), default to 0.
                outer_dims = full_array.dims[:-2]
                for d in outer_dims:
                    if d not in slice_dict:
                        slice_dict[d] = 0

                sliced_array = full_array.isel(**slice_dict)
                fig = go.Figure(layout=current_fig["layout"])
                update_xarray_plot(fig, sliced_array)
                return fig

            cls._nd_callback_registered = True
            logger.info("Registered ND array slicing callbacks")

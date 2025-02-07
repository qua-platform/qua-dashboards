import xarray as xr
from typing import Type, TypeVar, Optional
from dash.development.base_component import Component
from dash import html, dcc, Input, Output, State, MATCH
from plotly import graph_objects as go
from qua_dashboards.logging_config import logger
from qua_dashboards.data_dashboard.plotting import update_xarray_plot
from .base_data_component import BaseDataComponent

GRAPH_STYLE = {"aspect-ratio": "1 / 1", "max-width": "400px"}
T = TypeVar("T", bound=Component)


class DataArrayComponent(BaseDataComponent):
    # Dictionary to store full 3D arrays (keyed by label) for use in callbacks.
    _data_3d = {}
    _3d_callback_registered = False

    @classmethod
    def can_handle(cls, value: any) -> bool:
        # Now accept 2D or 3D xarray DataArrays.
        return isinstance(value, xr.DataArray) and value.ndim in (2, 3)

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
            # Existing behavior for 2D arrays.
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

        elif value.ndim == 3:
            # New behavior for 3D arrays: slice along the first (outer) dimension.
            outer_dim = value.dims[0]
            # If a coordinate exists for the outer dimension, use it
            # otherwise use indices.
            if outer_dim in value.coords:
                coord = value.coords[outer_dim].values
            else:
                coord = list(range(value.shape[0]))
            first_value = coord[0] if len(coord) > 0 else 0

            # Choose a slider for numeric values and a dropdown for string values.
            if isinstance(first_value, (int, float)):
                control = dcc.Slider(
                    id={"type": "data-array-slicer", "index": label},
                    min=0,
                    max=len(coord) - 1,
                    step=1,
                    value=0,
                    marks={i: str(coord[i]) for i in range(len(coord))},
                )
            elif isinstance(first_value, str):
                control = dcc.Dropdown(
                    id={"type": "data-array-slicer", "index": label},
                    options=[
                        {"label": str(v), "value": i} for i, v in enumerate(coord)
                    ],
                    value=0,
                )
            else:
                # Fallback: use a slider with indices.
                control = dcc.Slider(
                    id={"type": "data-array-slicer", "index": label},
                    min=0,
                    max=len(coord) - 1,
                    step=1,
                    value=0,
                    marks={i: str(coord[i]) for i in range(len(coord))},
                )

            # Create the initial slice (using index 0 of the outer dimension).
            sliced_array = value.isel({outer_dim: 0})
            fig = go.Figure(layout=dict(margin=dict(l=20, r=20, t=20, b=20)))
            update_xarray_plot(fig, sliced_array)
            # Assign a pattern-matching id to the graph so it updates by the callback.
            graph = dcc.Graph(
                id={"type": "data-array-graph", "index": label},
                figure=fig,
                style=GRAPH_STYLE,
            )
            # Place both the graph and the slicing control in the collapsible section.
            collapse_component.children = [graph, control]
            # Store the full 3D array so the callback can access it.
            cls._data_3d[label] = value
            logger.info("Created 3D data array component with slicing control")
        else:
            logger.error("DataArrayComponent only supports 2D or 3D arrays")

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
        """
        Registers a pattern-matching callback for 3D arrays.
        This callback listens for changes in the slicing control (slider or dropdown)
        and updates the corresponding graph.
        """
        if not cls._3d_callback_registered:

            @app.callback(
                Output({"type": "data-array-graph", "index": MATCH}, "figure"),
                [Input({"type": "data-array-slicer", "index": MATCH}, "value")],
                [
                    State({"type": "data-array-graph", "index": MATCH}, "figure"),
                    State({"type": "data-array-graph", "index": MATCH}, "id"),
                ],
            )
            def update_3d_graph(slice_index, current_fig, graph_id):
                label = graph_id["index"]
                full_array = cls._data_3d.get(label)
                if full_array is None:
                    return current_fig
                outer_dim = full_array.dims[0]
                sliced_array = full_array.isel({outer_dim: slice_index})
                # Create a new figure based on the current layout.
                fig = go.Figure(layout=current_fig["layout"])
                update_xarray_plot(fig, sliced_array)
                return fig

            cls._3d_callback_registered = True
            logger.info("Registered 3D array slicing callbacks")

from typing import Type, Optional, TypeVar


from qua_dashboards.data_dashboard.data_components.base_data_component import (
    BaseDataComponent,
)
from qua_dashboards.data_dashboard.plotting import update_xarray_plot
from qua_dashboards.logging_config import logger


import dash_bootstrap_components as dbc
import xarray as xr
from dash import dcc, html
from dash.development.base_component import Component
from plotly import graph_objects as go


GRAPH_STYLE = {"aspect-ratio": "1 / 1", "max-width": "400px"}
T = TypeVar("T", bound=Component)


class DatasetComponent(BaseDataComponent):
    @classmethod
    def can_handle(cls, value: any) -> bool:
        return isinstance(value, xr.Dataset) and value.ndim <= 2

    @classmethod
    def create_component(
        cls,
        label: str,
        value: xr.Dataset,
        existing_component: Optional[Component] = None,
        root_component_class: Type[T] = html.Div,
    ) -> T:
        """
        Create or update a component for visualizing an xarray Dataset with multiple DataArrays.

        Args:
            label (str): The label for the component.
            value (xr.Dataset): The Dataset to visualize.
            existing_component (Optional[Component]): An existing component to update, if any.
            root_component_class (Type[T]): The class type for the root component.

        Returns:
            T: The created or updated component.
        """
        if not cls._validate_existing_component(
            existing_component, value, root_component_class
        ):
            logger.info(f"Creating new dataset component ({label}: {value})")
            root_component = cls.create_collapsible_root_component(
                label, root_component_class, "xarray_dataset_component"
            )

            collapse_component = root_component.children[1]

            # Create a new graph for each DataArray in the Dataset
            graph_containers = []
            for data_array_name, data_array in value.data_vars.items():
                fig = go.Figure(layout=dict(margin=dict(l=20, r=20, t=20, b=20)))
                graph = dcc.Graph(
                    figure=fig,
                    style=GRAPH_STYLE,
                    id={"type": "graph", "index": data_array_name},
                )
                graph_containers.append(
                    dbc.Col(
                        html.Div(
                            [
                                html.Label(
                                    f"{label}/{data_array_name}",
                                    style={
                                        "display": "flex",
                                        "justify-content": "center",
                                    },
                                ),
                                graph,
                            ],
                        )
                    )
                )

            collapse_component.children = dbc.Row(
                graph_containers, style={"display": "flex"}
            )

        else:
            value_str = str(value)
            if len(value_str) > 60:
                value_str = value_str[:60] + "..."
            logger.info(f"Using existing dataset component ({label}: {value_str})")
            root_component = existing_component
            collapse_component = root_component.children[1]
            graph_containers = collapse_component.children.children

        for graph_container, (data_array_name, data_array) in zip(
            graph_containers, value.data_vars.items()
        ):
            graph = graph_container.children.children[1]
            update_xarray_plot(graph.figure, data_array)

        return root_component

    @staticmethod
    def _validate_existing_component(
        component: Component, value: xr.Dataset, root_component_class: Type[T]
    ) -> bool:
        if not isinstance(component, root_component_class):
            return False
        if not getattr(component, "data-class", None) == "xarray_dataset_component":
            return False

        collapse_component = component.children[1]
        graph_containers = collapse_component.children.children
        labels = [
            graph_container.children.children[0].children
            for graph_container in graph_containers
        ]
        html_array_names = [label.rsplit("/", 1)[1] for label in labels]

        xarray_array_names = list(value.data_vars.keys())

        if html_array_names != xarray_array_names:
            return False

        return True

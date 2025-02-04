import xarray as xr
from typing import Type, TypeVar, Optional
from dash.development.base_component import Component
from dash import html, dcc
import dash_bootstrap_components as dbc
from plotly import graph_objects as go
from qua_dashboards.logging_config import logger
from qua_dashboards.data_dashboard.plotting import update_xarray_plot

GRAPH_STYLE = {"aspect-ratio": "1 / 1", "max-width": "400px"}

T = TypeVar("T", bound=Component)


def _create_root_component(
    label: str, root_component_class: Type[T], data_class: str
) -> T:
    """
    Create the root component structure with a label and a collapsible section.

    Args:
        label (str): The label for the component.
        root_component_class (Type[T]): The class type for the root component.

    Returns:
        T: An instance of the root component class.
    """
    return root_component_class(
        id=f"data-entry-{label}",
        children=[
            dbc.Label(
                label,
                id={"type": "collapse-button", "index": label},
                style={"fontWeight": "bold"},
            ),
            dbc.Collapse(
                [],
                id={"type": "collapse", "index": label},
                is_open=True,
            ),
        ],
        **{"data-class": data_class},
    )


def _validate_data_array_component(
    component: Component, value: xr.DataArray, root_component_class: Type[T]
) -> bool:
    """
    Validate the structure of a component to ensure it matches the expected layout.

    Args:
        component (Component): The component to validate.
        root_component_class (Type[T]): The expected class type for the root component.

    Returns:
        bool: True if the component structure is valid, False otherwise.
    """
    if not isinstance(component, root_component_class):
        return False
    if not getattr(component, "data-class", None) == "xarray_data_array_component":
        return False
    if value.ndim > 2:
        return False
    return True


def create_data_array_component(
    label: str,
    value: xr.DataArray,
    existing_component: Optional[Component] = None,
    root_component_class: Type[T] = html.Div,
) -> T:
    """
    Create or update a component for visualizing a single xarray DataArray.

    Args:
        label (str): The label for the component.
        value (xr.DataArray): The DataArray to visualize.
        existing_component (Optional[Component]): An existing component to update, if any.
        root_component_class (Type[T]): The class type for the root component.

    Returns:
        T: The created or updated component.
    """
    if not _validate_data_array_component(
        existing_component, value, root_component_class
    ):
        logger.info(f"Creating new data array component ({label}: {value})")
        root_component = _create_root_component(
            label, root_component_class, "xarray_data_array_component"
        )
    else:
        logger.info(f"Using existing data array component ({label}: {value})")
        root_component = existing_component

    label_component, collapse_component = root_component.children

    if not collapse_component.children:
        # Create a new graph if none exists
        fig = go.Figure(layout=dict(margin=dict(l=20, r=20, t=20, b=20)))
        graph = dcc.Graph(
            figure=fig,
            style=GRAPH_STYLE,
        )
        collapse_component.children = [graph]
    else:
        # Use the existing graph
        graph = collapse_component.children[0]

    logger.info("Updating xarray plot")
    update_xarray_plot(graph.figure, value)

    return root_component


def _validate_dataset_component(
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


def create_dataset_component(
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
    if not _validate_dataset_component(existing_component, value, root_component_class):
        logger.info(f"Creating new dataset component ({label}: {value})")
        root_component = _create_root_component(
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
                                style={"display": "flex", "justify-content": "center"},
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
        logger.info(f"Using existing dataset component ({label}: {value})")
        root_component = existing_component
        collapse_component = root_component.children[1]
        graph_containers = collapse_component.children.children

    for graph_container, (data_array_name, data_array) in zip(
        graph_containers, value.data_vars.items()
    ):
        graph = graph_container.children.children[1]
        update_xarray_plot(graph.figure, data_array)

    return root_component

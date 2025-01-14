import xarray as xr
from typing import Type, TypeVar, Optional
from dash.development.base_component import Component
from dash import html, dcc
import dash_bootstrap_components as dbc
from plotly import graph_objects as go
from qua_dashboards.logging_config import logger
from qua_dashboards.data_visualizer.plotting import update_xarray_plot

GRAPH_STYLE = {"aspect-ratio": "1 / 1", "max-width": "400px"}

T = TypeVar("T", bound=Component)


def _create_root_component(label: str, root_component_class: Type[T]) -> T:
    """
    Create the root component structure with a label and a collapsible section.

    Args:
        label (str): The label for the component.
        root_component_class (Type[T]): The class type for the root component.

    Returns:
        T: An instance of the root component class.
    """
    return root_component_class(
        id=label,
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
    )


def _validate_component_structure(
    component: Component, root_component_class: Type[T]
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
    if len(component.children) != 2:
        return False
    if not isinstance(component.children[0], dbc.Label):
        return False
    if not isinstance(component.children[1], dbc.Collapse):
        return False

    collapse_component = component.children[1]
    if not all(isinstance(child, dbc.Col) for child in collapse_component.children):
        return False
    if not all(
        isinstance(col.children[0], dcc.Graph) for col in collapse_component.children
    ):
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
    if not _validate_component_structure(existing_component, root_component_class):
        logger.info(f"Creating new data array component ({label}: {value})")
        root_component = _create_root_component(label, root_component_class)
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
    if not _validate_component_structure(existing_component, root_component_class):
        logger.info(f"Creating new dataset component ({label}: {value})")
        root_component = _create_root_component(label, root_component_class)
    else:
        logger.info(f"Using existing dataset component ({label}: {value})")
        root_component = existing_component

    label_component, collapse_component = root_component.children

    if not collapse_component.children:
        # Create a new graph for each DataArray in the Dataset
        graphs = []
        for data_array_name, data_array in value.data_vars.items():
            fig = go.Figure(layout=dict(margin=dict(l=20, r=20, t=20, b=20)))
            graph = dcc.Graph(
                figure=fig,
                style=GRAPH_STYLE,
                id={"type": "graph", "index": data_array_name},
            )
            update_xarray_plot(graph.figure, data_array)
            graphs.append(dbc.Col(graph, width="auto"))

        collapse_component.children = [
            dbc.Row(graphs, style={"display": "flex", "flex-wrap": "nowrap"})
        ]
    else:
        # Update existing graphs
        for col, (data_array_name, data_array) in zip(
            collapse_component.children[0].children, value.data_vars.items()
        ):
            graph = col.children[0]
            update_xarray_plot(graph.figure, data_array)

    return root_component

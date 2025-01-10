import xarray as xr
from typing import Type, TypeVar, Optional

from dash.development.base_component import Component
from dash import html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from plotly import graph_objects as go

from qua_dashboards.logging_config import logger
from qua_dashboards.data_visualizer.plotting import update_xarray_plot


GRAPH_STYLE = {"aspect-ratio": "1 / 1", "max-width": "400px"}


T = TypeVar("T", bound=Component)


def create_data_array_component(
    label: str,
    value: xr.DataArray,
    existing_component: Optional[Component] = None,
    root_component_class: Type[T] = html.Div,
) -> T:
    if existing_component is None:
        root_component = root_component_class(
            id=label,
            children=[
                dbc.Label(label, id={"type": "collapse-button", "index": label}),
                dbc.Collapse(
                    [str(value)],
                    id={"type": "collapse", "index": label},
                    is_open=True,
                ),
            ],
        )
    else:
        root_component = existing_component

    label_component, collapse_component = root_component.children

    if not isinstance(collapse_component.children[0], dcc.Graph):
        fig = go.Figure(layout=dict(margin=dict(l=20, r=20, t=20, b=20)))
        graph = dcc.Graph(
            figure=fig,
            style=GRAPH_STYLE,
        )
        collapse_component.children = [graph]
    else:
        graph = collapse_component.children[0]

    logger.info("Updating xarray plot")
    update_xarray_plot(graph.figure, value)

    return root_component

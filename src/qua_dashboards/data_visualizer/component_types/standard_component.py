from typing import Type, TypeVar, Optional, Any

from dash.development.base_component import Component
from dash import html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from plotly import graph_objects as go

from qua_dashboards.logging_config import logger
from qua_dashboards.data_visualizer.plotting import update_xarray_plot


GRAPH_STYLE = {"aspect-ratio": "1 / 1", "max-width": "400px"}


T = TypeVar("T", bound=Component)


def create_standard_component(
    label: str,
    value: Any,
    existing_component: Optional[Component] = None,
    root_component_class: Type[T] = html.Div,
) -> T:
    if existing_component is None or len(existing_component.children) != 2:
        root_component = root_component_class(
            id=label,
            children=[
                dbc.Label(label, style={"fontWeight": "bold"}),
                dbc.Label(":  "),
                dbc.Label(str(value)),
            ],
        )
    else:
        root_component = root_component_class(id=label)

    value_component = root_component.children[-1]
    value_component.children = str(value)

    return root_component

from typing import Type, TypeVar, Optional, Any

from dash.development.base_component import Component
from dash import html
from dash import dcc
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
    if not _validate_standard_component(
        component=existing_component,
        value=value,
        root_component_class=root_component_class,
    ):
        logger.info(f"Creating new standard component ({label}: {value})")
        root_component = root_component_class(
            id=label,
            children=[
                dbc.Label(label, style={"fontWeight": "bold"}),
                dbc.Label(":  ", style={"whiteSpace": "pre"}),
                dbc.Label(str(value)),
            ],
            **{"data-class": "standard_component"},
        )
    else:
        logger.info(f"Using existing standard component ({label}: {value})")
        root_component = existing_component

    value_component = root_component.children[-1]
    value_component.children = str(value)

    return root_component


def _validate_standard_component(
    component: Component, value: Any, root_component_class: Type[T]
) -> bool:
    if not isinstance(component, root_component_class):
        return False
    if not getattr(component, "data-class", None) == "standard_component":
        return False
    return True

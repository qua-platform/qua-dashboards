import xarray as xr
from typing import Type, TypeVar, Optional
from dash.development.base_component import Component
from dash import html, dcc
import dash_bootstrap_components as dbc
from plotly import graph_objects as go
from qua_dashboards.logging_config import logger
from qua_dashboards.data_dashboard.plotting import update_xarray_plot
from .base_data_component import BaseDataComponent

GRAPH_STYLE = {"aspect-ratio": "1 / 1", "max-width": "400px"}
T = TypeVar("T", bound=Component)


class DataArrayComponent(BaseDataComponent):
    @classmethod
    def can_handle(cls, value: any) -> bool:
        return isinstance(value, xr.DataArray) and value.ndim <= 2

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

    @staticmethod
    def _validate_existing_component(
        component: Component, value: xr.DataArray, root_component_class: Type[T]
    ) -> bool:
        if not isinstance(component, root_component_class):
            return False
        if not getattr(component, "data-class", None) == "xarray_data_array_component":
            return False
        return True

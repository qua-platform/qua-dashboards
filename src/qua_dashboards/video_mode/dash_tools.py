from abc import ABC
from enum import Flag, auto
from typing import Any, Dict, List, Literal, Optional

from dash import html
import plotly.graph_objects as go
import xarray as xr
import dash_bootstrap_components as dbc

from qua_dashboards.utils.dash_utils import create_input_field


__all__ = ["xarray_to_plotly", "BaseDashComponent", "ModifiedFlags"]


class ModifiedFlags(Flag):
    """Flags indicating what needs to be modified after parameter changes."""

    NONE = 0
    PARAMETERS_MODIFIED = auto()
    PROGRAM_MODIFIED = auto()
    CONFIG_MODIFIED = auto()


class BaseDashComponent(ABC):
    def __init__(self, *args, component_id: str, **kwargs):
        assert not args, "BaseDashComponent does not accept any positional arguments"
        assert not kwargs, "BaseDashComponent does not accept any keyword arguments"

        self.component_id = component_id

    def update_parameters(self, parameters: Dict[str, Dict[str, Any]]) -> ModifiedFlags:
        """Update the component's attributes based on the input values."""
        return ModifiedFlags.NONE

    def get_dash_components(self, include_subcomponents: bool = True) -> List[html.Div]:
        """Return a list of Dash components.

        Args:
            include_subcomponents (bool, optional): Whether to include subcomponents. Defaults to True.

        Returns:
            List[html.Div]: A list of Dash components.
        """
        return []

    def get_component_ids(self) -> List[str]:
        """Return a list of component IDs for this component including subcomponents."""
        return [self.component_id]


def xarray_to_plotly(da: xr.DataArray):
    """Convert an xarray DataArray to a Plotly figure.

    Args:
        da (xr.DataArray): The data array to convert.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure with the data.
    """
    if len(da.coords) != 2:
        raise ValueError("DataArray must have exactly 2 coordinates.")

    coords_iter = iter(da.coords.items())

    y_label, y_coord = next(coords_iter)
    y_label = y_coord.attrs.get("long_name", y_label)
    y_unit = y_coord.attrs.get("units", "")

    x_label, x_coord = next(coords_iter)
    x_label = x_coord.attrs.get("long_name", x_label)
    x_unit = x_coord.attrs.get("units", "")

    z_label = da.attrs.get("long_name", da.name or "Value")
    z_unit = da.attrs.get("units", "")

    xaxis_label = f"{x_label} ({x_unit})" if x_unit else x_label
    yaxis_label = f"{y_label} ({y_unit})" if y_unit else y_label
    zaxis_label = f"{z_label} ({z_unit})" if z_unit else z_label

    fig = go.Figure(
        go.Heatmap(
            z=da.values,
            x=x_coord.values,
            y=y_coord.values,
            colorscale="plasma",
            colorbar=dict(title=zaxis_label),
        )
    )
    fig.update_layout(xaxis_title=xaxis_label, yaxis_title=yaxis_label)
    return fig


def xarray_to_plotly_click(da: xr.DataArray, added_points):
    """Convert an xarray DataArray to a Plotly figure.
       Add a separate trace (go.Scatter) for added points by clicks

    Args:
        da (xr.DataArray): The data array to convert.
        added_points (dict): Added points by clicking on the figure

    Returns:
        plotly.graph_objects.Figure: A Plotly figure with the data.
    """
    if len(da.coords) != 2:
        raise ValueError("DataArray must have exactly 2 coordinates.")

    coords_iter = iter(da.coords.items())

    y_label, y_coord = next(coords_iter)
    y_label = y_coord.attrs.get("long_name", y_label)
    y_unit = y_coord.attrs.get("units", "")

    x_label, x_coord = next(coords_iter)
    x_label = x_coord.attrs.get("long_name", x_label)
    x_unit = x_coord.attrs.get("units", "")

    z_label = da.attrs.get("long_name", da.name or "Value")
    z_unit = da.attrs.get("units", "")

    xaxis_label = f"{x_label} ({x_unit})" if x_unit else x_label
    yaxis_label = f"{y_label} ({y_unit})" if y_unit else y_label
    zaxis_label = f"{z_label} ({z_unit})" if z_unit else z_label

    fig = go.Figure(data=[
        go.Heatmap(
            z=da.values,
            x=x_coord.values,
            y=y_coord.values,
            colorscale="plasma",
            colorbar=dict(title=zaxis_label),
            name='',
        ),
        go.Scatter(
            x=added_points['x'], 
            y=added_points['y'], 
            mode='markers + text', 
            marker=dict(color='white', size=10, line=dict(color='black', width=1)),
            text=added_points['labels'],
            textposition='top center',#'middle center', 
            textfont=dict(color='white'), #,shadow='1px 1px 10px white'), # offset-x | offset-y | blur-radius | color
            showlegend=False,
            name='',
        ),
    ])
    fig.update_layout(xaxis_title=xaxis_label, yaxis_title=yaxis_label,
                      clickmode = 'event + select')
    return fig


def xarray_to_heatmap(da: xr.DataArray):
    """Convert an xarray DataArray to a Plotly heatmap.

    Args:
        da (xr.DataArray): The data array to convert.

    Returns:
        plotly.graph_objects.Heatmap: A Plotly heatmap with the data.
        dictionary with keys xaxis_title and yaxis_title
    """
    if len(da.coords) != 2:
        raise ValueError("DataArray must have exactly 2 coordinates.")

    coords_iter = iter(da.coords.items())

    y_label, y_coord = next(coords_iter)
    y_label = y_coord.attrs.get("long_name", y_label)
    y_unit = y_coord.attrs.get("units", "")

    x_label, x_coord = next(coords_iter)
    x_label = x_coord.attrs.get("long_name", x_label)
    x_unit = x_coord.attrs.get("units", "")

    z_label = da.attrs.get("long_name", da.name or "Value")
    z_unit = da.attrs.get("units", "")

    xaxis_label = f"{x_label} ({x_unit})" if x_unit else x_label
    yaxis_label = f"{y_label} ({y_unit})" if y_unit else y_label
    zaxis_label = f"{z_label} ({z_unit})" if z_unit else z_label

    heatmap = go.Heatmap(
        z=da.values,
        x=x_coord.values,
        y=y_coord.values,
        colorscale="plasma",
        colorbar=dict(title=zaxis_label),
        zorder=0,
        name='',
    )

    titles = {'xaxis_title': xaxis_label, 'yaxis_title': yaxis_label}
    dict_heatmap = {'heatmap': heatmap, 'xy-titles': titles}
    return dict_heatmap


def create_axis_layout(
    axis: Literal["x", "y"],
    span: float,
    points: int,
    min_span: float,
    max_span: Optional[float] = None,
    units: Optional[str] = None,
    component_id: Optional[str] = None,
):
    if component_id is None:
        ids = {"span": f"{axis.lower()}-span", "points": f"{axis.lower()}-points"}
    else:
        ids = {
            "span": {"type": component_id, "index": f"{axis.lower()}-span"},
            "points": {"type": component_id, "index": f"{axis.lower()}-points"},
        }
    return dbc.Col(
        dbc.Card(
            [
                dbc.CardHeader(axis.upper()),
                dbc.CardBody(
                    [
                        create_input_field(
                            id=ids["span"],
                            label="Span",
                            value=span,
                            min=min_span,
                            max=max_span,
                            input_style={"width": "100px"},
                            units=units,
                        ),
                        create_input_field(
                            id=ids["points"],
                            label="Points",
                            value=points,
                            min=1,
                            max=501,
                            step=1,
                        ),
                    ]
                ),
            ],
            className="h-100",
        ),
        md=6,
        className="mb-3",
    )

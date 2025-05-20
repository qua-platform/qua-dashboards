from typing import Literal, Optional

import plotly.graph_objects as go
import xarray as xr
import dash_bootstrap_components as dbc

from qua_dashboards.utils.dash_utils import create_input_field


__all__ = [
    "xarray_to_plotly",
]


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
    fig.update_layout(
        xaxis_title=xaxis_label, yaxis_title=yaxis_label, template="plotly_dark"
    )
    return fig

def xarray_to_heatmap(da: xr.DataArray):
    """Convert an xarray DataArray to a Plotly heatmap.

    Args:
        da (xr.DataArray): The data array to convert.

    Returns:
        dict{'heatmap'=heatmap,'xy-titles'=titles}, where
            - heatmap (plotly.graph_objects.Heatmap) contains the data
            - titles is a dictionary containing the axis labels 
              {'xaxis_title': xaxis_label, 'yaxis_title': yaxis_label}
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
                dbc.CardHeader(axis.upper(), className="text-light"),
                dbc.CardBody(
                    [
                        create_input_field(
                            id=ids["span"],
                            label="Span",
                            value=span,
                            min=min_span,
                            max=max_span,
                            input_style={"width": "100px"},
                            units=units if units is not None else "",
                        ),
                        create_input_field(
                            id=ids["points"],
                            label="Points",
                            value=points,
                            min=1,
                            max=501,
                            step=1,
                        ),
                    ],
                    className="text-light",
                ),
            ],
            color="dark",
            inverse=True,
            className="h-100 tab-card-dark",
        ),
        md=6,
        className="mb-3",
    )

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from qua_dashboards.logging_config import logger
import xarray as xr
from typing import Any, Tuple, Union


def get_axis_info(data_array: xr.DataArray, dim: str) -> Tuple[Any, str]:
    """
    Retrieve the coordinate values and a formatted axis label (including units, if available)
    for the given dimension.

    Args:
        data_array: The xarray DataArray.
        dim: The dimension name.

    Returns:
        A tuple of (coordinate values, axis label).
    """
    if dim in data_array.coords:
        coord = data_array.coords[dim]
        units = coord.attrs.get("units") if hasattr(coord, "attrs") else None
    else:
        size = data_array.sizes.get(dim, len(data_array))
        coord = list(range(size))
        units = None
    label = dim + (f" ({units})" if units else "")
    return coord, label


def apply_log_transform(values: Union[np.ndarray, list], log_flag: bool) -> np.ndarray:
    """
    Apply a log10 transformation to the values if log_flag is True and all values are positive.
    Otherwise, return the original values.

    Args:
        values: The numeric values to potentially transform.
        log_flag: Whether to apply log transformation.

    Returns:
        The (possibly transformed) values as a numpy array.
    """
    arr = np.array(values)
    if log_flag:
        if (arr <= 0).any():
            logger.warning(
                "Non-positive values encountered for log scale; skipping log transform."
            )
            return arr
        return np.log10(arr)
    return arr


def update_axis_scaling(fig: Figure, data_array: xr.DataArray) -> None:
    """
    Update the x- and y-axis scaling on the figure based on log-scaling flags in data_array.attrs.

    Args:
        fig: The Plotly figure to update.
        data_array: The xarray DataArray containing potential log scale flags.
    """
    if data_array.attrs.get("log_x", False):
        fig.update_xaxes(type="log")
    if data_array.attrs.get("log_y", False):
        fig.update_yaxes(type="log")


def update_global_layout(fig: Figure, data_array: xr.DataArray) -> None:
    """
    Update the figure's layout with global settings from data_array.attrs.
    A title is applied only if explicitly provided in the attributes.

    Args:
        fig: The Plotly figure to update.
        data_array: The xarray DataArray containing global layout metadata.
    """
    layout = data_array.attrs.get("plot_layout", {})
    if "title" in data_array.attrs:
        fig.update_layout(title=data_array.attrs["title"], **layout)
    else:
        fig.update_layout(**layout)


def plot_1d(data_array: xr.DataArray) -> Figure:
    """
    Create a new 1D line plot for the given xarray DataArray.

    Args:
        data_array: A 1D xarray DataArray.

    Returns:
        A Plotly figure representing the line plot.
    """
    x_dim = data_array.dims[0]
    x, x_label = get_axis_info(data_array, x_dim)
    y_label = data_array.name if data_array.name is not None else "Value"
    if "units" in data_array.attrs:
        y_label += f" ({data_array.attrs['units']})"
    fig = px.line(x=x, y=data_array.values)
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    update_axis_scaling(fig, data_array)
    update_global_layout(fig, data_array)
    return fig


def update_1d(fig: Figure, data_array: xr.DataArray) -> Figure:
    """
    Update an existing 1D line plot with new data from the xarray DataArray.

    Args:
        fig: The existing Plotly figure.
        data_array: A 1D xarray DataArray.

    Returns:
        The updated Plotly figure.
    """
    x_dim = data_array.dims[0]
    x, x_label = get_axis_info(data_array, x_dim)
    y_label = data_array.name if data_array.name is not None else "Value"
    if "units" in data_array.attrs:
        y_label += f" ({data_array.attrs['units']})"
    if not fig.data:
        fig.add_trace(go.Scatter())
    fig.update_traces(x=x, y=data_array.values)
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    update_axis_scaling(fig, data_array)
    update_global_layout(fig, data_array)
    return fig


def plot_2d(data_array: xr.DataArray) -> Figure:
    """
    Create a new 2D heatmap plot for the given xarray DataArray.

    Args:
        data_array: A 2D xarray DataArray.

    Returns:
        A Plotly figure representing the heatmap.
    """
    y_dim, x_dim = data_array.dims[0], data_array.dims[1]
    x, x_label = get_axis_info(data_array, x_dim)
    y, y_label = get_axis_info(data_array, y_dim)
    z = apply_log_transform(data_array.values, data_array.attrs.get("log_z", False))
    # Build colorbar title from the data array's name and units.
    colorbar_title = ""
    if data_array.name:
        colorbar_title += data_array.name
    if "units" in data_array.attrs:
        colorbar_title += f" ({data_array.attrs['units']})"
    heatmap = go.Heatmap(z=z, x=x, y=y, colorbar=dict(title=colorbar_title))
    fig = go.Figure(data=heatmap)
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    update_axis_scaling(fig, data_array)
    update_global_layout(fig, data_array)
    return fig


def update_2d(fig: Figure, data_array: xr.DataArray) -> Figure:
    """
    Update an existing 2D heatmap plot with new data from the xarray DataArray.

    Args:
        fig: The existing Plotly figure.
        data_array: A 2D xarray DataArray.

    Returns:
        The updated Plotly figure.
    """
    y_dim, x_dim = data_array.dims[0], data_array.dims[1]
    x, x_label = get_axis_info(data_array, x_dim)
    y, y_label = get_axis_info(data_array, y_dim)
    z = apply_log_transform(data_array.values, data_array.attrs.get("log_z", False))
    if not fig.data:
        fig.add_trace(go.Heatmap())
    fig.update_traces(z=z, x=x, y=y)
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    update_axis_scaling(fig, data_array)
    # Update colorbar title for all heatmap traces.
    colorbar_title = ""
    if data_array.name:
        colorbar_title += data_array.name
    if "units" in data_array.attrs:
        colorbar_title += f" ({data_array.attrs['units']})"
    for trace in fig.data:
        if isinstance(trace, go.Heatmap):
            trace.colorbar.title = colorbar_title
    update_global_layout(fig, data_array)
    return fig


def plot_xarray(data_array: xr.DataArray, fig: Figure = None) -> Figure:
    """
    Create a new Plotly figure from an xarray DataArray.
    Supports 1D and 2D arrays.

    Args:
        data_array: An xarray DataArray (1D or 2D).
        fig: Optional existing figure (ignored in creation).

    Returns:
        A new Plotly figure.

    Raises:
        ValueError: If data_array.ndim is not 1 or 2.
    """
    logger.info(f"Plotting xarray data with dimensions: {data_array.dims}")
    if data_array.ndim == 1:
        return plot_1d(data_array)
    elif data_array.ndim == 2:
        return plot_2d(data_array)
    else:
        logger.error(f"Unsupported array dimensions: {data_array.ndim}")
        raise ValueError(
            f"Only 1D and 2D arrays are supported, not {data_array.ndim}D."
        )


def update_xarray_plot(fig: Figure, data_array: xr.DataArray) -> Figure:
    """
    Update an existing Plotly figure with new data from an xarray DataArray.
    Supports 1D and 2D arrays.

    Args:
        fig: The existing Plotly figure.
        data_array: An xarray DataArray (1D or 2D).

    Returns:
        The updated Plotly figure.

    Raises:
        ValueError: If data_array.ndim is not 1 or 2.
    """
    logger.info(f"Updating plot with xarray data: {data_array.name}")
    if data_array.ndim == 1:
        return update_1d(fig, data_array)
    elif data_array.ndim == 2:
        return update_2d(fig, data_array)
    else:
        logger.error(f"Unsupported array dimensions: {data_array.ndim}")
        raise ValueError(
            f"Only 1D and 2D arrays are supported, not {data_array.ndim}D."
        )

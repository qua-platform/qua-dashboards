import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from qua_dashboards.logging_config import logger


def get_axis_info(data_array, dim):
    """
    Returns the coordinate values and axis label (with units if available)
    for the given dimension.
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


def apply_log_transform(values, log_flag):
    """
    If log_flag is True and all values are positive, returns log10(values);
    otherwise returns values.
    """
    if log_flag:
        if (values <= 0).any():
            logger.warning(
                "Non-positive values encountered for log scale; skipping log transform."
            )
            return values
        return np.log10(values)
    return values


def update_axis_scaling(fig, data_array):
    """
    Updates x- and y-axes types on fig if log scaling is requested.
    """
    if data_array.attrs.get("log_x", False):
        fig.update_xaxes(type="log")
    if data_array.attrs.get("log_y", False):
        fig.update_yaxes(type="log")


def update_global_layout(fig, data_array):
    """
    Applies global layout customization from data_array.attrs.
    """
    layout = data_array.attrs.get("plot_layout", {})
    title = data_array.attrs.get(
        "title", data_array.name if data_array.name is not None else ""
    )
    fig.update_layout(title=title, **layout)


def plot_1d(data_array):
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


def update_1d(fig, data_array):
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


def plot_2d(data_array):
    y_dim, x_dim = data_array.dims[0], data_array.dims[1]
    x, x_label = get_axis_info(data_array, x_dim)
    y, y_label = get_axis_info(data_array, y_dim)
    z = data_array.values
    z = apply_log_transform(z, data_array.attrs.get("log_z", False))
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


def update_2d(fig, data_array):
    y_dim, x_dim = data_array.dims[0], data_array.dims[1]
    x, x_label = get_axis_info(data_array, x_dim)
    y, y_label = get_axis_info(data_array, y_dim)
    z = data_array.values
    z = apply_log_transform(z, data_array.attrs.get("log_z", False))
    if not fig.data:
        fig.add_trace(go.Heatmap())
    fig.update_traces(z=z, x=x, y=y)
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    update_axis_scaling(fig, data_array)
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


def plot_xarray(data_array, fig=None):
    """
    Creates a new Plotly figure from an xarray DataArray.
    Supports 1D and 2D arrays.
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


def update_xarray_plot(fig, data_array):
    """
    Updates an existing Plotly figure with new data from an xarray DataArray.
    Supports 1D and 2D arrays.
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

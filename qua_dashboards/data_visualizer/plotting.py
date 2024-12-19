import plotly.express as px
import plotly.graph_objects as go
from qua_dashboards.logging_config import logger


def plot_xarray(data_array, fig=None):
    logger.info(f"Plotting xarray data with dimensions: {data_array.dims}")

    if data_array.ndim == 1:
        logger.debug("Creating 1D line plot")
        fig = px.line(x=data_array.coords[data_array.dims[0]], y=data_array.values)
    elif data_array.ndim == 2:
        logger.debug("Creating 2D heatmap")
        fig = go.Figure(
            data=go.Heatmap(
                z=data_array.values,
                x=data_array.coords[data_array.dims[1]],
                y=data_array.coords[data_array.dims[0]],
            )
        )
    else:
        logger.error(f"Unsupported array dimensions: {data_array.ndim}")
        raise ValueError(
            f"Only 1D and 2D arrays are supported, not {data_array.ndim}D."
        )

    return fig


def update_xarray_plot(fig, data_array):
    logger.info(f"Updating plot with xarray data: {data_array.name}")

    if data_array.ndim == 1:
        logger.debug("Updating 1D line plot")
        if not fig.data:
            fig.add_trace(go.Scatter())
        fig.update_traces(x=data_array.coords[data_array.dims[0]], y=data_array.values)
    elif data_array.ndim == 2:
        logger.debug("Updating 2D heatmap")
        if not fig.data:
            fig.add_trace(go.Heatmap())
        fig.update_traces(
            z=data_array.values,
            x=data_array.coords[data_array.dims[1]],
            y=data_array.coords[data_array.dims[0]],
        )
    else:
        logger.error(f"Unsupported array dimensions: {data_array.ndim}")
        raise ValueError(
            f"Only 1D and 2D arrays are supported, not {data_array.ndim}D."
        )

    return fig

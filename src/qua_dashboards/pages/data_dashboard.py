import time
import traceback
from typing import Any, Optional
import xarray as xr
import dash
from dash import Dash, html, dcc, State, Input, Output, callback, get_app
from dash.development.base_component import Component
import dash_bootstrap_components as dbc
from qua_dashboards.data_dashboard.component_types.xarray_component import (
    create_dataset_component,
)
from qua_dashboards.logging_config import logger
from flask import request, jsonify
import matplotlib.pyplot as plt

from qua_dashboards.utils.data_utils import deserialise_data, serialise_data
from qua_dashboards.utils.dash_utils import convert_to_dash_component
from dash.dependencies import MATCH
from qua_dashboards.data_dashboard.component_types import (
    create_data_array_component,
    create_standard_component,
    create_image_component,
)
from qua_dashboards.logging_config import logger

logger.info("Registering page data-dashboard")


dash.register_page(__name__)

update_interval = 1000
include_title = True
update_button = False

layout = html.Div(
    [
        html.Div(id="data-container", children=[]),
        dcc.Interval(
            id="interval-component", interval=update_interval, n_intervals=0
        ),
    ],
    style={"margin": "10px"},
)

update_button = update_button
if update_button:
    layout.children.insert(0, dbc.Button("Update", id="update-button"))
if include_title:
    layout.children.insert(0, html.H1("Data Visualizer", id="title"))


_data_dashboard_requires_update: bool = False
_data_dashboard_data = {}
_data_dashboard_collapse_button_clicks = {}

## setup callbacks
logger.info("Setting up callbacks for data-dashboard")

inputs = [Input("interval-component", "n_intervals")]
if update_button:
    inputs.append(Input("update-button", "n_clicks"))


@callback(
    [Output("data-container", "children")],
    inputs,
    [State("data-container", "children")],
)
def update_if_required(*args):
    global _data_dashboard_requires_update
    global _data_dashboard_data
    # if not update_button: args == [n_intervals, current_children]
    # if update_button: args == [n_intervals, n_clicks, current_children]
    current_children = args[-1]

    if not _data_dashboard_requires_update:
        raise dash.exceptions.PreventUpdate
    _data_dashboard_requires_update = False

    t0 = time.perf_counter()

    current_children = convert_to_dash_component(current_children)

    logger.info(f"Updating data-container children")

    current_children_dict = {child.id: child for child in current_children}

    children = []
    for key, value in _data_dashboard_data.items():
        child = value_to_dash_component(
            label=key,
            value=value,
            existing_component=current_children_dict.get(key),
        )
        children.append(child)

    print(f"Update taken: {time.perf_counter() - t0:.2f} seconds")
    return (children,)


@callback(
    Output({"type": "collapse", "index": MATCH}, "is_open"),
    [Input({"type": "collapse-button", "index": MATCH}, "n_clicks")],
    [State({"type": "collapse", "index": MATCH}, "is_open")],
)
def toggle_collapse(n_clicks, is_open):
    global _data_dashboard_collapse_button_clicks
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open

    # Get the id of the component that triggered the callback
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Check if the button's n_clicks has increased
    previous_clicks = _data_dashboard_collapse_button_clicks.get(triggered_id, 0)
    if n_clicks and n_clicks != previous_clicks:
        _data_dashboard_collapse_button_clicks[triggered_id] = n_clicks
        if is_open:
            logger.debug(f"Closing collapse: {triggered_id} with {n_clicks=}")
        else:
            logger.debug(f"Opening collapse: {triggered_id} with {n_clicks=}")
        return not is_open

    return is_open

def value_to_dash_component(
    label: str, value: Any, existing_component: Optional[Component] = None
):
    if isinstance(value, xr.DataArray):
        return create_data_array_component(
            label=label,
            value=value,
            existing_component=existing_component,
        )
    elif isinstance(value, xr.Dataset):
        return create_dataset_component(
            label=label,
            value=value,
            existing_component=existing_component,
        )
    elif isinstance(value, plt.Figure):
        image_base64 = serialise_data(value)
        return create_image_component(
            label=label,
            image_base64=image_base64,
            existing_component=existing_component,
        )
    elif isinstance(value, str) and value.startswith("data:image/png;base64,"):
        return create_image_component(
            label=label,
            image_base64=value,
            existing_component=existing_component,
        )
    else:
        return create_standard_component(
            label=label,
            value=value,
            existing_component=existing_component,
        )



def update_data(data):
    global _data_dashboard_data
    global _data_dashboard_requires_update

    _data_dashboard_data = data
    _data_dashboard_requires_update = True

## setup papi
app = get_app()
@app.server.route("/data-dashboard/update-data", methods=["POST"])
def update_data_endpoint():
    serialised_data = request.json
    data = deserialise_data(serialised_data)
    update_data(data)
    return jsonify(success=True)

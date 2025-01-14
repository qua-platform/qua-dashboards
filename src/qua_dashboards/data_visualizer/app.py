import time
from typing import Any, Optional
import xarray as xr
import dash
from dash import Dash, html, dcc, State, Input, Output
import dash_bootstrap_components as dbc
from qua_dashboards.logging_config import logger
from flask import request, jsonify

from qua_dashboards.utils.data_utils import deserialise_data
from qua_dashboards.utils.dash_utils import convert_to_dash_component
from dash.dependencies import MATCH
from qua_dashboards.data_visualizer.component_types import (
    create_data_array_component,
    create_standard_component,
)


class DataVisualizerApp:
    def __init__(
        self,
        update_interval: int = 100,
        title: str = "Data Visualizer",
        include_title: bool = True,
        update_button: bool = False,
    ):
        self.app = Dash(
            __name__,
            title=title,
            assets_folder="../assets",
            external_stylesheets=[dbc.themes.BOOTSTRAP],
        )
        logger.info("Dash app initialized")

        self.app.layout = html.Div(
            [
                dbc.ListGroup(id="data-container", children=[], flush=True),
                dcc.Interval(
                    id="interval-component", interval=update_interval, n_intervals=0
                ),
            ]
        )

        self.update_button = update_button
        if update_button:
            self.app.layout.children.insert(0, dbc.Button("Update", id="update-button"))
        if include_title:
            self.app.layout.children.insert(0, html.H1("Data Visualizer", id="title"))

        self._collapse_button_clicks = {}

        self._requires_update: bool = False
        self.data = {}
        self.setup_callbacks()
        self.setup_api()

    def run(self, threaded: bool = False):
        self.app.run(debug=True, threaded=threaded)

    def setup_callbacks(self, app=None):
        if app is None:
            app = self.app

        logger.info("Setting up callbacks for data-visualizer")

        inputs = [Input("interval-component", "n_intervals")]
        if self.update_button:
            inputs.append(Input("update-button", "n_clicks"))

        @app.callback(
            [Output("data-container", "children")],
            inputs,
            [State("data-container", "children")],
        )
        def update_if_required(*args):
            # if not self.update_button: args == [n_intervals, current_children]
            # if self.update_button: args == [n_intervals, n_clicks, current_children]
            current_children = args[-1]

            if not self._requires_update:
                raise dash.exceptions.PreventUpdate
            self._requires_update = False

            t0 = time.perf_counter()

            current_children = convert_to_dash_component(current_children)

            logger.info(f"Updating data-container children")

            current_children_dict = {child.id: child for child in current_children}

            children = []
            for key, value in self.data.items():
                child = self.value_to_dash_component(
                    label=key,
                    value=value,
                    existing_component=current_children_dict.get(key),
                )
                children.append(child)

            print(f"Update taken: {time.perf_counter() - t0:.2f} seconds")
            return (children,)

        @app.callback(
            Output({"type": "collapse", "index": MATCH}, "is_open"),
            [Input({"type": "collapse-button", "index": MATCH}, "n_clicks")],
            [State({"type": "collapse", "index": MATCH}, "is_open")],
        )
        def toggle_collapse(n_clicks, is_open):
            ctx = dash.callback_context
            if not ctx.triggered:
                return is_open

            # Get the id of the component that triggered the callback
            triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

            # Check if the button's n_clicks has increased
            previous_clicks = self._collapse_button_clicks.get(triggered_id, 0)
            if n_clicks and n_clicks != previous_clicks:
                self._collapse_button_clicks[triggered_id] = n_clicks
                if is_open:
                    logger.debug(f"Closing collapse: {triggered_id} with {n_clicks=}")
                else:
                    logger.debug(f"Opening collapse: {triggered_id} with {n_clicks=}")
                return not is_open

            return is_open

    @staticmethod
    def value_to_dash_component(
        label: str, value: Any, existing_component: Optional[dbc.ListGroupItem] = None
    ):
        if isinstance(value, xr.DataArray):
            return create_data_array_component(
                label=label,
                value=value,
                existing_component=existing_component,
                root_component_class=dbc.ListGroupItem,
            )
        else:
            return create_standard_component(
                label=label,
                value=value,
                existing_component=existing_component,
                root_component_class=dbc.ListGroupItem,
            )

    def update_data(self, data):
        self.data = data
        self._requires_update = True

    def setup_api(self, app=None):
        if app is None:
            app = self.app
        server = app.server

        @server.route("/data-visualizer/update-data", methods=["POST"])
        def update_data_endpoint():
            serialised_data = request.json
            data = deserialise_data(serialised_data)
            self.update_data(data)
            return jsonify(success=True)


if __name__ == "__main__":
    app = DataVisualizerApp()
    app.run(threaded=True)

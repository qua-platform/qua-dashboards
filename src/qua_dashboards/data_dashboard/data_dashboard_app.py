import time
from typing import Optional

import dash
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, State, Input, Output
from dash.development.base_component import Component
from dash.dependencies import MATCH
from flask import request, jsonify

from qua_dashboards.logging_config import logger
from qua_dashboards.utils import deserialise_data, convert_to_dash_component
from qua_dashboards.data_dashboard.data_components import (
    DataArrayComponent,
    DatasetComponent,
    ImageComponent,
    StandardComponent,
)


class DataDashboardApp:
    def __init__(
        self,
        update_interval: int = 500,
        title: str = "Data Dashboard",
        include_title: bool = False,
        update_button: bool = False,
        app: Optional[Dash] = None,
    ):
        self.component_types = [
            DatasetComponent,
            DataArrayComponent,
            ImageComponent,
            StandardComponent,
        ]
        self.layout = html.Div(
            [
                html.Div(id="data-container", children=[]),
                dcc.Interval(
                    id="interval-component", interval=update_interval, n_intervals=0
                ),
                dcc.Store(id="toggle-store", data=""),
            ],
            style={"margin": "10px"},
        )

        if app is None:
            self.app = Dash(
                __name__,
                title=title,
                assets_folder="../assets",
                external_stylesheets=[dbc.themes.BOOTSTRAP],
            )
            self.app.layout = self.layout
        else:
            self.app = app
        logger.info("Dash app initialized")

        self.update_button = update_button
        if update_button:
            self.app.layout.children.insert(0, dbc.Button("Update", id="update-button"))
        if include_title:
            self.app.layout.children.insert(0, html.H1("Data Dashboard", id="title"))

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

        logger.info("Setting up callbacks for data-dashboard")

        inputs = [Input("interval-component", "n_intervals")]
        if self.update_button:
            inputs.append(Input("update-button", "n_clicks"))

        @app.callback(
            [Output("toggle-store", "data"), Output("data-container", "children")],
            inputs,
            [State("data-container", "children")],
        )
        def update_if_required(*args):
            current_children = args[-1]

            if not self._requires_update:
                raise dash.exceptions.PreventUpdate
            self._requires_update = False

            t0 = time.perf_counter()

            current_children = convert_to_dash_component(current_children)

            logger.info("Updating data-container children")

            current_children_dict = {
                child.id.replace("data-entry-", ""): child
                for child in current_children
                if hasattr(child, "id") and child.id is not None
            }

            children = []
            for key, value in self.data.items():
                child = self.create_component(
                    label=key,
                    value=value,
                    existing_component=current_children_dict.get(key),
                )
                children.append(child)

            logger.info(f"Data update took: {time.perf_counter() - t0:.2f} seconds")
            return (
                "update",
                children,
            )

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

        app.clientside_callback(
            """
        function(store_data) {
            console.log("store_data", store_data)
        }
        """,
            Output("dummy-output", "children"),
            Input("toggle-store", "data"),
        )

        # Register callbacks for each component type
        for component_type in self.component_types:
            component_type.register_callbacks(app)

    def create_component(
        self, label: str, value: any, existing_component: Optional[Component] = None
    ) -> Component:
        for component_type in self.component_types:
            if component_type.can_handle(value):
                return component_type.create_component(
                    label=label, value=value, existing_component=existing_component
                )

    def update_data(self, data):
        self.data = data
        self._requires_update = True

    def setup_api(self, app=None):
        if app is None:
            app = self.app
        server = app.server

        @server.route("/data-dashboard/update-data", methods=["POST"])
        def update_data_endpoint():
            serialised_data = request.json
            data = deserialise_data(serialised_data)
            self.update_data(data)
            return jsonify(success=True)


def main():
    app = DataDashboardApp(update_interval=1000)
    app.run(threaded=True)


if __name__ == "__main__":
    main()

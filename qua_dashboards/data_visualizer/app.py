import xarray as xr
import dash
from dash import html, dcc
from qua_dashboards.logging_config import logger
from flask import request, jsonify
from plotly import graph_objects as go
from qua_dashboards.data_visualizer.plotting import update_xarray_plot
from qua_dashboards.data_visualizer.data_serialisation import deserialise_data
from qua_dashboards.utils.dash_utils import convert_to_dash_component
from dash_extensions.enrich import (
    DashProxy,
    Output,
    Input,
    State,
    BlockingCallbackTransform,
)


class DataVisualizer:
    def __init__(self):
        self.app = DashProxy(
            __name__,
            title="Data Visualizer",
            transforms=[BlockingCallbackTransform(timeout=10)],
        )
        self.server = self.app.server  # Access the Flask server
        logger.info("Dash app initialized")

        self.app.layout = html.Div(
            [
                html.H1("Data Visualizer"),
                html.Div(id="data-container", children=[]),
                dcc.Interval(id="interval-component", interval=1000, n_intervals=0),
                html.Button("Update data", id="update-data-button"),
            ]
        )

        self._requires_update: bool = False
        self.data = {}
        self.setup_callbacks()
        self.setup_api()

    def run(self, threaded: bool = False):
        self.app.run(debug=True, threaded=threaded)

    def setup_callbacks(self):
        @self.app.callback(
            [Output("data-container", "children")],
            [Input("interval-component", "n_intervals")],
            # [Input("update-data-button", "n_clicks")],
            [State("data-container", "children")],
        )
        def update_if_required(n_intervals, current_children):
            # def update_if_required(n_clicks, current_children):
            # if not n_clicks:
            #     raise dash.exceptions.PreventUpdate

            if not self._requires_update:
                raise dash.exceptions.PreventUpdate

            current_children = convert_to_dash_component(current_children)

            print("Updating data-container children")

            current_children_dict = {child.id: child for child in current_children}

            children = []
            for key, value in self.data.items():
                if key in current_children_dict:
                    child = current_children_dict[key]
                else:
                    child = html.Div(id=key)

                self.update_div_value(child, value)
                children.append(child)
            return (children,)

    @staticmethod
    def update_div_value(div, value):
        if isinstance(value, xr.DataArray):
            if (
                div.children is None
                or len(div.children) != 1
                or not isinstance(div.children[0], dcc.Graph)
            ):
                fig = go.Figure()
                graph = dcc.Graph(figure=fig)
                logger.info("Plotting new xarray plot")
                div.children = [graph]
            else:
                logger.info("Updating xarray plot")
                graph = div.children[0]
                fig = graph.figure
            update_xarray_plot(fig, value)
        else:
            div.children = [html.P(str(value))]

    def update_data(self, data):
        self.data = data
        self._requires_update = True

    def setup_api(self):
        @self.server.route("/update-data", methods=["POST"])
        def update_data_endpoint():
            serialised_data = request.json
            data = deserialise_data(serialised_data)
            self.update_data(data)
            return jsonify(success=True)


if __name__ == "__main__":
    app = DataVisualizer()
    app.run(threaded=True)

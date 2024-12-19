import time
import xarray as xr
import dash
from dash import Dash, html, dcc, State, Input, Output
from qua_dashboards.logging_config import logger
from flask import request, jsonify
from plotly import graph_objects as go
from qua_dashboards.data_visualizer.plotting import update_xarray_plot
from qua_dashboards.data_visualizer.data_serialisation import deserialise_data
from qua_dashboards.utils.dash_utils import convert_to_dash_component

# from dash_extensions.enrich import (
#     DashProxy,
#     Output,
#     Input,
#     State,
#     BlockingCallbackTransform,
# )

GRAPH_STYLE = {"aspect-ratio": "1 / 1", "max-width": "800px"}


class DataVisualizer:
    def __init__(self):
        self.app = Dash(
            __name__,
            title="Data Visualizer",
        )
        self.server = self.app.server  # Access the Flask server
        logger.info("Dash app initialized")

        self.app.layout = html.Div(
            [
                html.H1("Data Visualizer"),
                html.Div(id="data-container", children=[]),
                dcc.Interval(id="interval-component", interval=200, n_intervals=0),
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
            t0 = time.perf_counter()

            current_children = convert_to_dash_component(current_children)

            print("Updating data-container children")

            current_children_dict = {child.id: child for child in current_children}

            children = []
            for key, value in self.data.items():
                if key in current_children_dict:
                    child = current_children_dict[key]
                else:
                    child = html.Div(id=key)

                self.update_div_value(child, value, title=key)
                children.append(child)

            self._requires_update = False
            print(f"Update taken: {time.perf_counter() - t0:.2f} seconds")
            return (children,)

    @staticmethod
    def update_div_value(div, value, title=None):
        if isinstance(value, xr.DataArray):
            if div.children is None:
                div.children = []

            try:
                graph = next(
                    child for child in div.children if isinstance(child, dcc.Graph)
                )
            except StopIteration:
                fig = go.Figure(layout=dict(margin=dict(l=20, r=20, t=20, b=20)))
                graph = dcc.Graph(figure=fig, style=GRAPH_STYLE)
                logger.info("Plotting new xarray plot")
                div.children = [graph]
                if title:
                    div.children.insert(0, html.H2(title))

            logger.info("Updating xarray plot")
            update_xarray_plot(graph.figure, value)
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

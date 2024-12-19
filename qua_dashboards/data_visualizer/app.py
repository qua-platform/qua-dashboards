import time
from typing import Any
import xarray as xr
import dash
from dash import Dash, html, dcc, State, Input, Output
import dash_bootstrap_components as dbc
from qua_dashboards.logging_config import logger
from flask import request, jsonify
from plotly import graph_objects as go
from qua_dashboards.data_visualizer.plotting import update_xarray_plot
from qua_dashboards.utils.data_serialisation import deserialise_data
from qua_dashboards.utils.dash_utils import convert_to_dash_component

# from dash_extensions.enrich import (
#     DashProxy,
#     Output,
#     Input,
#     State,
#     BlockingCallbackTransform,
# )

GRAPH_STYLE = {"aspect-ratio": "1 / 1", "max-width": "400px"}


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
                dcc.Interval(id="interval-component", interval=2000, n_intervals=0),
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
            [State("data-container", "children")],
        )
        def update_if_required(n_intervals, current_children):
            if not self._requires_update:
                raise dash.exceptions.PreventUpdate
            t0 = time.perf_counter()

            current_children = convert_to_dash_component(current_children)

            print(f"Updating data-container children:\n{current_children}")

            current_children_dict = {child.id: child for child in current_children}

            children = []
            for key, value in self.data.items():
                child = self.value_to_dash_component(
                    label=key, value=value, component=current_children_dict.get(key)
                )
                children.append(child)

            self._requires_update = False
            print(f"Update taken: {time.perf_counter() - t0:.2f} seconds")
            return (children,)

    @staticmethod
    def value_to_dash_component(label: str, value: Any, component=None):
        if isinstance(value, xr.DataArray):
            if component is None:
                component = html.Div(id=label, children=[])

            if (
                len(component.children) != 2
                or not isinstance(component.children[0], html.Button)
                or not isinstance(component.children[1], dcc.Graph)
            ):
                title_component = html.Button(label)
                fig = go.Figure(layout=dict(margin=dict(l=20, r=20, t=20, b=20)))
                graph = dcc.Graph(figure=fig, style=GRAPH_STYLE)
                component.children = [title_component, graph]

            title_component, graph = component.children

            logger.info("Updating xarray plot")
            update_xarray_plot(graph.figure, value)
        else:
            component = html.Div([html.P([html.B(f"{label}: "), str(value)])], id=label)

        return component

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

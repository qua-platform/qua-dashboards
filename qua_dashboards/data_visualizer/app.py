import xarray as xr
import dash
from dash import State, html, dcc, Input, Output
from qua_dashboards.logging_config import logger
from flask import request, jsonify
from qua_dashboards.data_visualizer.plotting import (
    plot_xarray,
    update_xarray_plot,
)
from qua_dashboards.data_visualizer.data_serialisation import deserialise_data


class DataVisualizer:
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.server = self.app.server  # Access the Flask server
        logger.info("Dash app initialized")

        self.app.layout = html.Div(
            [
                html.H1("Data Visualizer"),
                html.Button("Click me", id="my-button", n_clicks=0),
                html.Div(id="data-container", children=[html.Label("Hello")]),
                dcc.Interval(id="interval-component", interval=1000, n_intervals=0),
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
            Output("data-container", "children"),
            # [Input("interval-component", "n_intervals")],
            [Input("my-button", "n_clicks")],
            [State("data-container", "children")],
        )
        def update_if_required(n_clicks, current_children):
            logger.info(f"{current_children=}")
            if n_clicks == 0:
                return current_children
            if not self._requires_update:
                return current_children
            elif any(isinstance(child, dict) for child in current_children):
                # Why is it sometimes a dict?
                print("Warning: data-container children is a dict")
                current_children = [
                    html.Div(**child["props"]) for child in current_children
                ]
                # return current_children

            logger.debug(f"{current_children=}")

            print("Updating data-container children")
            # new_item = html.Div(f"Item ")
            # return current_children + [new_item]

            current_children_dict = {child.id: child for child in current_children}

            children = []
            for key, value in self.data.items():
                if key in current_children_dict:
                    child = current_children_dict[key]
                else:
                    child = html.Div(id=key)

                self.update_div_value(child, value)
                current_children.append(child)

            # TODO REMOVEME
            current_children[0]["props"]["children"] = "Bye"
            return current_children

    @staticmethod
    def update_div_value(div, value):
        if isinstance(value, xr.DataArray):
            if (
                div.children is None
                or len(div.children) != 1
                or not isinstance(div.children[0], dcc.Graph)
            ):
                fig = plot_xarray(value)
                div.children = [dcc.Graph(figure=fig)]
            else:
                update_xarray_plot(div.children[0], value)
        else:
            # div.children = [html.P(str(value))]
            pass

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

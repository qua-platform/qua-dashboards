import time
from typing import Any, Optional
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
from dash.dependencies import MATCH

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
            external_stylesheets=[dbc.themes.BOOTSTRAP],
        )
        self.server = self.app.server  # Access the Flask server
        logger.info("Dash app initialized")

        self.app.layout = html.Div(
            [
                html.H1("Data Visualizer"),
                dbc.ListGroup(id="data-container", children=[], flush=True),
                dcc.Interval(id="interval-component", interval=2000, n_intervals=0),
            ]
        )

        self._collapse_button_clicks = {}

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

            print(f"Updating data-container children")  #:\n{current_children}")

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

        @self.app.callback(
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
        label: str, value: Any, component: Optional[dbc.ListGroupItem] = None
    ):
        if component is None:
            component = dbc.ListGroupItem(
                id=label,
                children=[
                    dbc.Button(label, id={"type": "collapse-button", "index": label}),
                    dbc.Collapse(
                        [str(value)],
                        id={"type": "collapse", "index": label},
                        is_open=True,
                    ),
                ],
            )

        assert len(component.children) == 2
        title, collapse_component = component.children
        assert isinstance(title, dbc.Button)
        assert isinstance(collapse_component, dbc.Collapse)

        if isinstance(value, xr.DataArray):
            if not isinstance(collapse_component.children[0], dcc.Graph):
                fig = go.Figure(layout=dict(margin=dict(l=20, r=20, t=20, b=20)))
                graph = dcc.Graph(
                    figure=fig,
                    style=GRAPH_STYLE,
                )
                collapse_component.children = [graph]
            else:
                graph = collapse_component.children[0]

            logger.info("Updating xarray plot")
            update_xarray_plot(graph.figure, value)
        else:
            collapse_component.children = [str(value)]

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

import numpy as np
import dash
from dash import State, html, dcc, Input, Output
from dash.dash import time
from qua_dashboards.logging_config import logger


class DataVisualizer:
    def __init__(self):
        self.app = dash.Dash(__name__)
        logger.info("Dash app initialized")

        self.app.layout = html.Div(
            [
                html.H1("Data Visualizer"),
                html.Div(id="data-container", children=[]),
                # dcc.Store(id="data-store", storage_type="session"),
                dcc.Interval(id="interval-component", interval=1000, n_intervals=0),
            ]
        )

        self._requires_update: bool = False
        self.data = {}
        self.setup_callbacks()

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
                return (current_children,)

            children = []
            for key, value in self.data.items():
                try:
                    child = next(
                        child for child in current_children if child["id"] == key
                    )
                except StopIteration:
                    child = html.Div(id=key)

                child.children = [html.Label(str(value))]
                children.append(child)

            return (children,)

    def update_data(self, data):
        self.data = data
        self._requires_update = True


if __name__ == "__main__":
    app = DataVisualizer()
    app.run(threaded=True)

    while True:
        data = {"current_time": time.time()}
        if np.random.rand() > 0.5:
            data["random_value"] = np.random.rand()

        app.update_data(data)
        time.sleep(1)

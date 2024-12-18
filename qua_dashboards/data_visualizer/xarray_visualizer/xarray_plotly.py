import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotting import plot_xarray, update_plot_figure
from data_generation import generate_random_xarray
from qua_dashboards.logging_config import logger
from typing import Optional
import xarray


class XarrayPlotlyDashboard:
    def __init__(self):
        # Initialize the Dash app
        self.app = dash.Dash(__name__)
        logger.info("Dash app initialized")

        # Initialize dataset
        self.dataset: Optional[xarray.Dataset] = None

        # Define the app layout
        self.app.layout = html.Div(
            [
                html.H1("Xarray Plotly Dashboard"),
                dcc.Dropdown(id="array-selector", style={"display": "none"}),
                dcc.Graph(id="xarray-plot"),
                html.Button("Generate New Data", id="generate-button", n_clicks=0),
            ]
        )

        # Set up callbacks
        self.setup_callbacks()

    def setup_callbacks(self):
        @self.app.callback(
            [
                Output("xarray-plot", "figure"),
                Output("array-selector", "options"),
                Output("array-selector", "style"),
                Output("array-selector", "value"),
            ],
            [Input("generate-button", "n_clicks")],
            [State("xarray-plot", "figure"), State("array-selector", "options")],
        )
        def generate_new_dataset(n_clicks, current_figure, current_options):
            logger.info("Generating new dataset")
            self.dataset = generate_random_xarray()
            options = [{"label": var, "value": var} for var in self.dataset.data_vars]
            logger.debug(f"Created dropdown options: {options}")
            fig = plot_xarray(self.dataset[list(self.dataset.data_vars)[0]])

            return (
                fig,
                options,
                {"display": "block" if len(options) > 1 else "none"},
                list(self.dataset.data_vars)[0],
            )

        @self.app.callback(
            [Output("xarray-plot", "figure")],
            [Input("array-selector", "value")],
            [State("xarray-plot", "figure")],
        )
        def update_plot(n_clicks, selected_array, current_figure):
            logger.info("Changing array to selected")
            fig = go.Figure(current_figure)

            if selected_array is None or selected_array not in self.dataset.data_vars:
                selected_array = list(self.dataset.data_vars)[0]
                logger.info(f"Selected default array: {selected_array}")
            else:
                logger.info(f"Using selected array: {selected_array}")

            fig = update_plot_figure(fig, self.dataset[selected_array])

            return fig

    def run(self, threaded=False):
        logger.info("Starting Dash server")
        self.app.run_server(debug=True, threaded=threaded)


if __name__ == "__main__":
    dashboard = XarrayPlotlyDashboard()
    dashboard.run()

    # for k in range(10):
    #     import time

    #     time.sleep(1)
    #     logger.info("Updating dataset")

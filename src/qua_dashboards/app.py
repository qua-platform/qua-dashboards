import dash
from dash import Dash, html
from qua_dashboards.logging_config import logger
from qua_dashboards.pages import dashboards_registry
import dash_bootstrap_components as dbc


app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)


for key, dashboard in dashboards_registry.items():
    logger.info(f"Setting up dashboard: {key}")
    dashboard.setup_api(app=app)
    dashboard.setup_callbacks(app=app)


app.layout = html.Div(
    [
        dash.page_container,
    ]
)


if __name__ == "__main__":
    app.run(debug=True)

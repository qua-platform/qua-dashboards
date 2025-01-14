import dash
from dash import Dash, html, dcc
from qua_dashboards.logging_config import logger
from qua_dashboards.pages import dashboards_registry


app = Dash(__name__, use_pages=True)


for key, dashboard in dashboards_registry.items():
    logger.info(f"Setting up dashboard: {key}")
    dashboard.setup_api(app=app)
    dashboard.setup_callbacks(app=app)


app.layout = html.Div(
    [
        html.H1("Multi-page app with Dash Pages"),
        html.Div(
            [
                html.Div(
                    dcc.Link(
                        f"{page['name']} - {page['path']}", href=page["relative_path"]
                    )
                )
                for page in dash.page_registry.values()
            ]
        ),
        dash.page_container,
    ]
)

if __name__ == "__main__":
    app.run(debug=True)

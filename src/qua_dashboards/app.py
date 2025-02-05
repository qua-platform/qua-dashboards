from typing import Optional

import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc


def create_app(requests_pathname_prefix: Optional[str] = None) -> Dash:
    kwargs = {}
    if requests_pathname_prefix is not None:
        kwargs["requests_pathname_prefix"] = requests_pathname_prefix
    app = Dash(
        __name__,
        use_pages=True,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        **kwargs,
    )

    # TODO remove
    dash.register_page(
        "another_home", layout=html.Div("We're home!"), path="/other-path"
    )
    dash.register_page(
        "very_important", layout=html.Div("Don't miss it!"), path="/important", order=0
    )

    app.layout = html.Div()
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)

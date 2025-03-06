from typing import Optional

from dash import Dash, html
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

    app.layout = html.Div()
    return app


def main():
    app = create_app()
    app.run(debug=False)


if __name__ == "__main__":
    main()

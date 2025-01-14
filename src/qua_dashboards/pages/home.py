import dash
from dash import html, dcc

dash.register_page(__name__, path="/")


def layout(**kwags):
    layout = html.Div(
        [
            html.H1("Available Dashboards"),
            html.Div(
                [
                    html.Div(
                        dcc.Link(
                            page["name"],
                            href=page["relative_path"],
                        )
                    )
                    for page in dash.page_registry.values()
                    if page["path"] != "/"
                ]
            ),
        ]
    )

    return layout

from dash import Dash, html, Input, Output, State

app = Dash(__name__)

app.layout = html.Div(
    [
        html.Button("Click me", id="my-button", n_clicks=0),
        html.Div(id="my-div", children=[html.Div("No clicks yet")]),
    ]
)


@app.callback(
    Output("my-div", "children"),
    [Input("my-button", "n_clicks")],
    [State("my-div", "children")],
)
def append_to_div(n_clicks, current_children):
    if n_clicks > 0:
        print(f"Button clicked {n_clicks} times")
        print(f"{current_children=}")

        if isinstance(current_children[0], dict):
            print(f"Warning: current_children entries are dicts: {current_children}")
        else:
            print(f"current_children entries are not dicts: {current_children}")

        return current_children + [html.Div(f"Item {n_clicks}")]
    return current_children


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)

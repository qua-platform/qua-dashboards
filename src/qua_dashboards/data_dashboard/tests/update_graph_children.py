import numpy as np
from dash import Dash, html, Input, Output, State, dcc

import plotly.graph_objects as go

# Create a minimal 1D line figure using Plotly Graph Objects
fig = go.Figure()
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Line"))

app = Dash(__name__)

app.layout = html.Div(
    [
        html.Button("Click me", id="my-button", n_clicks=0),
        html.Div(
            id="my-div", children=[html.Div(dcc.Graph(id="my-graph", figure=fig))]
        ),
    ]
)


@app.callback(
    Output("my-div", "children"),
    [Input("my-button", "n_clicks")],
    [State("my-div", "children")],
)
def append_to_div(n_clicks, current_children):
    # current_children = [dcc.Graph(**current_children[0]["props"])]
    print("\n" * 10 + f"{current_children=}")
    # fig = go.Figure(**current_children[0]["props"]["figure"])

    # fig.add_trace(
    #     go.Scatter(
    #         x=[0, 1], y=[np.random.rand(), np.random.rand()], mode="lines", name="Line"
    #     )
    # )
    # current_children[0]["props"]["figure"] = fig
    return current_children


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)

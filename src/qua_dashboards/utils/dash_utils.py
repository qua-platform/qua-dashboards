from typing import Union
from dash import dcc, html
from dash.dash import Any, warnings
import dash_bootstrap_components as dbc
from plotly import graph_objects as go


def convert_to_dash_component(elem: Union[dict, list]) -> Any:
    if isinstance(elem, list):
        return [convert_to_dash_component(e) for e in elem]
    elif isinstance(elem, dict):
        if "namespace" not in elem:
            return {
                key: convert_to_dash_component(value) for key, value in elem.items()
            }

        if elem["namespace"] == "dash_html_components":
            cls = getattr(html, elem["type"])
            if "children" in elem["props"]:
                children = elem["props"].pop("children")
                elem["props"]["children"] = convert_to_dash_component(children)
            return cls(**elem["props"])

        if elem["namespace"] == "dash_bootstrap_components":
            cls = getattr(dbc, elem["type"])
            if "children" in elem["props"]:
                children = elem["props"].pop("children")
                elem["props"]["children"] = convert_to_dash_component(children)
            return cls(**elem["props"])

        if elem["type"] == "Graph":
            if isinstance(elem["props"].get("figure"), dict):
                elem["props"]["figure"] = go.Figure(**elem["props"]["figure"])
            return dcc.Graph(**elem["props"])
        raise ValueError(f"Unknown element: {elem}")
    else:
        return elem


def create_input_field(
    id,
    label,
    value,
    debounce=True,
    input_style=None,
    div_style=None,
    units=None,
    **kwargs,
):
    if input_style is None:
        input_style = {"width": "80px"}

    elements = [
        dbc.Col(
            dbc.Label(
                f"{label}:",
                html_for=id,
                className="mr-2",
                style={"white-space": "nowrap"},
            ),
            width="auto",
        ),
        dbc.Col(
            dbc.Input(
                id=id,
                type="number",
                value=value,
                debounce=debounce,
                style=input_style,
                **kwargs,
            ),
            width="auto",
        ),
    ]
    if units is not None:
        elements.append(dbc.Col(dbc.Label(units, className="ml-2"), width="auto"))

    return dbc.Row(
        elements,
        className="align-items-center mb-2",
        style=div_style,
    )

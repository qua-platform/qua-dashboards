from typing import Union
from dash import dcc, html
from dash.dash import Any, warnings
from plotly import graph_objects as go


def convert_to_dash_component(elem: Union[dict, list]) -> Any:
    if isinstance(elem, list):
        return [convert_to_dash_component(e) for e in elem]
    elif isinstance(elem, dict):
        if "type" not in elem:
            return elem
        component_type = elem["type"]
        if component_type == "Div":
            if isinstance(elem["props"].get("children"), (dict, list)):
                elem["props"]["children"] = convert_to_dash_component(
                    elem["props"]["children"]
                )
            return html.Div(**elem["props"])
        elif component_type == "Graph":
            if isinstance(elem["props"].get("figure"), dict):
                elem["props"]["figure"] = go.Figure(**elem["props"]["figure"])
            return dcc.Graph(**elem["props"])
        else:
            warnings.warn(f"Unknown component type: {component_type}")
            return elem
    else:
        return elem

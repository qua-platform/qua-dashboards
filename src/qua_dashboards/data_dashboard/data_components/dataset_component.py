import xarray as xr
from typing import Type, TypeVar, Optional, Any, List, Dict
from dash import html, dcc
import dash
from dash.dependencies import Output, Input, State, MATCH
import dash_bootstrap_components as dbc
from qua_dashboards.logging_config import logger
from qua_dashboards.data_dashboard.data_components.data_array_component import (
    DataArrayComponent,
)
from qua_dashboards.utils import convert_to_dash_component

T = TypeVar("T", bound=html.Div)


class DatasetComponent:
    """
    A component for visualizing an xarray Dataset.

    Provides two view modes:
      - Tabbed view: Each variable is shown in its own tab.
      - All view: Variables are stacked vertically.

    A toggle (dbc.RadioItems) allows switching views. If the displayed variables
    have not changed, the existing component is reused, preserving state.
    """

    _datasets = {}  # Stores datasets keyed by label.
    _callback_registered = False

    @classmethod
    def can_handle(cls, value: any) -> bool:
        return isinstance(value, xr.Dataset)

    @classmethod
    def _validate_existing_component(
        cls, component: html.Div, dataset: xr.Dataset, root_component_class: Type[T]
    ) -> bool:
        if not isinstance(component, root_component_class):
            return False
        if getattr(component, "data-class", None) != "xarray_dataset_component":
            return False
        try:
            # Expect two children: a toggle and a content container.
            content = component.children[1]
            if content.children is None:
                return False
            # Unwrap if content.children is an html.Div wrapping the actual view.
            actual_children = content.children

            displayed_vars: List[str] = []
            if isinstance(actual_children, dcc.Tabs):
                for tab in actual_children.children:
                    if hasattr(tab, "label"):
                        displayed_vars.append(tab.label)
            elif isinstance(actual_children, list):
                for child in actual_children:
                    # Expect the id to be like "data-entry-variableName"
                    if (
                        hasattr(child, "id")
                        and isinstance(child.id, str)
                        and child.id.startswith("data-entry-")
                    ):
                        displayed_vars.append(child.id.replace("data-entry-", ""))
                    else:
                        try:
                            displayed_vars.append(child.children[0].children)
                        except Exception as e:
                            logger.error(
                                f"Error extracting variable name in all-view: {e}"
                            )
            else:
                return False
            expected_vars = list(dataset.data_vars.keys())
            return displayed_vars == expected_vars
        except Exception as e:
            logger.error(f"Error in _validate_existing_component: {e}")
            return False

    @classmethod
    def create_component(
        cls,
        label: str,
        value: xr.Dataset,
        existing_component: Optional[html.Div] = None,
        root_component_class: Type[T] = html.Div,
    ) -> T:
        """
        Create or update a component for an xarray Dataset.

        If an existing component passes validation, it is updated.
        """
        cls._datasets[label] = value
        if existing_component is None or not cls._validate_existing_component(
            existing_component, value, root_component_class
        ):
            root_component = root_component_class(
                id=f"data-entry-{label}",
                children=[
                    dbc.RadioItems(
                        id={"type": "dataset-view-toggle", "index": label},
                        options=[
                            {"label": "Tabbed", "value": "tabbed"},
                            {"label": "Show All", "value": "all"},
                        ],
                        value="tabbed",
                        inline=True,
                        style={
                            "display": "flex",
                            "justifyContent": "center",
                            "marginBottom": "10px",
                        },
                        input_checked_style={
                            "backgroundColor": "#007bff",
                            "borderColor": "#007bff",
                        },
                        input_style={"marginRight": "10px"},
                        label_style={
                            "display": "inline-block",
                            "padding": "5px 10px",
                            "border": "1px solid #007bff",
                            "borderRadius": "4px",
                            "cursor": "pointer",
                        },
                    ),
                    html.Div(id={"type": "dataset-content", "index": label}),
                ],
                **{"data-class": "xarray_dataset_component"},
            )
        else:
            root_component = existing_component

        # Delegate view generation to a dedicated function.
        cls._update_dataset_view(label, value, root_component)
        return root_component

    @classmethod
    def _update_dataset_view(
        cls, label: str, dataset: xr.Dataset, root_component: html.Div
    ) -> None:
        """
        Determine the current view mode and update the dataset content accordingly.
        """
        # Get current view mode from the toggle.
        current_mode = "tabbed"
        for child in root_component.children:
            if (
                isinstance(child, dbc.RadioItems)
                and isinstance(child.id, dict)
                and child.id.get("type") == "dataset-view-toggle"
            ):
                current_mode = child.value or "tabbed"
                break

        # Retrieve existing view from content container, if any.
        existing_view = None
        for child in root_component.children:
            if (
                isinstance(child, html.Div)
                and isinstance(child.id, dict)
                and child.id.get("type") == "dataset-content"
            ):
                existing_view = child.children
                break

        # Generate new content based on the current mode.
        if current_mode == "tabbed":
            content = cls._generate_tabbed_view(
                label, dataset, existing_tabs=existing_view
            )
        else:
            content = cls._generate_all_view(
                label, dataset, existing_all_view=existing_view
            )
        # Update the content container.
        for child in root_component.children:
            if (
                isinstance(child, html.Div)
                and isinstance(child.id, dict)
                and child.id.get("type") == "dataset-content"
            ):
                child.children = content
                break

    @classmethod
    def _generate_tabbed_view(
        cls, label: str, dataset: xr.Dataset, existing_tabs: Optional[Any] = None
    ) -> Any:
        """
        Generate a tabbed view for the dataset.
        Each tab displays one variable using DataArrayComponent.
        Reuse an existing data array component if present.
        """
        existing_mapping: Dict[str, Any] = {}
        if (
            existing_tabs is not None
            and hasattr(existing_tabs, "children")
            and isinstance(existing_tabs, dcc.Tabs)
        ):
            for tab in existing_tabs.children:
                if hasattr(tab, "label"):
                    var = tab.label
                    if tab.children and len(tab.children) > 0:
                        existing_mapping[var] = tab.children[0]
        tabs = []
        for var_name, data_array in dataset.data_vars.items():
            existing_da = existing_mapping.get(var_name)
            # Use only the variable name as label (not composite)
            da_component = DataArrayComponent.create_component(
                var_name, data_array, existing_component=existing_da
            )
            tab = dcc.Tab(
                label=var_name,
                children=[da_component],
                style={"padding": "6px", "fontWeight": "normal"},
                selected_style={"padding": "6px", "fontWeight": "bold"},
            )
            tabs.append(tab)
        return dcc.Tabs(children=tabs)

    @classmethod
    def _generate_all_view(
        cls, label: str, dataset: xr.Dataset, existing_all_view: Optional[Any] = None
    ) -> Any:
        """
        Generate an "all view" that stacks variables vertically.
        Each variable's component is wrapped in an html.Div.
        Reuse existing data array components if present.
        """
        existing_mapping: Dict[str, Any] = {}
        if existing_all_view is not None and isinstance(existing_all_view, list):
            for child in existing_all_view:
                if (
                    hasattr(child, "id")
                    and isinstance(child.id, str)
                    and child.id.startswith("data-entry-")
                ):
                    var = child.id.replace("data-entry-", "")
                    if child.children:
                        existing_mapping[var] = child.children
        components: List[Any] = []
        for var_name, data_array in dataset.data_vars.items():
            existing_da = existing_mapping.get(var_name)
            da_component = DataArrayComponent.create_component(
                var_name, data_array, existing_component=existing_da
            )
            comp = html.Div(
                da_component,
                id=f"data-entry-{var_name}",
                style={"marginBottom": "20px"},
            )
            components.append(comp)
        return html.Div(components)

    @classmethod
    def register_callbacks(cls, app) -> None:
        """
        Register a callback that updates the dataset view when toggled.
        """
        if cls._callback_registered:
            return

        @app.callback(
            Output({"type": "dataset-content", "index": MATCH}, "children"),
            Input({"type": "dataset-view-toggle", "index": MATCH}, "value"),
            State({"type": "dataset-content", "index": MATCH}, "children"),
            State({"type": "dataset-content", "index": MATCH}, "id"),
        )
        def update_dataset_view(view_mode, existing_content, content_id):
            # Only trigger if the radio toggle was actually clicked.
            ctx = dash.callback_context
            if (
                not ctx.triggered
                or "dataset-view-toggle" not in ctx.triggered[0]["prop_id"]
            ):
                return dash.no_update

            label = content_id["index"]
            # If existing content exists, try to rearrange it.
            if existing_content is not None:
                existing_content = convert_to_dash_component(existing_content)
                if view_mode == "tabbed":
                    # If already in tabbed view, return as is.
                    if isinstance(existing_content, dcc.Tabs):
                        return existing_content
                    # Otherwise, wrap the existing children into tabs.
                    tabs = []
                    for child in existing_content.children:
                        # Assume each child has an id like "data-entry-variableName"
                        var_name = child.id.replace("data-entry-", "")
                        tab = dcc.Tab(
                            label=var_name,
                            children=[child],
                            style={"padding": "6px", "fontWeight": "normal"},
                            selected_style={"padding": "6px", "fontWeight": "bold"},
                        )
                        tabs.append(tab)
                    return dcc.Tabs(children=tabs)
                elif view_mode == "all":
                    # If already in "all" view (i.e. not wrapped in dcc.Tabs), just return as is.
                    if not isinstance(existing_content, dcc.Tabs):
                        return existing_content
                    # Otherwise, convert the tab structure into a vertical layout.
                    components = []
                    for tab in existing_content.children:
                        if tab.children:
                            components.append(
                                html.Div(
                                    tab.children[0],
                                    id=f"data-entry-{tab.label}",
                                    style={"marginBottom": "20px"},
                                )
                            )
                    return html.Div(components)

            # Fallback: if there’s no existing content, use the stored dataset.
            dataset = DatasetComponent._datasets.get(label)
            if dataset is None:
                return html.Div("No data available")
            if view_mode == "tabbed":
                return DatasetComponent._generate_tabbed_view(label, dataset)
            elif view_mode == "all":
                return DatasetComponent._generate_all_view(label, dataset)

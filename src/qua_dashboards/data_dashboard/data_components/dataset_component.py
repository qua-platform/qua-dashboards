import xarray as xr
from typing import Type, TypeVar, Optional, Any, List
from dash import html, dcc, Input, Output, State, MATCH
from dash.development.base_component import Component
import dash_bootstrap_components as dbc
from qua_dashboards.logging_config import logger
from qua_dashboards.data_dashboard.data_components.data_array_component import (
    DataArrayComponent,
)

T = TypeVar("T", bound=Component)


class DatasetComponent:
    """
    A component for visualizing an xarray Dataset.

    Provides two view modes:
      - Tabbed view: Each data variable is shown in its own tab.
      - All view: All data variables are displayed in a vertically stacked layout.

    A toggle control (using dbc.RadioItems) allows the user to switch between these views.
    If the displayed variables have not changed, the existing component is reused
    so that user selections (like the current view mode or active tab) are preserved.
    """

    _datasets = {}  # Stores datasets keyed by label.
    _callback_registered = False

    @classmethod
    def can_handle(cls, value: any) -> bool:
        return isinstance(value, xr.Dataset)

    @classmethod
    def _validate_existing_component(
        cls, component: Component, dataset: xr.Dataset, root_component_class: Type[T]
    ) -> bool:
        """
        Validate that an existing dataset component is compatible with the provided dataset.
        Checks that:
          - The component is an instance of the expected root_component_class.
          - It has a custom attribute "data-class" equal to "xarray_dataset_component".
          - The variable names currently displayed (from the content container)
            match the keys in dataset.data_vars.
        """
        if not isinstance(component, root_component_class):
            return False
        if getattr(component, "data-class", None) != "xarray_dataset_component":
            return False

        try:
            # The component layout is assumed to have two children:
            # 1. The toggle (dbc.RadioItems) and 2. the content container.
            content = component.children[1]
            if content.children is None:
                return False
            displayed_vars: List[str] = []
            # If the view is "tabbed", content.children is a dcc.Tabs component.
            if isinstance(content.children, dcc.Tabs):
                tabs = content.children.children
                # Extract each tab's label.
                displayed_vars = [tab.label for tab in tabs if hasattr(tab, "label")]
            # If the view is "all", content.children is an html.Div containing a list.
            elif isinstance(content.children, list):
                for child in content.children:
                    # Assume each child wraps a DataArrayComponent which was created with label = var_name.
                    # Here we try to extract the variable name from a custom id or from the first child text.
                    if hasattr(child, "id") and child.id:
                        # For example, if DataArrayComponent sets id like "data-entry-{var_name}"
                        displayed_vars.append(child.id.replace("data-entry-", ""))
                    else:
                        try:
                            # As fallback, assume the first child's children contains the name.
                            displayed_vars.append(child.children[0].children)
                        except Exception as e:
                            logger.error(
                                f"Error extracting variable name in all-view: {e}"
                            )
            else:
                return False

            expected_vars = list(dataset.data_vars.keys())
            # For reusability, the order must match.
            return displayed_vars == expected_vars
        except Exception as e:
            logger.error(f"Error in _validate_existing_component: {e}")
            return False

    @classmethod
    def create_component(
        cls,
        label: str,
        value: xr.Dataset,
        existing_component: Optional[Component] = None,
        root_component_class: Type[T] = html.Div,
    ) -> T:
        """
        Create or update a component for an xarray Dataset.

        The layout consists of:
          - A toggle control (using dbc.RadioItems) for switching between "tabbed" and "all" views.
          - A content container (an html.Div) that displays the plots.

        If an existing component passes validation (i.e. the variable names have not changed),
        it is updated rather than recreated.
        """
        cls._datasets[label] = value

        if existing_component is None or not cls._validate_existing_component(
            existing_component, value, root_component_class
        ):
            # Create a new component.
            root_component = root_component_class(
                [
                    dbc.RadioItems(
                        id={"type": "dataset-view-toggle", "index": label},
                        options=[
                            {"label": "Tabbed", "value": "tabbed"},
                            {"label": "Show All", "value": "all"},
                        ],
                        value="tabbed",
                        inline=True,
                        # Custom styling to mimic a polished left/right toggle.
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
                ]
            )
            root_component.__setattr__("data-class", "xarray_dataset_component")
        else:
            root_component = existing_component

        # Preserve the current toggle value.
        current_mode = "tabbed"
        for child in root_component.children:
            if (
                isinstance(child, dbc.RadioItems)
                and isinstance(child.id, dict)
                and child.id.get("type") == "dataset-view-toggle"
            ):
                current_mode = child.value or "tabbed"
                break

        if current_mode == "tabbed":
            content = cls._generate_tabbed_view(label, value)
        else:
            content = cls._generate_all_view(label, value)

        # Update the content container (the second child).
        for child in root_component.children:
            if (
                isinstance(child, html.Div)
                and isinstance(child.id, dict)
                and child.id.get("type") == "dataset-content"
            ):
                child.children = content

        return root_component

    @classmethod
    def _generate_tabbed_view(cls, label: str, dataset: xr.Dataset) -> Any:
        """
        Generate a tabbed view (using dcc.Tabs) for the dataset.
        Each tab displays one data variable using DataArrayComponent.
        """
        tabs = []
        for var_name, data_array in dataset.data_vars.items():
            da_component = DataArrayComponent.create_component(var_name, data_array)
            tab = dcc.Tab(
                label=var_name,
                children=[da_component],
                style={"padding": "6px", "fontWeight": "normal"},
                selected_style={"padding": "6px", "fontWeight": "bold"},
            )
            tabs.append(tab)
        return dcc.Tabs(children=tabs)

    @classmethod
    def _generate_all_view(cls, label: str, dataset: xr.Dataset) -> Any:
        """
        Generate an "all view" that displays every data variable in the dataset
        in a stacked vertical layout.
        """
        components: List[Any] = []
        for var_name, data_array in dataset.data_vars.items():
            da_component = DataArrayComponent.create_component(var_name, data_array)
            components.append(html.Div(da_component, style={"marginBottom": "20px"}))
        return html.Div(components)

    @classmethod
    def register_callbacks(cls, app) -> None:
        """
        Register a callback that updates the dataset content container when the toggle changes.
        """
        if not cls._callback_registered:

            @app.callback(
                Output({"type": "dataset-content", "index": MATCH}, "children"),
                Input({"type": "dataset-view-toggle", "index": MATCH}, "value"),
                State({"type": "dataset-content", "index": MATCH}, "id"),
            )
            def update_dataset_view(view_mode, content_id):
                label = content_id["index"]
                dataset = cls._datasets.get(label)
                if dataset is None:
                    logger.error(f"No dataset stored for label {label}")
                    return html.Div("No data available")
                if view_mode == "tabbed":
                    return cls._generate_tabbed_view(label, dataset)
                elif view_mode == "all":
                    return cls._generate_all_view(label, dataset)
                else:
                    return html.Div("Unknown view mode")

            cls._callback_registered = True

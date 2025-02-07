from typing import Type, TypeVar, Optional, Any
from dash.development.base_component import Component
from dash import html
import dash_bootstrap_components as dbc
from qua_dashboards.logging_config import logger
from .base_data_component import BaseDataComponent

T = TypeVar("T", bound=Component)


class StandardComponent(BaseDataComponent):
    """
    A fallback component that can display any value as a simple label.

    This component creates a basic layout with a label and the string representation
    of the value. It serves as the default component when no other specialized
    components can handle the value type.
    """

    @classmethod
    def can_handle(cls, value: Any) -> bool:
        """
        This component can handle any value type.

        Args:
            value (Any): The value to check.

        Returns:
            bool: Always returns True as this is the fallback component.
        """
        return True

    @classmethod
    def create_component(
        cls,
        label: str,
        value: Any,
        existing_component: Optional[Component] = None,
        root_component_class: Type[T] = html.Div,
    ) -> T:
        """
        Create or update a standard component displaying a label and value.

        Args:
            label (str): The label for the component.
            value (Any): Any value that will be converted to string for display.
            existing_component (Optional[Component]): An existing component to update,
                if available. Defaults to None.
            root_component_class (Type[T]): The class to use for the root component.
                Defaults to html.Div.

        Returns:
            T: The created or updated component.
        """
        value_str = str(value)
        if len(value_str) > 200:
            value_str = value_str[:200] + "..."

        if not cls._validate_existing_component(
            existing_component, root_component_class
        ):
            logger.info(f"Creating new standard component ({label}: {value_str})")
            root_component = root_component_class(
                id=f"data-entry-{label}",
                children=[
                    dbc.Label(label, style={"fontWeight": "bold"}),
                    dbc.Label(":  ", style={"whiteSpace": "pre"}),
                    dbc.Label(value_str),
                ],
                **{"data-class": "standard_component"},
            )
        else:
            logger.info(f"Using existing standard component ({label}: {value_str})")
            root_component = existing_component
            # Update the value label
            value_component = root_component.children[-1]
            value_component.children = value_str

        return root_component

    @staticmethod
    def _validate_existing_component(
        component: Component, root_component_class: Type[T]
    ) -> bool:
        """
        Validate that an existing component matches the expected structure.

        Args:
            component (Component): The component to validate.
            root_component_class (Type[T]): The expected class of the root component.

        Returns:
            bool: True if the component is valid and can be reused, False otherwise.
        """
        if not isinstance(component, root_component_class):
            return False
        if not getattr(component, "data-class", None) == "standard_component":
            return False
        return True

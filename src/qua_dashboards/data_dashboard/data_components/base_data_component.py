from abc import ABC, abstractmethod
from typing import Optional, Type, TypeVar
from dash.development.base_component import Component
import dash_bootstrap_components as dbc

T = TypeVar("T", bound=Component)


class BaseDataComponent(ABC):
    """Base class for dashboard components that can validate and create Dash components."""

    @classmethod
    @abstractmethod
    def can_handle(cls, value: any) -> bool:
        """Check if this component can handle the given value type."""
        pass

    @classmethod
    @abstractmethod
    def create_component(
        cls,
        label: str,
        value: any,
        existing_component: Optional[Component] = None,
        root_component_class: Type[T] = None,
    ) -> Component:
        """Create or update a Dash component for the given value."""
        pass

    @staticmethod
    def create_collapsible_root_component(
        label: str, root_component_class: Type[T], data_class: str
    ) -> T:
        return root_component_class(
            id=f"data-entry-{label}",
            children=[
                dbc.Label(
                    label,
                    id={"type": "collapse-button", "index": label},
                    style={"fontWeight": "bold"},
                ),
                dbc.Collapse(
                    [],
                    id={"type": "collapse", "index": label},
                    is_open=True,
                ),
            ],
            **{"data-class": data_class},
        )

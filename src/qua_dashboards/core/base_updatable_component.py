from abc import ABC
from enum import Flag, auto

from qua_dashboards.core.base_component import BaseComponent

from dash import html
from typing import Any, Dict, List


__all__ = ["ModifiedFlags", "BaseUpdatableComponent"]


class ModifiedFlags(Flag):
    """Flags indicating what needs to be modified after parameter changes."""

    NONE = 0
    PARAMETERS_MODIFIED = auto()
    PROGRAM_MODIFIED = auto()
    CONFIG_MODIFIED = auto()


class BaseUpdatableComponent(BaseComponent, ABC):
    def __init__(self, *args, component_id: str, **kwargs):
        assert not args, (
            "BaseUpdatableComponent does not accept any positional arguments"
        )
        assert not kwargs, (
            "BaseUpdatableComponent does not accept any keyword arguments"
        )

        self.component_id = component_id

    def update_parameters(self, parameters: Dict[str, Dict[str, Any]]) -> ModifiedFlags:
        """Update the component's attributes based on the input values."""
        return ModifiedFlags.NONE

    def get_dash_components(self, include_subcomponents: bool = True) -> List[html.Div]:
        """Return a list of Dash components.

        Args:
            include_subcomponents (bool, optional): Whether to include subcomponents.
            Defaults to True.

        Returns:
            List[html.Div]: A list of Dash components.
        """
        return []

    def get_component_ids(self) -> List[str]:
        """Return a list of component IDs for this component including subcomponents."""
        return [self.component_id]

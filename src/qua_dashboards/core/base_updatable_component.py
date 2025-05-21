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
    """
    Extends BaseComponent for components with UI-updatable parameters.

    This class provides a standard interface for components whose parameters
    can be modified through the Dash UI, allowing for dynamic updates and
    communication of what aspects of the component were affected by these
    changes (e.g., if a program needs regeneration or a configuration
    needs to be reloaded).
    """

    def __init__(self, *args: Any, component_id: str, **kwargs: Any) -> None:
        """
        Initializes the BaseUpdatableComponent.

        Args:
            component_id: A unique string identifier for this component instance.
        """
        # Pass through component_id and any other args/kwargs to BaseComponent
        super().__init__(component_id=component_id, *args, **kwargs)  # type: ignore

    def update_parameters(self, parameters: Dict[str, Dict[str, Any]]) -> ModifiedFlags:
        """
        Update the component's attributes based on the input values from the UI.

        Subclasses should implement this method to handle changes to their
        specific parameters. The `parameters` dictionary is typically structured
        with the component's own ID as a key.

        Args:
            parameters: A dictionary where keys are component IDs (matching
                `self.component_id` for this component's parameters) and values are
                dictionaries of parameter names to their new values.
                Example: {'my-component-id': {'param_a': 10, 'param_b': 'value'}}

        Returns:
            ModifiedFlags indicating what aspects of the component were changed
            and might require further action (e.g., re-generating a program).
            Defaults to ModifiedFlags.NONE.
        """
        return ModifiedFlags.NONE

    def get_dash_components(self, include_subcomponents: bool = True) -> List[html.Div]:
        """
        Return a list of Dash components representing the UI for editing parameters.

        Subclasses should override this to provide the specific input fields
        or controls for their manageable parameters.

        Args:
            include_subcomponents (bool, optional): Whether to include UI components
                from sub-components that this component might manage. Defaults to True.

        Returns:
            A list of Dash html.Div or similar components. Defaults to an empty list.
        """
        return []

    def get_components(self) -> List["BaseUpdatableComponent"]:
        """
        Return a list of component instances for this component's parameter UIs.

        These components are used in Dash callbacks to manage and update parameter
        values. Typically, this will be `[self]`, but can include managed sub-components
        if `include_subcomponents` was True in `get_dash_components`.

        Returns:
            A list of component instances. Defaults to `[self]`.
        """
        return [self]

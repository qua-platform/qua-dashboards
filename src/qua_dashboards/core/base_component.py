import abc
from typing import Any, Dict

from dash import Dash
from dash.development.base_component import Component


class BaseComponent(abc.ABC):
    """
    Abstract Base Class for pluggable dashboard components.

    Defines the interface required for components to be used with the 
    dashboard builder.
    """

    def __init__(
        self, 
        component_params: Dict[str, Any], 
        shared_objects: Dict[str, Any]
    ):
        """
        Initialize the component.

        Args:
            component_params: Dictionary of parameters specific to this component's configuration.
            shared_objects: Dictionary of objects potentially shared between components 
                (e.g., instrument instances, QMM).
        """
        self.params = component_params
        self.shared = shared_objects
        # You might add common initialization here if needed later

    @abc.abstractmethod
    def get_layout(self) -> Component:
        """
        Generate the Dash layout for this component.

        Returns:
            A Dash component representing the UI for this component.
        """
        pass

    @abc.abstractmethod
    def register_callbacks(self, app: Dash, **kwargs):
        """
        Register all necessary Dash callbacks for this component's interactivity.

        Args:
            app: The Dash application instance.
            **kwargs: Optional keyword arguments, potentially used for passing
                      enhancement details from other components.
        """
        pass
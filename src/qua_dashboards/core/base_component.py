import abc
import logging
from typing import Dict, Optional, Any

from dash import Dash

# Import the base Component type for type hinting return values
from dash.development.base_component import Component

logger = logging.getLogger(__name__)


__all__ = ["BaseComponent"]


class BaseComponent(abc.ABC):
    """
    Abstract Base Class for pluggable dashboard components.

    Defines the essential interface required for components to be integrated
    into a dashboard built by a dashboard builder. Each component instance
    must have a unique ID assigned during initialization, which is used
    for namespacing its internal Dash elements.
    """

    def __init__(self, *args: Any, component_id: str, **kwargs: Any) -> None:
        """
        Initialize the base component.

        Subclasses MUST call `super().__init__(component_id=component_id)` within their
        own `__init__` method if they override this. Positional and keyword
        arguments are generally discouraged for direct BaseComponent initialization
        to promote clarity through named arguments in subclasses.

        Args:
            component_id: A unique string identifier for this component instance.
                This ID is crucial for generating namespaced IDs for internal
                Dash elements to prevent collisions in a multi-component
                dashboard. It should be unique across all components
                instantiated within a single dashboard.
        """
        # While *args and **kwargs are used for signature flexibility for subclasses,
        # this direct base class initialization path asserts against their use
        # to guide towards cleaner subclass __init__ signatures.
        if args:
            logger.warning(
                f"BaseComponent ({component_id}): Positional arguments {args} were "
                "passed during __init__ but are not used by BaseComponent itself. "
                "Ensure subclasses handle them if they are intended for the subclass."
            )
        if kwargs:
            logger.warning(
                f"BaseComponent ({component_id}): Keyword arguments {kwargs} were "
                "passed during __init__ but are not used by BaseComponent itself. "
                "Ensure subclasses handle them if they are intended for the subclass."
            )

        if not isinstance(component_id, str) or not component_id:
            raise ValueError("component_id must be a non-empty string")

        self.component_id: str = component_id
        self.layout_columns: int = 12  # Default, can be overridden by subclasses
        self.layout_rows: int = 8  # Default, can be overridden by subclasses
        logger.debug(
            f"Initializing BaseComponent with component_id='{self.component_id}'"
        )

    def _get_id(self, element_name: str) -> Dict[str, str]:
        """
        Generates a namespaced Dash component ID (pattern-matching dict).

        This helper ensures that IDs generated within this component instance
        are unique within the larger dashboard application by incorporating the
        component's unique `component_id`. Use this for all internal elements
        that need an ID (inputs, outputs, stores, graph, buttons, etc.).

        Args:
            element_name: A descriptive name for the element within this component
                (e.g., 'update-button', 'main-graph', 'data-store').

        Returns:
            A dictionary suitable for use as a Dash component ID, using the
            recommended pattern-matching structure.
            Example: {'type': 'comp-my_video_mode', 'index': 'main-graph'}
        """
        return {"type": f"comp-{self.component_id}", "index": element_name}

    @abc.abstractmethod
    def get_layout(self) -> Optional[Component]:
        """
        Generate the Dash layout for this component instance.

        This method must be implemented by subclasses. It should return a single
        Dash component (e.g., html.Div, dbc.Container) that represents the
        entire UI structure for this component. All internal elements requiring
        an ID within this layout MUST use IDs generated via `self._get_id()`.

        Returns:
            A Dash component instance, or None if the component has no layout.
        """
        pass

    @abc.abstractmethod
    def register_callbacks(self, app: Dash, **kwargs: Any) -> None:
        """
        Register all necessary Dash callbacks for this component's interactivity.

        This method must be implemented by subclasses. Callbacks defined here
        should use Inputs, Outputs, and States referencing component IDs
        generated via `self._get_id()`.

        Args:
            app: The main Dash application instance to which callbacks will be added.
            **kwargs: Additional keyword arguments that specific orchestrators or
                builders might pass, such as shared stores or interface objects
                to other components. Subclasses can choose to use these or ignore them.
        """
        pass

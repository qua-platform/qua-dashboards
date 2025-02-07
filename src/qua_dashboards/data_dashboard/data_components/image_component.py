from typing import Type, TypeVar, Optional
from dash.development.base_component import Component
from dash import html
import matplotlib.pyplot as plt
from qua_dashboards.logging_config import logger
from .base_data_component import BaseDataComponent

T = TypeVar("T", bound=Component)


class ImageComponent(BaseDataComponent):
    """
    A component for displaying images in the dashboard.

    This component can handle both matplotlib figures and base64-encoded image strings.
    It creates a collapsible div containing an image element that can be updated
    with new image data.
    """

    @classmethod
    def can_handle(cls, value: any) -> bool:
        """
        Check if the value can be handled by this component.

        Args:
            value (any): The value to check.

        Returns:
            bool: True if the value is either a matplotlib Figure or a base64-encoded
                 PNG image string, False otherwise.
        """
        return isinstance(value, plt.Figure) or (
            isinstance(value, str) and value.startswith("data:image/png;base64,")
        )

    @classmethod
    def create_component(
        cls,
        label: str,
        value: any,
        existing_component: Optional[Component] = None,
        root_component_class: Type[T] = html.Div,
    ) -> T:
        """
        Create or update a component for visualizing an image.

        Args:
            label (str): The label for the component.
            value (Union[plt.Figure, str]): Either a matplotlib figure or a
                base64-encoded image string to display.
            existing_component (Optional[Component]): An existing component to update,
                if available. Defaults to None.
            root_component_class (Type[T]): The class to use for the root component.
                Defaults to html.Div.

        Returns:
            T: The created or updated component.
        """
        # Validate and create/reuse component
        if not cls._validate_existing_component(
            existing_component, root_component_class
        ):
            logger.info(f"Creating new image component ({label})")
            root_component = cls.create_collapsible_root_component(
                label, root_component_class, "image_component"
            )
            # Initialize collapse component with empty image
            collapse_component = root_component.children[1]
            collapse_component.children = [
                html.Img(
                    id={"type": "image", "index": label},
                    style={"max-width": "100%"},  # Make image responsive
                )
            ]
        else:
            logger.info(f"Using existing image component ({label})")
            root_component = existing_component

        # Update the image source with new data
        collapse_component = root_component.children[1]
        image_component = collapse_component.children[0]
        # Convert matplotlib figure to base64 if necessary
        image_component.src = (
            value if isinstance(value, str) else cls._serialize_figure(value)
        )

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
        if not getattr(component, "data-class", None) == "image_component":
            return False
        return True

    @staticmethod
    def _serialize_figure(fig: plt.Figure) -> str:
        """
        Convert a matplotlib figure to a base64-encoded PNG string.

        Args:
            fig (plt.Figure): The matplotlib figure to convert.

        Returns:
            str: A base64-encoded PNG string with the appropriate data URI prefix.
        """
        import io
        import base64

        # Save figure to a temporary buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)

        # Convert buffer contents to base64 string
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

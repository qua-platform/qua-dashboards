from typing import Type, TypeVar, Optional
from dash.development.base_component import Component
from dash import html
import dash_bootstrap_components as dbc
from qua_dashboards.logging_config import logger

T = TypeVar("T", bound=Component)


def _create_root_component(
    label: str, root_component_class: Type[T], data_class: str
) -> T:
    """
    Create the root component structure with a label and a collapsible image section.

    Args:
        label (str): The label for the component.
        root_component_class (Type[T]): The class type for the root component.

    Returns:
        T: An instance of the root component class.
    """
    return root_component_class(
        id=f"data-entry-{label}",
        children=[
            dbc.Label(
                label,
                id={"type": "collapse-button", "index": label},
                style={"fontWeight": "bold"},
            ),
            dbc.Collapse(
                [
                    html.Img(
                        id={"type": "image", "index": label},
                        style={"max-width": "100%"},
                    ),
                ],
                id={"type": "collapse", "index": label},
                is_open=True,
            ),
        ],
        **{"data-class": data_class},
    )


def _validate_image_component(
    component: Component, root_component_class: Type[T]
) -> bool:
    """
    Validate the structure of a component to ensure it matches the expected layout.

    Args:
        component (Component): The component to validate.
        root_component_class (Type[T]): The expected class type for the root component.

    Returns:
        bool: True if the component structure is valid, False otherwise.
    """
    if not isinstance(component, root_component_class):
        return False
    if not getattr(component, "data-class", None) == "image_component":
        return False
    return True


def create_image_component(
    label: str,
    image_base64: str,
    existing_component: Optional[Component] = None,
    root_component_class: Type[T] = html.Div,
) -> T:
    """
    Create or update a component for visualizing a base64-encoded image.

    Args:
        label (str): The label for the component.
        image_base64 (str): The base64-encoded image string.
        existing_component (Optional[Component]): An existing component to update, if any.
        root_component_class (Type[T]): The class type for the root component.

    Returns:
        T: The created or updated component.
    """
    if not _validate_image_component(existing_component, root_component_class):
        logger.info(f"Creating new image component ({label})")
        root_component = _create_root_component(
            label, root_component_class, "image_component"
        )
    else:
        logger.info(f"Using existing image component ({label})")
        root_component = existing_component

    # Update the image source
    collapse_component = root_component.children[1]
    image_component = collapse_component.children[0]
    image_component.src = image_base64

    return root_component

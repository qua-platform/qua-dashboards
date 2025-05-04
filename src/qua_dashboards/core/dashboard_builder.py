import logging
from typing import Sequence

from dash import Dash, html
import dash_bootstrap_components as dbc
from qua_dashboards.core.base_component import BaseComponent

# --- Logger Setup ---

logger = logging.getLogger(__name__)


# --- Dashboard Builder Function ---


def build_dashboard(
    components: Sequence[BaseComponent], title: str = "Dashboard"
) -> Dash:
    """
    Builds a Dash application instance from a list of component objects using
    simple Bootstrap layout for multiple components.

    Args:
        components: List of objects adhering to the DashboardComponent protocol.
        title: The title for the Dash application.

    Returns:
        A configured Dash application instance.
    """
    app = Dash(
        __name__,
        title=title,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
    )

    if not components:
        logger.warning("No components provided to build_dashboard.")
        app.layout = html.Div("No components were provided for this dashboard.")
        return app

    # --- Determine Layout Strategy ---
    if len(components) == 1:
        logger.info("Using layout from single provided component.")
        # If only one component, use its layout directly
        app.layout = components[0].get_layout()
    else:
        # Multiple components: arrange them in columns within a row
        logger.info(f"Arranging {len(components)} components using Bootstrap Row/Col.")
        num_comps = len(components)
        # Basic equal width distribution, ensuring at least width 1
        col_width = max(1, 12 // num_comps)

        component_layouts = [comp.get_layout() for comp in components]

        app.layout = dbc.Container(
            [
                dbc.Row(
                    [
                        # Create a column for each component's layout
                        dbc.Col(layout, width=col_width)
                        for layout in component_layouts
                    ],
                    # Add gutters (spacing) between columns if desired,
                    # e.g., className="g-3"
                    className="g-3",
                )
            ],
            fluid=True,  # Use full width of the viewport
            # Add margin/padding to the container if desired
            style={"padding": "15px"},
        )

    # --- Register Callbacks ---
    logger.debug(f"Registering callbacks for {len(components)} components.")
    for comp in components:
        try:
            comp.register_callbacks(app)
        except Exception as e:
            comp_name = type(comp).__name__
            logger.error(
                f"Error registering callbacks for component {comp_name}: {e}",
                exc_info=True,
            )

    logger.info("Dashboard build complete.")
    return app

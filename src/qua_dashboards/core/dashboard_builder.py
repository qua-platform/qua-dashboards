# qua_dashboards/core/dashboard_builder.py
import dash_dynamic_grid_layout as dgl
from dash import Dash, html
import dash_bootstrap_components as dbc
from qua_dashboards.core import BaseComponent
from typing import Sequence, List, Dict, Any
import logging
from dash import dcc

__all__ = ["build_dashboard"]

logger = logging.getLogger(__name__)


def build_dashboard(
    components: Sequence[BaseComponent],
    shared_stores: Sequence[dcc.Store] = None,  # Allow passing shared stores
    initial_grid_layout: List[Dict[str, Any]] = None,  # For defining initial positions
    title: str = "Dashboard",
) -> Dash:
    app = Dash(
        __name__,
        title=title,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        assets_folder="../assets",
    )

    if not components:
        logger.warning(msg="No components provided to build_dashboard.")
        app.layout = html.Div("No components were provided for this dashboard.")
        return app

    draggable_children = []
    # Default layout if none provided
    default_item_layout = []
    for i, comp in enumerate(components):
        # Each component's layout is wrapped.
        # Ensure component_id is accessible and unique for DraggableWrapper key.
        comp_key = (
            f"draggable-{comp.component_id}-{i}"
            if hasattr(comp, "component_id")
            else f"draggable-{i}"
        )
        draggable_children.append(
            dgl.DraggableWrapper(
                children=[comp.get_layout()],
                handleText=comp.component_id,
                id=comp_key,
            )
        )

        # Basic stacking layout if no specific initial_grid_layout is given
        default_item_layout.append(
            {
                "i": comp_key,  # item key
                "x": 0,  # x position
                "y": i * 4,  # y position
                "w": comp.layout_columns,  # width
                "h": comp.layout_rows,  # height
                "minW": 2,  # minimum width
                "minH": 2,  # minimum height
            }
        )

    final_item_layout = (
        initial_grid_layout if initial_grid_layout else default_item_layout
    )

    grid = dgl.DashGridLayout(
        id="main-dashboard-dynamic-grid",
        items=draggable_children,
        rowHeight=100,  # Adjust as needed
        # Defines number of columns in the grid for different screen size breakpoints.
        cols={
            "lg": 12,  # Large screens (1200px+)
            "md": 10,  # Medium screens (996px-1199px)
            "sm": 6,  # Small screens (768px-995px)
            "xs": 4,  # Extra-small screens (480px-767px)
            "xxs": 2,  # Extra-extra-small screens (<480px)
        },
        itemLayout=final_item_layout,
    )

    app_children = [grid]
    if shared_stores:
        app_children.extend(shared_stores)

    app.layout = html.Div(app_children, style={"padding": "15px"})

    logger.debug(f"Registering callbacks for {len(components)} components.")
    for comp in components:
        try:
            # Callbacks now might need access to other component IDs or store IDs
            # This part of the component's contract might need to evolve.
            # For now, assume comp.register_callbacks(app) is self-sufficient
            # or that components get necessary store IDs during their init.
            comp.register_callbacks(app)
        except Exception as e:
            comp_name = type(comp).__name__
            logger.error(
                f"Error registering callbacks for component {comp_name}: {e}",
                exc_info=True,
            )

    logger.info("Dashboard build complete with top-level dynamic grid.")
    return app

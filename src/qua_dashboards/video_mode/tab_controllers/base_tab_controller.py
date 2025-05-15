import abc
from typing import Dict, Any

from dash import Dash
from dash.development.base_component import Component

from qua_dashboards.core import BaseComponent

__all__ = ["BaseTabController"]


class BaseTabController(BaseComponent, abc.ABC):
    """
    Interface for Tab Controllers.

    Each implementation of this interface is responsible for:
    1. Providing the label and unique value for its tab.
    2. Providing the Dash layout for its control panel (to be shown in the left-side tab).
    3. Registering its own callbacks for interactivity within its control panel.
    4. Interacting with the SharedViewerComponent (via orchestrator-provided store IDs)
       to display relevant data or visualizations.
    5. Managing its own local state, especially upon activation/deactivation.
    """

    _TAB_LABEL: str
    _TAB_VALUE: str

    def __init__(self, component_id: str, is_active: bool = False, **kwargs: Any):
        """
        Initializes the ITabController.

        Args:
            component_id: Unique ID for this component instance.
            **kwargs: Additional keyword arguments passed to BaseComponent.
        """
        super().__init__(component_id=component_id, **kwargs)
        self.is_active = is_active

    def get_tab_label(self) -> str:
        """Gets the display label for this tab."""
        return self._TAB_LABEL

    def get_tab_value(self) -> str:
        """Gets the unique value identifier for this tab."""
        return self._TAB_VALUE

    @abc.abstractmethod
    def get_layout(self) -> Component:
        """
        Returns the Dash layout for this tab's control panel.

        This layout will be rendered inside the dcc.Tab on the left side.

        Returns:
            A Dash component representing the layout for this tab.
        """
        pass

    @abc.abstractmethod
    def register_callbacks(
        self,
        app: Dash,
        orchestrator_stores: Dict[str, Any],  # Actual store IDs (dict type)
        shared_viewer_store_ids: Dict[str, Any],  # Actual store IDs (dict type)
        data_acquirer_component_id: str,
        **kwargs: Any,
    ) -> None:
        """
        Registers callbacks specific to this tab controller.

        Args:
            app: The main Dash application.
            orchestrator_stores: IDs of stores managed by the orchestrator
                                 (e.g., LATEST_PROCESSED_DATA_STORE,
                                 VIEWER_DATA_STORE).
            shared_viewer_store_ids: IDs of stores that the SharedViewerComponent
                                     listens to (e.g., VIEWER_DATA_STORE,
                                     VIEWER_LAYOUT_CONFIG_STORE).
            data_acquirer_component_id: The component_id of the data acquirer,
                                        useful for creating pattern-matching IDs
                                        for acquirer's parameters.
            **kwargs: Additional arguments from the orchestrator.
        """
        pass

    @abc.abstractmethod
    def on_tab_activated(self) -> Dict[str, Any]:
        """
        Called by the orchestrator when this tab becomes active.

        The tab should use this to prepare and return commands/data for the
        SharedViewerComponent and the data source mode. This method should
        return a dictionary mapping specific store IDs (conceptual suffixes like
        VideoModeComponent._VIEWER_DATA_STORE_SUFFIX) to their new values.

        If the tab needs access to the latest live data for an action upon activation
        (e.g., to pre-populate a field based on live conditions, NOT for direct display
        if it's a static view tab), it should register its own callback that listens to
        the LATEST_PROCESSED_DATA_STORE.

        Returns:
            A dictionary where keys are conceptual store suffixes (defined in
            VideoModeComponent) and values are the data to be placed in those stores.
            For example:
            {
                VideoModeComponent._VIEWER_DATA_STORE_SUFFIX: {'key': 'live'},
                VideoModeComponent._VIEWER_LAYOUT_CONFIG_STORE_SUFFIX: layout_dict
            }
            or for a live view:
            {
                VideoModeComponent._VIEWER_DATA_STORE_SUFFIX: {'key': 'live'},
            }
        """
        return {}

    @abc.abstractmethod
    def on_tab_deactivated(self) -> None:
        """
        Called by the orchestrator when this tab is no longer active.

        The tab can use this to save any important local volatile state or
        perform cleanup if needed.
        """
        pass

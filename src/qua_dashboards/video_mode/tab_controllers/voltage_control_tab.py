import logging
import uuid
from typing import Any, Dict, Union

import dash_bootstrap_components as dbc
import dash
from dash import Dash, Input, Output, State, html, ctx, ALL, dcc

from qua_dashboards.video_mode.data_acquirers.base_data_acquirer import (
    BaseDataAcquirer,
    ModifiedFlags,
)
import numpy as np
from qua_dashboards.video_mode.tab_controllers.base_tab_controller import (
    BaseTabController,
)
from qua_dashboards.video_mode import data_registry

from qua_dashboards.voltage_control.voltage_control_component import VoltageControlComponent
logger = logging.getLogger(__name__)

__all__ = ["VoltageControlTab"]

class VoltageControlTabController(BaseTabController): 
    """
    Controls the VoltageControl Tab Controller. 

    This tab allows users to change the values of the 
    """
    
    _TAB_LABEL = "Voltage Control View"
    _TAB_VALUE = "voltage-control-tab"


    def __init__(
        self, 
        voltage_control_component: VoltageControlComponent, 
        component_id: str = "voltage-control-tab-controller", 
        is_active: bool = False, 
        **kwargs: Any,
    ) -> None: 
        super().__init__(component_id=component_id, is_active=is_active, **kwargs)
        self.voltage_control_component = voltage_control_component
        self.component_id = component_id

    def get_layout(self) -> dbc.Card: 
        card_body = dbc.CardBody([
            html.H5("Voltage Control", className="card-title text-light"),
            self.voltage_control_component.get_layout()
        ])
        return dbc.Card(
            card_body, color="dark", inverse=True, className="tab-card-dark"
        )
    

    def register_callbacks(self, app: dash.Dash, orchestrator_stores: str, shared_viewer_store_ids: str) -> None:
        return self.voltage_control_component.register_callbacks(app)
    

    def on_tab_activated(self) -> Dict[str, Any]:
        """
        Called when this tab becomes active.
        
        Sets the viewer to continue displaying live data.
        """
        from qua_dashboards.video_mode.video_mode_component import VideoModeComponent
        
        logger.info(f"VoltageControlTabController '{self.component_id}' activated.")
        
        current_live_version = data_registry.get_current_version(
            data_registry.LIVE_DATA_KEY
        )
        
        if current_live_version is None:
            current_live_version = str(uuid.uuid4())
            logger.debug(
                f"{self.component_id}: No current live data version found. "
                f"Using placeholder: {current_live_version}"
            )
        
        viewer_data_payload = {
            "key": data_registry.LIVE_DATA_KEY,
            "version": current_live_version,
        }
        
        updates = {
            VideoModeComponent.VIEWER_DATA_STORE_SUFFIX: viewer_data_payload,
            VideoModeComponent.VIEWER_UI_STATE_STORE_SUFFIX: {},
            VideoModeComponent.VIEWER_LAYOUT_CONFIG_STORE_SUFFIX: {},
        }
        return updates
    
    def on_tab_deactivated(self) -> None:
        """Called when this tab is no longer active."""
        logger.info(f"VoltageControlTabController '{self.component_id}' deactivated.")
        pass

    

    
    




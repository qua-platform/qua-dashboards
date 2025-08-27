import logging
from typing import Any
import dash_bootstrap_components as dbc
from dash import html, ctx, ALL, no_update, Dash, Input, Output

from qua_dashboards.video_mode.tab_controllers.base_tab_controller import (
    BaseTabController,
)

logger = logging.getLogger(__name__)

__all__ = ["LiveViewTabController"]


class SettingsTabController(BaseTabController):
    """
    Controls the 'Settings' tab in the Video Mode application.

    This tab allows the user to adjust the settings of the measurement, such as the readout power and frequency, or the readout parameter (I/Q/R/Phase)
    """

    _TAB_LABEL = "Settings Tab"
    _TAB_VALUE = "settings-tab"
    _DUMMY_OUT_SUFFIX = "dummy-settings-updates"

    def __init__(
        self,
        data_acquirer,
        component_id: str = "settings-tab",
        is_active: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initializes the Settings Tab.

        Args:
            component_id: A unique string identifier for this component instance.
            data_acquirer: The Data Acquirer instance that the measurement uses
            **kwargs: Additional keyword arguments passed to BaseComponent.
        """
        super().__init__(component_id=component_id, is_active=is_active, **kwargs)
        self._data_acquirer_instance = data_acquirer
        logger.info(
            f"Settings Tab '{self.component_id}' initialized with "
            f"Data Acquirer '{self._data_acquirer_instance}'."
        )

    def get_layout(self):
        inner_controls = self._data_acquirer_instance.qua_inner_loop_action.get_dash_components(include_subcomponents=False)
        result_type_selector = dbc.Row(
            [
                dbc.Label("Result Type", width="auto", className="col-form-label"),
                dbc.Col(
                    dbc.Select(
                        id=self._data_acquirer_instance._get_id("result-type"),
                        options=[
                            {"label": rt, "value": rt} for rt in self._data_acquirer_instance.result_types
                        ],
                        value=self._data_acquirer_instance.result_type,
                    ),
                    width=True,
                ),
            ],
            className="mb-2 align-items-center",
        )
        return dbc.Card(
            dbc.CardBody([html.H5("Settings", className="text-light"), *inner_controls, result_type_selector,
                        html.Div(
                        id=self._get_id(self._DUMMY_OUT_SUFFIX), style={"display": "none"}
                    ),]),
            color="dark",
            inverse=True,
            className="tab-card-dark",
        )
    
    def on_tab_activated(self):
        logger.info(f"SettingsTabController '{self.component_id}' activated.")
        return super().on_tab_activated()

    def on_tab_deactivated(self):
        logger.info(f"SettingsTabController '{self.component_id}' deactivated.")
        return super().on_tab_deactivated()
    
    def register_callbacks(self, app: Dash, **kwargs):
        acq = self._data_acquirer_instance
        result_type_id = acq._get_id("result-type")

        @app.callback(
            Output(self._get_id(self._DUMMY_OUT_SUFFIX), "children"),
            Input(result_type_id, "value"),
            prevent_initial_call=True,
        )
        def _update_result_type(new_val):
            if new_val is None:
                return no_update
            acq.update_parameters({acq.component_id: {"result-type": new_val}})
            return no_update
    






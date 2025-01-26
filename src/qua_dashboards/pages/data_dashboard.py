import dash
from qua_dashboards.data_dashboard.app import DataDashboardApp
from qua_dashboards.logging_config import logger

logger.info("Registering page data-dashboard")


dash.register_page(__name__)


data_dashboard_app = DataDashboardApp()
layout = data_dashboard_app.app.layout


from qua_dashboards.pages import dashboards_registry

dashboards_registry["data-dashboard"] = data_dashboard_app

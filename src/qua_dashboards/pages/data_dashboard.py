import dash
from qua_dashboards.logging_config import logger
from qua_dashboards.data_dashboard.data_dashboard_app import DataDashboardApp


logger.info("Registering page data-dashboard")


dash.register_page(__name__)


dashboard_app = DataDashboardApp(app=dash.get_app())

layout = dashboard_app.layout

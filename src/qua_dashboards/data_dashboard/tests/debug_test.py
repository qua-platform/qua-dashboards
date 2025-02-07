from qua_dashboards.data_dashboard.data_dashboard_app import DataDashboardApp
import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

data_dashboard_app = DataDashboardApp(update_interval=60000, update_button=True)
data_dashboard_app.app.run_server(debug=True)

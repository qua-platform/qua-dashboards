import dash
from qua_dashboards.data_visualizer.app import DataVisualizer
from qua_dashboards.logging_config import logger

logger.info("Registering page data-visualizer")


dash.register_page(__name__)


data_visualizer = DataVisualizer()
layout = data_visualizer.app.layout


from qua_dashboards.pages import dashboards_registry

dashboards_registry["data-visualizer"] = data_visualizer

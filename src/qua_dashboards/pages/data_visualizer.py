import dash
from qua_dashboards.data_visualizer.app import DataVisualizer


dash.register_page(__name__)


data_visualizer = DataVisualizer()
layout = data_visualizer.app.layout

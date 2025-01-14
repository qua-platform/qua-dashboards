from qua_dashboards.data_visualizer.app import DataVisualizerApp
import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

data_visualizer = DataVisualizerApp(update_interval=60000, update_button=True)
app = data_visualizer.app
app.run_server(debug=True)

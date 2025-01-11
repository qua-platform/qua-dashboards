from qua_dashboards.data_visualizer.app import DataVisualizer
import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

data_visualizer = DataVisualizer(update_interval=60000)
app = data_visualizer.app
app.run_server(debug=True)

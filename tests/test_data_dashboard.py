from time import sleep
import xarray as xr
import numpy as np
from src.qua_dashboards.data_dashboard.app import DataDashboardApp
from qua_dashboards.utils.data_utils import send_data_to_dash


def test_002_data_dashboard_basic_types(dash_duo):
    data_dashboard_app = DataDashboardApp()
    dash_duo.start_server(data_dashboard_app.app)

    dash_duo.wait_for_text_to_equal("#title", "Data Dashboard", timeout=4)
    assert dash_duo.get_logs() == [], "browser console should contain no error"

    send_data_to_dash(
        {
            "text": "Hello, world!",
            "number": 123,
            "float": 1.23,
            "boolean": True,
            "list": [1, 2, 3],
            "dict": {"a": 1, "b": 2},
        },
        url="http://localhost:58050",
    )
    sleep(1)
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_003_data_dashboard_data_array(dash_duo):
    data_dashboard_app = DataDashboardApp()
    dash_duo.start_server(data_dashboard_app.app)

    dash_duo.wait_for_text_to_equal("#title", "Data Dashboard", timeout=4)
    assert dash_duo.get_logs() == [], "browser console should contain no error"

    send_data_to_dash(
        {
            "text": "Hello, world!",
            "array": xr.DataArray(np.random.rand(10, 10), name=f"my_arr_1"),
            "array2": xr.DataArray(np.random.rand(100, 100), name=f"my_arr_2"),
        },
        url="http://localhost:58050",
    )
    sleep(1)
    assert dash_duo.get_logs() == [], "browser console should contain no error"

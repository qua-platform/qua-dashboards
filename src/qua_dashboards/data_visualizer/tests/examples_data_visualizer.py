# %% Imports
import requests
import time
import numpy as np
import xarray as xr
from qua_dashboards.utils.data_serialisation import serialise_data


def send_data_to_dash(data):
    serialised_data = serialise_data(data)
    response = requests.post("http://localhost:8050/update-data", json=serialised_data)
    if response.ok:
        print("Data sent successfully")
    else:
        print("Failed to send data")


# %% Send basic data
data = {"current_time": time.time(), "random_value": np.random.rand()}
send_data_to_dash(data)
time.sleep(1)


# %% Send xarray 2D data
data = {
    "array": xr.DataArray(np.random.rand(10, 10), name="my_arr"),
    "array2": xr.DataArray(np.random.rand(100, 100), name=f"my_arr_{i+1}"),
}
send_data_to_dash(data)

# %% Repeatedly send data
for i in range(100):
    data = {
        "array": xr.DataArray(np.random.rand(10, 10), name=f"my_arr_{i}"),
        "array2": xr.DataArray(np.random.rand(100, 100), name=f"my_arr_{i+1}"),
        "current_time": time.time(),
    }
    send_data_to_dash(data)
    time.sleep(0.2)

# %% Repeatedly send simple data
for i in range(100):
    data = {"current_time": time.time(), "random_value": np.random.rand()}
    send_data_to_dash(data)
    time.sleep(0.2)

# %%

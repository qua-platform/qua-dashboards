# %% Imports
import requests
import time
import numpy as np
import xarray as xr
from qua_dashboards.utils.data_utils import serialise_data, send_data_to_dash

# %% Send basic data
data = {"current_time": time.time(), "random_value": np.random.rand()}
send_data_to_dash(data)
time.sleep(1)


# %% Send xarray 2D data
data = {
    "array": xr.DataArray(np.random.rand(10, 10), name="my_arr_1"),
    "array2": xr.DataArray(np.random.rand(100, 100), name=f"my_arr_2"),
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


# %% Send xarray dataset
data = {
    "dataset": xr.Dataset(
        {"array": xr.DataArray(np.random.rand(10, 10), name="my_arr")}
    )
}
send_data_to_dash(data)

# %% Send xarray dataset with multiple data arrays
data = {
    "dataset": xr.Dataset(
        {
            "array": xr.DataArray(
                np.random.rand(10, 10),
                name="my_arr",
                coords={"x": np.arange(10), "y": np.arange(10)},
            ),
            "array2": xr.DataArray(
                np.random.rand(100, 100),
                name="my_arr2",
                coords={"x2": np.arange(100), "y2": np.arange(100)},
            ),
        }
    )
}
send_data_to_dash(data)
# %%

# %%

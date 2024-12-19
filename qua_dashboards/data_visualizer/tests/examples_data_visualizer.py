# %% Imports
import requests
import time
import numpy as np
import xarray as xr
from data_serialisation import serialise_data


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


# %% Send xarray data
data = {"array": xr.DataArray(np.random.rand(10, 10), name="my_arr")}
send_data_to_dash(data)
time.sleep(1)

# %%

# %%

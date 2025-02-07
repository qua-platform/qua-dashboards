# %% Imports
import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from qua_dashboards.data_dashboard import DataDashboardClient


def random_array(*dims):
    return xr.DataArray(
        np.random.rand(*dims),
        name="my_arr",
        coords={label: np.arange(dim) for label, dim in zip("xyzabc", dims)},
    )


def create_random_figure():
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, size=x.shape)
    ax.plot(x, y, label="Random Data")
    ax.set_title("Random Matplotlib Figure")
    ax.legend()
    return fig


client = DataDashboardClient()


# %% Send basic data
data = {
    "current_time": time.time(),
    "random_value": np.random.rand(),
}
client.send_data(data)


# %% Send xarray 2D data
data = {
    "array": random_array(10, 10),
    "array2": random_array(100, 100),
}
client.send_data(data)

# %% Repeatedly send data
for i in range(100):
    data = {
        "array": random_array(10, 10),
        "array2": random_array(100, 100),
        "current_time": time.time(),
    }
    client.send_data(data)
    time.sleep(0.2)

# %% Repeatedly send simple data
for i in range(100):
    data = {"current_time": time.time(), "random_value": np.random.rand()}
    client.send_data(data)
    time.sleep(0.2)


# %% Send xarray dataset
data = {
    "dataset": xr.Dataset(
        {"array": random_array(10, 10)}
    )
}
client.send_data(data)

# %% Send xarray dataset with multiple data arrays
data = {
    "dataset": xr.Dataset(
        {
            "array": random_array(10, 10),
            "array2": random_array(100, 100),
        }
    )
}
client.send_data(data)


# %% Send Matplotlib figure with random data
for k in range(100):
    fig = create_random_figure()
    data = {"random_matplotlib_figure": fig}
    client.send_data(data)
    time.sleep(0.2)

# %% Send a 3D xarray data array
dims = (10, 10, 10)
data = {
    "bye": "hello",
    "data_array": random_array(*dims),
    "data_array_str": xr.DataArray(
        np.random.rand(*dims),
        name="my_arr",
        coords={
            "x": [f"q{i+1}" for i in range(dims[0])],
            "y": np.arange(dims[1]),
            "z": np.arange(dims[2]),
        },
    ),
}
client.send_data(data)

# %% Send a 3D xarray dataset
data = {
    "dataset": xr.Dataset(
        {
            "array": random_array(10, 10, 10),
        }
    )
}
client.send_data(data)


# %%

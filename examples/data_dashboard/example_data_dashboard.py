# %% Imports
import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import logging

from qua_dashboards.data_dashboard import DataDashboardClient
from qua_dashboards.logging_config import logger

logger.setLevel(logging.DEBUG)


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


# %% Basic scalar data example
data = {
    "current_time": time.time(),
    "random_value": np.random.rand(),
    "str_variable": "hello",
}
client.send_data(data)

# %% DataArray examples - single arrays of different dimensions
data = {
    "data_array_1D": random_array(10),
    "data_array_2D": random_array(10, 10),
    "data_array_3D": random_array(10, 10, 10),
    "data_array_4D": random_array(2, 10, 10, 10),
}
client.send_data(data)

# %% DataArray example - with custom string coordinates
data = {
    "data_array_with_str_coords": xr.DataArray(
        np.random.rand(10, 10, 10),
        name="my_arr",
        coords={
            "x": [f"q{i+1}" for i in range(10)],
            "y": [f"q{i+1}" for i in range(10)],
            "z": [f"q{i+1}" for i in range(10)],
        },
    ),
}
client.send_data(data)

# %% Dataset example - simple single array
data = {"simple_dataset": xr.Dataset({"array": random_array(10, 10)})}
client.send_data(data)

# %% Dataset example - multiple arrays of different dimensions
data = {
    "complex_dataset": xr.Dataset(
        {
            "array_1D": random_array(10),
            "array_2D": random_array(10, 10),
            "array_3D": random_array(10, 10, 100),
        }
    )
}
client.send_data(data)

# %% Matplotlib figure example
fig = create_random_figure()
data = {"matplotlib_figure": fig}
client.send_data(data)

# %% Real-time data streaming example - basic scalars
for i in range(10):  # Reduced iterations for example
    data = {
        "current_time": time.time(),
        "random_value": np.random.rand(),
    }
    client.send_data(data)
    time.sleep(0.2)

# %% Real-time data streaming example - arrays
for i in range(10):  # Reduced iterations for example
    data = {
        "evolving_2D_array": random_array(10, 10),
        "current_time": time.time(),
    }
    client.send_data(data)
    time.sleep(0.2)

# %% Real-time data streaming example - Matplotlib
for i in range(10):  # Reduced iterations for example
    fig = create_random_figure()
    data = {"real_time_plot": fig}
    client.send_data(data)
    time.sleep(0.2)

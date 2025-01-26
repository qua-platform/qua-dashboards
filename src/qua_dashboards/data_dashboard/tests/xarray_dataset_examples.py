import xarray as xr
import numpy as np

# Create DataArrays
temperature = xr.DataArray(
    data=np.random.rand(4, 4),
    dims=["lat", "lon"],
    coords={"lat": np.linspace(-90, 90, 4), "lon": np.linspace(-180, 180, 4)},
    attrs={"units": "Celsius", "description": "Temperature data"},
)

precipitation = xr.DataArray(
    data=np.random.rand(4, 4),
    dims=["lat", "lon"],
    coords={"lat": np.linspace(-90, 90, 4), "lon": np.linspace(-180, 180, 4)},
    attrs={"units": "mm", "description": "Precipitation data"},
)

elevation = xr.DataArray(
    data=np.random.rand(3, 3),
    dims=["x", "y"],
    coords={"x": np.arange(3), "y": np.arange(3)},
    attrs={"units": "meters", "description": "Elevation data"},
)

velocity = xr.DataArray(
    data=np.random.rand(20),
    dims=["time"],
    attrs={"units": "m/s", "description": "Velocity data"},
)

acceleration = xr.DataArray(
    data=np.random.rand(20),
    dims=["time"],
    attrs={"units": "m/s^2", "description": "Acceleration data"},
)

random_1d = xr.DataArray(
    data=np.random.rand(10),
    dims=["time"],
    coords={"time": np.arange(10)},
    attrs={"units": "meters", "description": "Random 1D data"},
)

# Create Datasets using subsets of the DataArrays
# 1. Dataset with temperature and precipitation
dataset1 = xr.Dataset({"temperature": temperature, "precipitation": precipitation})

# 2. Dataset with elevation only
dataset2 = xr.Dataset({"elevation": elevation})

# 3. Dataset with velocity and acceleration
dataset3 = xr.Dataset({"velocity": velocity, "acceleration": acceleration})

# 4. Dataset with random 1D data
dataset4 = xr.Dataset({"random_1d": random_1d})

# 5. Dataset with temperature and elevation
dataset5 = xr.Dataset({"temperature": temperature, "elevation": elevation})

# 6. Dataset with all DataArrays
dataset6 = xr.Dataset(
    {
        "temperature": temperature,
        "precipitation": precipitation,
        "elevation": elevation,
        "velocity": velocity,
        "acceleration": acceleration,
        "random_1d": random_1d,
    }
)

# Print the datasets to verify
print(dataset1)
print(dataset2)
print(dataset3)
print(dataset4)
print(dataset5)
print(dataset6)

import xarray as xr
import numpy as np
import random
from .logging_config import logger


def generate_random_xarray():
    num_arrays = random.randint(1, 3)
    logger.info(f"Generating random xarray dataset with {num_arrays} arrays")
    data_vars = {}

    for i in range(num_arrays):
        if random.choice([True, False]):
            logger.debug(f"Generating 1D array_{i+1}")
            times = np.linspace(0, 10, 100)
            values = np.sin(times + random.uniform(0, 2 * np.pi))
            data_vars[f"array_{i+1}"] = xr.DataArray(
                values, coords=[times], dims=["time"]
            )
        else:
            logger.debug(f"Generating 2D array_{i+1}")
            x = np.linspace(0, 10, 100)
            y = np.linspace(0, 10, 100)
            z = np.sin(x[:, None] + y[None, :] + random.uniform(0, 2 * np.pi))
            data_vars[f"array_{i+1}"] = xr.DataArray(z, coords=[x, y], dims=["x", "y"])

    return xr.Dataset(data_vars)

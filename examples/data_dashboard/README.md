# Data Dashboard

The Data Dashboard is a versatile component within `qua-dashboards` designed for **visualizing various types of data sent from a Python client**.

## Functionality

Its primary function is to provide a flexible and dynamic way to display data structures such as:

- Scalars (numbers, strings, booleans)

- Lists and Dictionaries

- `xarray.DataArray` objects (1D, 2D, and ND)

- `xarray.Dataset` objects

- Matplotlib figures

The dashboard intelligently renders these data types into appropriate visual components, like plots, heatmaps, and interactive slicers for multi-dimensional arrays.

## Usage with QUAlibrate

The Data Dashboard serves as a crucial tool for **live plotting and data inspection within the QUAlibrate framework**.
It allows users to send experimental results or intermediate data from their QUAlibrate workflows directly to the dashboard for immediate visualization and analysis.

## Example

For detailed examples of how to send data to the dashboard and see it rendered, please refer to the example script located at [example_data_dashboard.py](example_data_dashboard.py)

This script showcases sending various data types and how they are displayed.

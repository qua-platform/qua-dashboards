# QUA Dashboards - Examples

This document provides an overview of the `qua-dashboards` library and the examples demonstrating its usage.
`qua-dashboards` offers a suite of tools for creating interactive web-based dashboards to control and monitor quantum experiments, particularly those involving Quantum Orchestration Platform (OPX) systems.

## Requirements

- Python 3.9 to 3.12
- QCoDeS (if using QDAC or other QCoDeS-compatible instruments with `VoltageControlComponent`)

## Installation

To install `qua-dashboards`, you can typically use pip:

```bash
pip install qua-dashboards
```

For development or to install from a local source, navigate to the root directory of the `qua-dashboards` repository and run:

```bash
pip install .
```

Or for an editable install:

```bash
pip install -e .
```

## Dashboard Components & Examples

The `examples` folder showcases the core components of `qua-dashboards`.
Each subfolder contains specific examples and a dedicated README with more detailed information.

### Data Dashboard

The Data Dashboard provides a flexible interface to visualize various data types sent from a Python client.
This includes scalars, arrays, `xarray` Datasets and DataArrays, and Matplotlib figures.
It is particularly useful for live plotting and data inspection within the QUAlibrate framework.
For more details, see the [Data Dashboard README](./examples/data_dashboard/README.md).
An illustrative script can be found at [examples/data_dashboard/example_data_dashboard.py](examples/data_dashboard/example_data_dashboard.py).

### Video Mode

Video Mode enables continuous, rapid 2D parameter scans with real-time visualization, which is ideal for characterizing quantum devices like spin qubits.
It supports both simulated data for testing purposes and live data acquisition with an OPX.
Further information is available in the [Video Mode README](./examples/video_mode/README.md).
Example scripts include [examples/video_mode/simulated_examples/example_video_mode_random.py](examples/video_mode/example_video_mode_random.py) for simulated random data and [examples/video_mode/example_video_mode_opx.py](examples/video_mode/example_video_mode_opx.py) for an example OPX integration.

### Voltage Control

The Voltage Control component offers a GUI to monitor and interactively update DC voltage levels from various sources.
This is highly useful for fine-tuning experimental parameters during an experiment.
More details can be found in the [Voltage Control README](./examples/voltage_control/README.md).
Examples are provided for simulated channels ([examples/voltage_control/example_voltage_control.py](examples/voltage_control/example_voltage_control.py)) and for QDevil QDAC integration ([examples/voltage_control/example_voltage_control_qdac.py](examples/voltage_control/example_voltage_control_qdac.py)).

### Virtual Gates

The Virtual Gates component is a GUI to add and edit virtual gating matrix layers, in conjunction with quam_builder's VirtualGateSet. 
This is particularly useful when correcting for cross-capacitance, defining arbitrary axes along the charge stability diagram, or rotating the frame of the stability diagram. 
An example, with an example quam machine is provided in ([examples/virtual_gates/example_virtual_gates.py](examples/virtual_gates/example_virtual_gates.py))

## Combining Components

A powerful feature of `qua-dashboards` is the ability to combine different components into a single, cohesive dashboard.
The script [examples/example_video_mode_full.py](examples/example_video_mode_full.py) demonstrates this by integrating the `VideoModeComponent` and `VoltageControlComponent`.
This allows for use cases such as interactively tuning DC voltage offsets for a 2D scan in Video Mode while observing the results in real-time.

Please explore the individual example folders and their READMEs for comprehensive guides on how to run and customize each dashboard.
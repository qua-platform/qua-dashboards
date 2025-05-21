# Voltage Control Dashboard

The `VoltageControlComponent` within `qua-dashboards` provides a graphical user interface (GUI) to monitor and update the DC voltage levels of various sources in your experimental setup.
This component is essential for experiments where precise and interactive control over DC biases is required, particularly in quantum device characterization.

## Core Functionality

- **View Voltages**: Displays the current voltage values for a list of configured DC parameters (e.g., gate voltages).

- **Update Voltages**: Allows users to input new voltage values directly through the dashboard, which are then applied to the corresponding physical or simulated DC sources.

- **Real-time Feedback**: The displayed values update periodically to reflect the actual state of the voltage sources.

## Integration with Other Dashboards

A key advantage of the `VoltageControlComponent` is its ability to be seamlessly integrated with other `qua-dashboards` components, such as the `VideoModeComponent`.
This allows for powerful experimental workflows where you can:

- **Tune DC voltages interactively**: Adjust gate voltages or other DC biases using the `VoltageControlComponent`.

- **Observe immediate effects**: Simultaneously view how these voltage changes affect your measurements in another dashboard component, like a live 2D scan in Video Mode.
  For instance, you can use the `VoltageControlComponent` to set the center point (offset) of a 2D scan performed by the Video Mode.

This combination enables efficient device tuning and characterization by providing direct control and immediate visual feedback.
An example demonstrating this combined usage can be found in `examples/combined_video_mode_voltage_control.py`.

## Examples

The `examples/voltage_control` folder contains scripts to help you get started:

### 1. Simulated Voltage Control ([example_voltage_control.py]())

- **Purpose**: This example uses simulated voltage parameters (`BasicParameter`).
  It's designed for **testing the functionality and familiarizing yourself with the `VoltageControlComponent` interface** without needing any actual hardware.

- **Use Case**:

  - Understand how to define voltage parameters for the dashboard.

  - Test the UI for viewing and setting voltages.

  - Develop custom dashboard layouts incorporating voltage control in a simulated environment.

- **How it Works**: It defines a list of `BasicParameter` objects, each representing a simulated DC voltage source.
  These are then passed to the `VoltageControlComponent` to create the dashboard.

### 2. QDAC Voltage Control (`example_voltage_control_qdac.py`)

- **Purpose**: This is the more relevant, **real-life use case**, demonstrating how to connect the `VoltageControlComponent` to an actual QDevil QDAC (Digital-to-Analog Converter).

- **Use Case**:

  - Controlling the gate voltages of a spin qubit device connected to a QDAC.

  - Integrating QDAC voltage control into larger experimental setups managed via `qua-dashboards`.

- **How it Works**:

  - It uses QCoDeS to connect to a QDAC instrument.

  - QCoDeS parameters representing the QDAC channels are then passed to the `VoltageControlComponent`.

  - The dashboard allows viewing and setting the DC voltage output of the specified QDAC channels.

By using these examples, you can quickly learn how to set up and utilize the `VoltageControlComponent` for both simulated testing and controlling real experimental hardware.

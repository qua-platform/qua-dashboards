# Video Mode for 2D Scanning

The Video Mode module within `qua-dashboards` is a powerful tool designed for performing continuous, rapid 2D scans and visualizing the results in real-time.
It is particularly well-suited for applications in quantum computing, such as characterizing spin qubit systems by sweeping two parameters (e.g., gate voltages) and measuring a corresponding signal.

The core functionality provides an interactive web frontend that displays a live plot of the 2D scan data, along with controls to adjust sweep parameters dynamically.

## Key Features

- **Live 2D Visualization**: See your scan data update in real-time as it's acquired.

- **Interactive Controls**: Adjust scan parameters like span, points, and acquisition settings directly from the dashboard.

- **Modular Design**: The video mode is built with flexibility in mind, allowing for different data acquisition backends and scan strategies.

  - **Data Acquirers**: Define how data is obtained (e.g., from a simulation or a real OPX).

  - **Scan Modes**: Control the pattern of the 2D sweep (e.g., raster, spiral).

  - **Inner Loop Actions**: Specify the precise QUA operations to perform at each point in the scan.

- **Annotation and Analysis**: Capture static frames from the live view for detailed annotation (points, lines) and basic analysis (e.g., slope calculations).

## Running Video Mode

The `[examples/video_mode](examples/video_mode)` folder provides practical demonstrations of how to use this module.

### Simulating Video Mode ([example_video_mode_random.py](example_video_mode_random.py))

For development, testing, or educational purposes, you can run Video Mode in a simulation.
This does not require any connection to an OPX.

- **Purpose**: This example is designed for **simulating the behavior and testing the functionality** of the Video Mode dashboard.
  It utilizes the `RandomDataAcquirer`, which generates random data points instead of interfacing with actual hardware.
- **Use Case**:
  - Developing and testing custom `VideoModeComponent` configurations without needing an OPX.
  - Familiarizing yourself with the video mode interface and features.
  - Debugging custom scan modes or inner loop actions in a controlled environment.
- **How it Works**: It sets up `SweepAxis` for X and Y dimensions and feeds them to a `RandomDataAcquirer`.
  The `VideoModeComponent` then displays the randomly generated 2D scan.
- **Further Details**: For a detailed guide, see the `example_video_mode_random.py` script.

### Running Video Mode with OPX ([example_video_mode_opx.py](example_video_mode_opx.py))

This is the primary mode for conducting experiments on actual quantum hardware.

- **Purpose**: This example demonstrates how to perform **actual 2D scans on a qubit chip** using a Quantum Orchestration Platform (OPX).
- **Use Case**:
  - Live characterization of quantum devices, such as mapping charge stability diagrams in spin qubit systems by sweeping two gate voltages and measuring current or sensor response.
  - Any experiment requiring a fast 2D parameter sweep with immediate visual feedback.
- **How it Works**:
  - **`OPXDataAcquirer`**: This component is central. It connects to a `QuantumMachinesManager` and executes a QUA program on the OPX.
  - **QUA Program**: The QUA program is dynamically generated based on your settings. It typically involves:
    - Iterating through the X and Y sweep values defined by `SweepAxis` objects, according to the chosen `ScanMode`.
    - At each point in the 2D grid, executing an `InnerLoopAction`. This action contains the QUA code for setting the appropriate DC offsets or playing pulses corresponding to the current X and Y values.
    - Performing a measurement (e.g., demodulation) as defined in the `InnerLoopAction`.
    - Streaming the measurement results (e.g., I and Q values) back to the host computer.
  - **`VideoModeComponent`**: This Dash component receives the streamed data from the `OPXDataAcquirer`, processes it (e.g., calculates magnitude or phase if needed), and updates the live 2D plot on the web interface.
  - **QUAM `Machine` Object**: The `OPXDataAcquirer` often uses a QUAM (QUA Metamodel) `Machine` object to generate the QUA configuration, making it easier to define elements, operations, and pulses.
  - **`BasicInnerLoopAction`**: A common starting point that handles setting DC offsets for two elements and performing a measurement. This can be customized or replaced with a more complex sequence specific to your experiment.

## Basic Usage Workflow (Focus on OPX)

1.  **Define Sweep Axes**: Create `SweepAxis` instances for your X and Y parameters (e.g., gate voltages).
    Specify their `name`, `label`, `units`, `span`, and `points`.
    Optionally, link them to `offset_parameter` for integration with voltage control components.

2.  **Configure QUAM `Machine`**: Ensure your QUAM `Machine` object correctly defines the QUA elements and operations you'll use for sweeping and measurement.

3.  **Define Inner Loop Action**:
    Create an instance of `BasicInnerLoopAction` or implement a custom class inheriting from `InnerLoopAction`.
    This class will contain the QUA code to be executed at each point of the 2D scan (e.g., `set_dc_offset`, `measure`).
    Pass the relevant QUAM elements and pulse definitions to it.

4.  **Choose Scan Mode**: Select a `ScanMode` (e.g., `RasterScan`, `SpiralScan`, `SwitchRasterScan`) to define the path of the 2D sweep.

5.  **Instantiate `OPXDataAcquirer`**:
    Provide the `QuantumMachinesManager` (qmm), your QUAM `Machine` object, the configured `InnerLoopAction`, `ScanMode`, and the `SweepAxis` objects.
    Specify the `result_type` (e.g., "I", "Q", "amplitude", "phase") you want to display.

6.  **Instantiate `VideoModeComponent`**: Pass the configured `OPXDataAcquirer` to the `VideoModeComponent`.
    Adjust `data_polling_interval_s` as needed.

7.  **Build and Run Dashboard**: Use `build_dashboard` to create the Dash application, including the `VideoModeComponent` (and any other components like `VoltageControlComponent`), and run it.

This Video Mode module provides a flexible and interactive way to perform and analyze 2D scans, accelerating the experimental workflow, especially in the context of spin qubit research.

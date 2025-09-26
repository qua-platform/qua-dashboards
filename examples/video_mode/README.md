# Video Mode for 2D Scanning

The Video Mode module within `qua-dashboards` is a powerful tool designed for performing continuous, rapid 1D and 2D scans with virtual gates and visualizing the results in real-time.

It is particularly well-suited for applications in quantum computing, such as characterizing spin qubit systems by sweeping two parameters (e.g., gate voltages) and measuring a corresponding signal.

The core functionality provides an interactive web frontend that displays a live plot of the 1D and 2D scan data, along with controls to adjust sweep parameters dynamically.

## Key Features

- **Live 1D and 2D Visualization**: See your scan data update in real-time as it's acquired.

- **Virtual Gates**: Use `VirtualGateSet` to virtualize your gates live during acquisition. 

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
    - Iterating through the X and Y sweep values derived from the `GateSet` (selected via `x_axis_name`/`y_axis_name`) according to the chosen `ScanMode`.
    - At each point in the 2D grid, executing an `InnerLoopAction`. This action contains the QUA code for setting the appropriate DC offsets or playing pulses corresponding to the current X and Y values.
    - Performing a measurement (e.g., demodulation) as defined in the `InnerLoopAction`.
    - Streaming the measurement results (e.g., I and Q values) back to the host computer.
  - **`VideoModeComponent`**: This Dash component receives the streamed data from the `OPXDataAcquirer`, processes it (e.g., calculates magnitude or phase if needed), and updates the live 2D plot on the web interface.
  - **QUAM `Machine` Object**: The `OPXDataAcquirer` often uses a QUAM (QUA Metamodel) `Machine` object to generate the QUA configuration, making it easier to define elements, operations, and pulses.
  - **`BasicInnerLoopAction`**: A common starting point that handles setting DC offsets for two elements and performing a measurement. This can be customized or replaced with a more complex sequence specific to your experiment.

## Basic Usage Workflow (Focus on OPX)

1. **Configure your QuAM machine**: Build QuAM channels that match your hardware, and ensure that your QuAM `machine` object correctly defines the QUA elements and operations you'll use for sweeping and measurement. 

  ```python
  from quam.core import InOutSingleChannel, BasicQuam
  from quam_builder.architecture.quantum_dots.components import VoltageGate, GateSet, VirtualGateSet

  machine = BasicQuam() # An example QuAM basic machine

  machine.channels["ch1"] = VoltageGate(
      opx_output=("con1", 1),  # OPX controller and port
      sticky=StickyChannelAddon(duration=1_000, digital=False),  # For DC offsets
      operations={"half_max_square": pulses.SquarePulse(amplitude=0.25, length=1000)}, #Ensure operation "half_max_square" exists in the channel object
  )
  # Define the second DC voltage output channel (e.g., for Y-axis sweep)
  machine.channels["ch2"] = VoltageGate(
      opx_output=("con1", 2),  # OPX controller and port
      sticky=StickyChannelAddon(duration=1_000, digital=False),  # For DC offsets
      operations={"half_max_square": pulses.SquarePulse(amplitude=0.25, length=1000)},
  )

  # Define the readout pulse and the channel used for measurement
  readout_pulse = pulses.SquareReadoutPulse(id="readout", length=1500, amplitude=0.1)
  machine.channels["ch_readout"] = InOutSingleChannel(
      opx_output=("con1", 3),  # Output for the readout pulse
      opx_input=("con1", 1),  # Input for acquiring the measurement signal
      intermediate_frequency=20000000,  # Set IF for the readout channel
      operations={"readout": readout_pulse},  # Assign the readout pulse to this channel
      sticky=StickyChannelAddon(duration=1_000, digital=False),  # For DC offsets
  )
  ```

2.  **Define Gate Set**: Build a `GateSet` that models your voltage gates (physical and virtual) and their relationships. Ensure it exposes the channels you intend to sweep.

  ```python
  ### Example implementation of GateSet
  from quam_builder.architecture.quantum_dots.components import GateSet  # Requires quam-builder
  channels = {
      "ch1": machine.channels["ch1"].get_reference(), # .get_reference() necessary if the channel is already parented by a QuAM machine
      "ch2": machine.channels["ch2"].get_reference(),
      "ch_readout": machine.channels["ch_readout"].get_reference()
  }
  gate_set = GateSet(id = "Plungers", channels = channels)
  machine.gate_set = gate_set


  ### Example implementation of VirtualGateSet
  from quam_builder.architecture.quantum_dots.components import VirtualGateSet  # Requires quam-builder
  channels = {
      "ch1": machine.channels["ch1"].get_reference(), # .get_reference() necessary if the channel is already parented by a QuAM machine
      "ch2": machine.channels["ch2"].get_reference(),
      "ch_readout": machine.channels["ch_readout"].get_reference()
  }
  gate_set = VirtualGateSet(id = "Plungers", channels = channels)
  gate_set.add_layer(
      source_gates = ["V1", "V2"], # Pick the virtual gate names here 
      target_gates = ["ch1", "ch2"], # Must be a subset of gates in the gate_set
      matrix = [[1, 0.2], [0.2, 1]] # Any example matrix
  )
  machine.gate_set = gate_set
  ```

3.  **Define Inner Loop Action**:
    Use `BasicInnerLoopAction` (or a custom implementation) to set the sequence and measurement. When using `OPXDataAcquirer`, this action can be created automatically from the provided `GateSet`, `readout_pulse`, and the selected `x_axis_name`/`y_axis_name`.

4.  **Choose Scan Mode**: Select a `ScanMode` (e.g., `RasterScan`, `SpiralScan`, `SwitchRasterScan`) to define the path of the 2D sweep.
  ```python
  from qua_dashboards.video_mode import (
      OPXDataAcquirer,
      scan_modes,
      VideoModeComponent,
  )
  scan_mode = scan_modes.SwitchRasterScan()
  ```


5.  **Instantiate `OPXDataAcquirer`**:
    Provide the `QuantumMachinesManager` (qmm), your QUAM `Machine` object, the `GateSet`, `x_axis_name`/`y_axis_name`, `available_readout_pulses`, and the `ScanMode`.
    Optionally set `result_type` (e.g., "I", "Q", "amplitude", "phase").

  ```python
  qmm = QuantumMachinesManager(host=host_ip, cluster_name=cluster_name)
  data_acquirer = OPXDataAcquirer(
      qmm=qmm,
      machine=machine,
      gate_set=gate_set,  # Replace with your GateSet instance
      x_axis_name="ch1",  # Must appear in gate_set.valid_channel_names; Virtual gate names also valid
      y_axis_name="ch2",  # Must appear in gate_set.valid_channel_names; Virtual gate names also valid
      scan_mode=scan_mode,
      result_type="I",  # "I", "Q", "amplitude", or "phase"
      available_readout_pulses=[readout_pulse] # Input a list of pulses. The default only reads out from the first pulse, unless the second one is chosen in the UI. 
  )
  ```

6.  **Instantiate `VideoModeComponent`**: Pass the configured `OPXDataAcquirer` to the `VideoModeComponent`.
    Adjust `data_polling_interval_s` as needed.

  ```python
  video_mode_component = VideoModeComponent(
    data_acquirer=data_acquirer,
    data_polling_interval_s=0.5,  # How often the dashboard polls for new data
    save_path = save_path
  )
  ```

  - If virtual gates are needed, also instantiate a the Virtual Gates component: 
  ```python
  from qua_dashboards.virtual_gates import VirtualLayerEditor, ui_update
  virtual_gates_component = VirtualLayerEditor(gateset = gate_set, component_id = 'Virtual Gates UI')
  ```


7.  **Build and Run Dashboard**: Use `build_dashboard` to create the Dash application, including the `VideoModeComponent` (and any other components like `VoltageControlComponent`), and run it.

```python
  app = build_dashboard(
      components=[video_mode_component, virtual_gates_component],
      title="OPX Video Mode Dashboard",  # Title for the web page
  )
  ui_update(app, video_mode_component) # Necessary for VirtualGates component. 
  logger.info("Dashboard built. Starting Dash server on http://localhost:8050")
  # Run the Dash server.
  # `host="0.0.0.0"` makes it accessible on your network.
  # `use_reloader=False` is often recommended for stability with background threads.
  app.run(debug=True, host="0.0.0.0", port=8050, use_reloader=False)

```

This Video Mode module provides a flexible and interactive way to perform and analyze 2D scans, accelerating the experimental workflow, especially in the context of spin qubit research.

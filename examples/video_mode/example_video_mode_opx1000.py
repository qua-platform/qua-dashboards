"""
Example Script: Video Mode with OPX1000

This script demonstrates how to use the VideoModeComponent with an OPXDataAcquirer
to perform live 2D scans on a quantum device. It sets up a QUA program to sweep
two DC voltage channels and measure a readout signal, displaying the results in a
real-time dashboard.

Quick How-to-Use:
1.  **Configure Hardware**:
    * Update `qmm = QuantumMachinesManager(...)` with your OPX host and cluster_name.
    * Modify the `machine = BasicQuam()` section to define the QUA elements
        (channels, pulses) that match your experimental setup.
        Ensure `ch1`, `ch2` (for sweeping) and `ch1_readout` (or your measurement
        channel) are correctly defined.
2.  **Adjust Scan Parameters**:
    * Modify `x_axis` and `y_axis` (`SweepAxis` objects) with your desired span,
        points, and any offset parameters.
    * Review `inner_loop_action` (`BasicInnerLoopAction`) and ensure the
        `x_element`, `y_element`, and `readout_pulse` correspond to your QUAM machine
        definitions. Adjust `use_dBm` or other parameters as needed.
    * Select a `scan_mode` (e.g., `SwitchRasterScan`, `RasterScan`).
    * Set `result_type` in `OPXDataAcquirer` (e.g., "I", "Q", "amplitude", "phase").
3.  **Run the Script**: Execute this Python file.
4.  **Open Dashboard**: Navigate to `http://localhost:8050` (or the address shown
    in your terminal) in a web browser to view the live video mode dashboard.

Note: The sections for "(Optional) Run program and acquire data" and "DEBUG: Generate QUA script"
and "Test simulation" are for direct execution/debugging and can be commented out
if you only intend to run the live dashboard.
"""

# %% Imports
import numpy as np
from matplotlib import pyplot as plt
from qm import QuantumMachinesManager, SimulationConfig, generate_qua_script
from quam.components import (
    BasicQuam,
    SingleChannel,
    InOutSingleChannel,
    pulses,
    StickyChannelAddon,
    InOutMWChannel, 
)

from quam_builder.architecture.quantum_dots.voltage_sequence.gate_set import GateSet
from quam_builder.architecture.quantum_dots.virtual_gates.virtual_gate_set import VirtualGateSet, VirtualQdacGateSet
from quam_builder.hardware.quam_channel import QdacOpxChannel, QdacOpxReadout, CoupledSingleChannel, CoupledInOutSingleChannel

from qua_dashboards.core import build_dashboard
from qua_dashboards.utils import setup_logging, BasicParameter
from qua_dashboards.video_mode import (
    SweepAxis,
    OPXDataAcquirer,
    scan_modes,
    BasicInnerLoopAction,
    VideoModeComponent,
)
from qua_dashboards.video_mode.video_mode_component import VideoModeComponent_with_GateSet

logger = setup_logging(__name__)

lffem1 = 3
lffem2 = 5
mwfem = 1

path = '/Users/kalidu_laptop/QUA'

# %% Create QUAM Machine Configuration and Connect to Quantum Machines Manager (QMM)

# Initialize a basic QUAM machine object.
# This object will be used to define the quantum hardware configuration (channels, pulses, etc.)
# and generate the QUA configuration for the OPX.
machine = BasicQuam()


# Coupled Single Channel simply a single channel with a 'couplings' input; the couplings input is not saved in the QUAM, instead is composed into a single big matrix which is saved.
# The matrix is only built for the VirtualGateSet. In this case, the couplings field is doing nothing, since this example uses GateSet
machine.channels['ch1'] = CoupledSingleChannel(
    id = 'Plunger1',
    opx_output=("con1", lffem2, 6),  # OPX controller and port
    sticky=StickyChannelAddon(duration=1_000, digital=False),  # For DC offsets
    operations={"step": pulses.SquarePulse(amplitude=0.1, length=1000)},
    couplings = {'Plunger2': 0.2, 'Plunger3': 0.15, 'Sensor1': 0.1}
)

machine.channels['ch2'] = CoupledSingleChannel(
    id = 'Plunger2', 
    opx_output=("con1", lffem1, 8),  # OPX controller and port
    sticky=StickyChannelAddon(duration=1_000, digital=False),  # For DC offsets
    operations={"step": pulses.SquarePulse(amplitude=0.1, length=1000)},
    couplings = {'Plunger1': 0.2, 'Plunger3': 0.25, 'Sensor1': 0.15}
)

machine.channels['ch3'] = CoupledSingleChannel(
    id = 'Plunger3', 
    opx_output=("con1", lffem1, 7),  # OPX controller and port
    sticky=StickyChannelAddon(duration=1_000, digital=False),  # For DC offsets
    operations={"step": pulses.SquarePulse(amplitude=0.1, length=1000)},
    couplings = {'Plunger1': 0.15, 'Plunger2': 0.25, 'Sensor1': 0.2}
)

# Define the readout pulse and the channel used for measurement

readout_pulse = pulses.SquareReadoutPulse(id="readout", length=1500, amplitude=0.1)
machine.channels["ch1_readout"] = CoupledInOutSingleChannel(
    id = "Sensor1",
    opx_output=("con1", lffem1, 1),  # Output for the readout pulse
    opx_input=("con1", lffem1, 1),  # Input for acquiring the measurement signal
    intermediate_frequency=0,  # Set IF for the readout channel
    operations={"readout": readout_pulse, "step": pulses.SquarePulse(amplitude=0.1,length=1500)},
    time_of_flight=32,  # Assign the readout pulse to this channel
    couplings = {'Plunger1': 0.1, 'Plunger2': 0.15, 'Plunger3': 0.2}
)


channels = {machine.channels['ch1'].name: machine.channels['ch1'].get_reference(),
            machine.channels['ch2'].name: machine.channels['ch2'].get_reference(),
            machine.channels['ch3'].name: machine.channels['ch3'].get_reference(),
            machine.channels['ch1_readout'].name: machine.channels['ch1_readout'].get_reference()}

readout = {machine.channels['ch1_readout'].name : machine.channels['ch1_readout'].get_reference()}
machine.gate_set = GateSet(id = 'Plungers', channels=channels)



# --- QMM Connection ---
# Replace with your actual OPX host and cluster name
# Example: qmm = QuantumMachinesManager(host="your_opx_ip", cluster_name="your_cluster")
qmm = QuantumMachinesManager(host="172.16.33.115", cluster_name="CS_3")

# Generate the QUA configuration from the QUAM machine object
config = machine.generate_config()

# Open a connection to the Quantum Machine (QM)
# This prepares the OPX with the generated configuration.
qm = qmm.open_qm(config, close_other_machines=True)


# %% Configure Video Mode Components

# Define BasicParameters for X and Y voltage offsets.
# These can be replaced with QDAC channels.
x_offset = BasicParameter(name="X Voltage Offset", initial_value=0)
y_offset = BasicParameter(name="Y Voltage Offset", initial_value=0)

x_span = BasicParameter(name="X span", initial_value=0.05)
y_span = BasicParameter(name="Y span", initial_value=0.05)

x_resolution = BasicParameter(name="X points", initial_value=101)
y_resolution = BasicParameter(name="Y points", initial_value=101)

x_span = 0.05
y_span = 0.05

x_resolution = 101
y_resolution = 101


# Define the action to be performed at each point in the QUA scan (inner loop).
# BasicInnerLoopAction sets DC offsets on two elements and performs a measurement.
# inner_loop_action = VirtualGateInnerLoopAction(
#     x_element=machine.channels["ch1"],  # QUAM element for X-axis voltage
#     y_element=machine.channels["ch2"],  # QUAM element for Y-axis voltage
#     readout_pulse=readout_pulse,  # QUAM readout pulse to use for measurement
#     # ramp_rate=1_000,                  # Optional: Voltage ramp rate (V/s)
#     use_dBm=True,  # If true, readout amplitude is in dBm
# )

inner_loop_action = BasicInnerLoopAction(
    x_element = machine.channels['ch1'], 
    y_element = machine.channels['ch2'],
    ramp_rate = 0, 
    readout_pulse=readout_pulse, 
    use_dBm = False
)

# Select the scan mode (how the 2D grid is traversed in QUA)
# Options include: RasterScan, SpiralScan, SwitchRasterScan
scan_mode = scan_modes.SwitchRasterScan()

# Instantiate the OPXDataAcquirer.
# This component handles the QUA program generation, execution, and data fetching.
data_acquirer = OPXDataAcquirer(
    qmm=qmm,
    machine=machine,
    qua_inner_loop_action=inner_loop_action,
    scan_mode=scan_mode,
    x_axis=SweepAxis(machine.channels['ch1'].name, span=x_span, points=x_resolution, offset_parameter=x_offset),
    y_axis=SweepAxis(machine.channels['ch2'].name, span=y_span, points=y_resolution, offset_parameter=y_offset),
    result_type="I",  # Specify the type of result to  display (e.g., "I", "Q", "amplitude", "phase")
)

# # %% (Optional) Test: Run QUA program once and acquire data directly
# # This section can be used for a single acquisition test before launching the dashboard.
# # Comment out if you only want to run the live dashboard.
# # results = data_acquirer.perform_actual_acquisition()
# # print(f"Mean of results: {np.mean(np.abs(results))}")

# # plt.figure()
# # plt.pcolormesh(results)
# # plt.colorbar()
# # plt.title("Single Acquisition Test")
# # plt.show()

# # %% Run Video Mode Dashboard

# # Instantiate the main VideoModeComponent, providing the configured data_acquirer.

video_mode_component = VideoModeComponent_with_GateSet(
    data_acquirer=data_acquirer,
    data_polling_interval_s=0.5, 
    gateset=machine.gate_set,
    # inner_loop_action=inner_loop_action, 
)

# virtual_layer_ui = VirtualLayerAdder(machine.gate_set, component_id = 'Virtual Gate Adder')
# virtual_layer_editor = VirtualLayerManager(machine.gate_set, component_id = 'Existing Virtual Gate Editor')

app = build_dashboard(
    components=[video_mode_component],
    title="Combined Dashboard",
)

#Live updating code for the Virtual Gating UI
# ui_update(app, machine.gate_set, virtual_layer_ui, virtual_layer_editor)


logger.info("Dashboard built. Starting Dash server on http://localhost:8050")
# Run the Dash server.
# `host="0.0.0.0"` makes it accessible on your network.
# `use_reloader=False` is often recommended for stability with background threads.
app.run(debug=True, host="127.0.0.1", port=8050, use_reloader=False)

    

# %% --- Debugging Sections (Optional) ---

# # DEBUG: Generate and print the QUA script
# # This can be useful to inspect the QUA code being sent to the OPX.
# qua_script = generate_qua_script(data_acquirer.generate_qua_program(), config)
# print("--- Generated QUA Script ---")
# print(qua_script)
# print("--------------------------")

# # DEBUG: Test QUA program simulation
# # This simulates the QUA program execution without running on the actual OPX.
# try:
#     prog = data_acquirer.generate_qua_program()
#     simulation_config = SimulationConfig(duration=10000)  # Duration in clock cycles (4ns)
#     job = qmm.simulate(config, prog, simulation_config)
#     simulated_samples = job.get_simulated_samples()
#     con1 = simulated_samples.con1

#     plt.figure(figsize=(10, 5))
#     con1.plot(analog_ports=["1", "2"], digital_ports=[]) # Specify ports to plot
#     plt.title("Simulated Analog Output (Ports 1 & 2)")
#     plt.show()

#     # Plot X vs Y voltage trajectory from simulation
#     plt.figure()
#     plt.plot(con1.analog["1"], con1.analog["2"])
#     plt.title("Simulated X-Y Voltage Trajectory")
#     plt.xlabel("Voltage X (simulated)")
#     plt.ylabel("Voltage Y (simulated)")
#     plt.grid(True)
#     plt.show()

#     # Plot the scan pattern used by the scan_mode
#     plt.figure()
#     data_acquirer.scan_mode.plot_scan(
#         data_acquirer.x_axis.points, data_acquirer.y_axis.points
#     )
#     plt.title("Scan Mode Pattern")
#     plt.show()

# except Exception as e:
#     logger.error(f"Error during QUA simulation: {e}")

"""
Example Script: Video Mode with OPX

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
)

from qua_dashboards.core import build_dashboard
from qua_dashboards.utils import setup_logging
from qua_dashboards.video_mode import (
    OPXDataAcquirer,
    scan_modes,
    VideoModeComponent,
)


logger = setup_logging(__name__)

# %% Create QUAM Machine Configuration and Connect to Quantum Machines Manager (QMM)

# Initialize a basic QUAM machine object.
# This object will be used to define the quantum hardware configuration (channels, pulses, etc.)
# and generate the QUA configuration for the OPX.
machine = BasicQuam()

# Define the first DC voltage output channel (e.g., for X-axis sweep)
machine.channels["ch1"] = SingleChannel(
    opx_output=("con1", 1),  # OPX controller and port
    sticky=StickyChannelAddon(duration=1_000, digital=False),  # For DC offsets
    operations={"step": pulses.SquarePulse(amplitude=0.25, length=1000)},
)
# Define the second DC voltage output channel (e.g., for Y-axis sweep)
machine.channels["ch2"] = SingleChannel(
    opx_output=("con1", 2),  # OPX controller and port
    sticky=StickyChannelAddon(duration=1_000, digital=False),  # For DC offsets
    operations={"step": pulses.SquarePulse(amplitude=0.25, length=1000)},
)

# Define the readout pulse and the channel used for measurement
readout_pulse = pulses.SquareReadoutPulse(id="readout", length=1500, amplitude=0.1)
machine.channels["ch1_readout"] = InOutSingleChannel(
    opx_output=("con1", 3),  # Output for the readout pulse
    opx_input=("con1", 1),  # Input for acquiring the measurement signal
    intermediate_frequency=0,  # Set IF for the readout channel
    operations={"readout": readout_pulse},  # Assign the readout pulse to this channel
)

# --- QMM Connection ---
# Replace with your actual OPX host and cluster name
# Example: qmm = QuantumMachinesManager(host="your_opx_ip", cluster_name="your_cluster")
qmm = QuantumMachinesManager(host="`127.0.0.1", cluster_name="CS_1")

# Generate the QUA configuration from the QUAM machine object
config = machine.generate_config()

# Open a connection to the Quantum Machine (QM)
# This prepares the OPX with the generated configuration.
qm = qmm.open_qm(config, close_other_machines=True)


# %% Configure Video Mode Components

# Configure a GateSet that defines the sweepable voltage gates.
# NOTE: Replace the placeholder with your actual GateSet construction.
# The GateSet must include channels matching the names used below (e.g., "ch1" and "ch2").
from quam_builder.architecture.quantum_dots import GateSet  # Requires quam-builder

# TODO: Build a GateSet aligned with your machine configuration
# gate_set = GateSet(...)
gate_set = None  # Placeholder. Replace with a real GateSet instance.

# Select the scan mode (how the 2D grid is traversed in QUA)
# Options include: RasterScan, SpiralScan, SwitchRasterScan
scan_mode = scan_modes.SwitchRasterScan()

# Instantiate the OPXDataAcquirer.
# This component handles the QUA program generation, execution, and data fetching.
data_acquirer = OPXDataAcquirer(
    qmm=qmm,
    machine=machine,
    gate_set=gate_set,  # Replace with your GateSet instance
    x_axis_name="ch1",  # Must appear in gate_set.valid_channel_names
    y_axis_name="ch2",  # Must appear in gate_set.valid_channel_names
    scan_mode=scan_mode,
    readout_pulse=readout_pulse,
    result_type="I",  # "I", "Q", "amplitude", or "phase"
)

# %% (Optional) Test: Run QUA program once and acquire data directly
# This section can be used for a single acquisition test before launching the dashboard.
# Comment out if you only want to run the live dashboard.
# results = data_acquirer.perform_actual_acquisition()
# print(f"Mean of results: {np.mean(np.abs(results))}")

# plt.figure()
# plt.pcolormesh(results)
# plt.colorbar()
# plt.title("Single Acquisition Test")
# plt.show()

# %% Run Video Mode Dashboard

# Instantiate the main VideoModeComponent, providing the configured data_acquirer.
video_mode_component = VideoModeComponent(
    data_acquirer=data_acquirer,
    data_polling_interval_s=0.5,  # How often the dashboard polls for new data
)

# Build the Dash application layout using the VideoModeComponent.
app = build_dashboard(
    components=[video_mode_component],
    title="OPX Video Mode Dashboard",  # Title for the web page
)

logger.info("Dashboard built. Starting Dash server on http://localhost:8050")
# Run the Dash server.
# `host="0.0.0.0"` makes it accessible on your network.
# `use_reloader=False` is often recommended for stability with background threads.
app.run(debug=True, host="0.0.0.0", port=8050, use_reloader=False)


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

"""
Example Script: Video Mode with OPX with DC Offsets provided by the QDAC

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
from dash import html, Output, Input, dcc
from qua_dashboards.video_mode.inner_loop_actions.virtual_gating_inner_loop_action import VirtualGateInnerLoopAction
from qua_dashboards.voltage_control.GateSet_Voltage_Control import GateSetControl
from quam_builder.architecture.quantum_dots.voltage_sequence.gate_set import GateSetQuam
from quam_builder.architecture.quantum_dots.virtual_gates.virtual_gate_set import VirtualGateSet, VirtualQdacGateSet
from quam_builder.hardware.quam_channel import QdacOpxChannel, QdacOpxReadout

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

from qua_dashboards.video_mode.data_acquirers.opx_data_acquirer import OPXQDACDataAcquirer
from qua_dashboards.voltage_control.virtual_layer_UI import VirtualLayerEditor, VirtualLayerManager
logger = setup_logging(__name__)

lffem1 = 3
lffem2 = 5
mwfem = 1

from qcodes_contrib_drivers.drivers.QDevil import QDAC2

try: 
    qdac.close()
except: 
    pass
qdac_addr = "172.16.33.101"
qdac = QDAC2.QDac2(
    "QDAC", visalib="@py", address=f"TCPIP::{qdac_addr}::5025::SOCKET"
)


# %% Create QUAM Machine Configuration and Connect to Quantum Machines Manager (QMM)

# Initialize a basic QUAM machine object.
# This object will be used to define the quantum hardware configuration (channels, pulses, etc.)
# and generate the QUA configuration for the OPX.
machine = GateSetQuam()


machine.channels['ch1'] = QdacOpxChannel(
    id = 'Plunger1',
    qdac = qdac, 
    qdac_channel = 1, 
    qdac_unit = 'V',
    opx_output=("con1", lffem2, 6),  # OPX controller and port
    sticky=StickyChannelAddon(duration=1_000, digital=False),  # For DC offsets
    operations={"step": pulses.SquarePulse(amplitude=0.1, length=1000)},
    couplings = {'Plunger2': 0.2, 'Plunger3': 0.15, 'Sensor1': 0.1}
)

machine.channels['ch2'] = QdacOpxChannel(
    id = 'Plunger2', 
    qdac = qdac, 
    qdac_channel = 2, 
    qdac_unit = 'V',
    opx_output=("con1", lffem1, 8),  # OPX controller and port
    sticky=StickyChannelAddon(duration=1_000, digital=False),  # For DC offsets
    operations={"step": pulses.SquarePulse(amplitude=0.1, length=1000)},
    couplings = {'Plunger1': 0.2, 'Plunger3': 0.25, 'Sensor1': 0.15}
)

machine.channels['ch3'] = QdacOpxChannel(
    id = 'Plunger3', 
    qdac = qdac, 
    qdac_channel = 3, 
    qdac_unit = 'V',
    opx_output=("con1", lffem1, 7),  # OPX controller and port
    sticky=StickyChannelAddon(duration=1_000, digital=False),  # For DC offsets
    operations={"step": pulses.SquarePulse(amplitude=0.1, length=1000)},
    couplings = {'Plunger1': 0.15, 'Plunger2': 0.25, 'Sensor1': 0.2}
)

readout_pulse = pulses.SquareReadoutPulse(id="readout", length=1000, amplitude=0.1)
# Define the readout pulse and the channel used for measurement
machine.channels["ch1_readout"] = QdacOpxReadout(
    id = 'Sensor1', 
    qdac = qdac, 
    qdac_channel = 4, 
    qdac_unit = 'V',
    opx_output=("con1", lffem1, 1),  # Output for the readout pulse
    opx_input=("con1", lffem1, 1),  # Input for acquiring the measurement signal
    intermediate_frequency=200e6,  # Set IF for the readout channel
    operations={"readout": readout_pulse, 
                "step": pulses.SquarePulse(amplitude=0.1, length=1000)},
    time_of_flight=32,  
    couplings = {'Plunger1': 0.1, 'Plunger2': 0.15, 'Plunger3': 0.2}
)

### Right now, the .get_reference() is necessary to map the channels, but should be improved eventually 
channels = {machine.channels['ch1'].name: machine.channels['ch1'].get_reference(),
            machine.channels['ch2'].name: machine.channels['ch2'].get_reference(),
            machine.channels['ch3'].name: machine.channels['ch3'].get_reference(),
            machine.channels['ch1_readout'].name: machine.channels['ch1_readout'].get_reference()}
readout = {'Resonator': machine.channels['ch1_readout'].get_reference()}

machine.gate_set = VirtualQdacGateSet(id = 'Plungers', channels=channels, readout=readout)
machine.gate_set.add_layer(
    source_gates = ['vPlunger1', 'vPlunger2', 'vPlunger3', 'vSensor1'], 
    target_gates = ['Plunger1', 'Plunger2', 'Plunger3', 'Sensor1'], 
    matrix = machine.gate_set.get_cross_capacitive_matrix()
)
machine.gate_set.add_layer(
    source_gates = ['det_Plunger1','det_Plunger2','det_Plunger3','det_Sensor1'],
    target_gates = ['vPlunger1','vPlunger2','vPlunger3','vSensor1'],
    matrix      = [[1,0.2,0,0], [0.2,1,0,0], [0,0,1,0], [0,0,0,1]]
)

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
x_offset = BasicParameter(name="X Voltage Offset", initial_value=0.0)
y_offset = BasicParameter(name="Y Voltage Offset", initial_value=0.0)

x_span = BasicParameter(name="X span", initial_value=0.06)
y_span = BasicParameter(name="Y span", initial_value=0.03)

x_resolution = BasicParameter(name="X points", initial_value=101)
y_resolution = BasicParameter(name="Y points", initial_value=101)

# Define the action to be performed at each point in the QUA scan (inner loop).
inner_loop_action = VirtualGateInnerLoopAction(
    x_element = machine.channels['ch1'], 
    y_element = machine.channels['ch2'],
    gateset=machine.gate_set,
    ramp_rate = 0,
    readout_pulse=readout_pulse
)

# Select the scan mode (how the 2D grid is traversed in QUA)
# Options include: RasterScan, SpiralScan, SwitchRasterScan
scan_mode = scan_modes.SwitchRasterScan()

# Instantiate the OPXQDACDataAcquirer.
# This component handles the QUA program generation, execution, and data fetching.
data_acquirer = OPXQDACDataAcquirer(
    gateset = machine.gate_set,
    qmm=qmm,
    machine=machine,
    qua_inner_loop_action=inner_loop_action,
    scan_mode=scan_mode,
    x_axis=SweepAxis(name = 'X', label = machine.channels['ch1'].name, span=x_span, points=x_resolution, offset_parameter=x_offset),
    y_axis=SweepAxis(name = 'Y', label = machine.channels['ch2'].name, span=y_span, points=y_resolution, offset_parameter=y_offset),
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

video_mode_component = VideoModeComponent_with_GateSet(
    data_acquirer=data_acquirer,
    data_polling_interval_s=0.5, 
    gateset=machine.gate_set,
    inner_loop_action=inner_loop_action,
    machine = machine
)

virtual_layer_ui = VirtualLayerEditor(machine.gate_set, component_id = 'VG_editor')
virtual_layer_editor = VirtualLayerManager(machine.gate_set, component_id = 'Existing Virtual Gate Editor')

gateset_control = GateSetControl(gateset=machine.gate_set)
app = build_dashboard(
    components=[video_mode_component, gateset_control],
    title="Combined Dashboard",
)

#Live updating code for the Virtual Gating UI
app.layout.children.append(dcc.Store(id="vg-layer-refresh-trigger", data=0))
app.layout.children.append(
    html.Div(id="VG_MANAGER_CONTAINER", children=virtual_layer_editor.get_layout()))
app.layout.children.append(
    html.Div(id="VG_EDITOR_CONTAINER", children=virtual_layer_ui.get_layout()))
@app.callback(
    Output("VG_EDITOR_CONTAINER", "children"),
    Input("vg-layer-refresh-trigger", "data"))
def refresh_editor_layout(_):
    return virtual_layer_ui.get_layout()
@app.callback(
    Output("VG_MANAGER_CONTAINER", "children"),
    Input("vg-layer-refresh-trigger", "data"),)
def refresh_manager_layout(_):
    return virtual_layer_editor.get_layout()
virtual_layer_ui.register_callbacks(app)
virtual_layer_editor.register_callbacks(app)


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

#     ### Fill in the string with the right port. f{FEM}-f{PORT} for the OPX1k

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

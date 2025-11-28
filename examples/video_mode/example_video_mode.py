"""
Example Script: Video Mode with OPX with Virtual Gating

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
    ports
)

from qua_dashboards.core import build_dashboard
from qua_dashboards.utils import setup_logging
from qua_dashboards.video_mode import (
    OPXDataAcquirer,
    scan_modes,
    VideoModeComponent,
)
from qua_dashboards.video_mode.tab_controllers import VoltageControlTabController
from qua_dashboards.voltage_control import VoltageControlComponent

from quam_builder.architecture.quantum_dots.qpu import BaseQuamQD
from quam_builder.architecture.quantum_dots.components import VoltageGate, VirtualGateSet, GateSet, ReadoutResonatorSingle


logger = setup_logging(__name__)

# %% Create QUAM Machine Configuration and Connect to Quantum Machines Manager (QMM)

# Initialize a basic QUAM machine object.
# This object will be used to define the quantum hardware configuration (channels, pulses, etc.)
# and generate the QUA configuration for the OPX.

lf_fem = 5
machine = BaseQuamQD()

p1 = VoltageGate(
    id = "plunger_1",
    opx_output=("con1", lf_fem, 1),  # OPX controller and port
    sticky=StickyChannelAddon(duration=1_000, digital=False),  # For DC offsets
)
p2 = VoltageGate(
    id = "plunger_2",
    opx_output=("con1", lf_fem, 2),  # OPX controller and port
    sticky=StickyChannelAddon(duration=1_000, digital=False),  # For DC offsets
)
s1 = VoltageGate(
    id = "sensor_1",
    opx_output=("con1", lf_fem, 8),  # OPX controller and port
    sticky=StickyChannelAddon(duration=1_000, digital=False),  # For DC offsets
)
s2 = VoltageGate(
    id = "sensor_2",
    opx_output=("con1", lf_fem, 7),  # OPX controller and port
    sticky=StickyChannelAddon(duration=1_000, digital=False),  # For DC offsets
)

readout_pulse = pulses.SquareReadoutPulse(length = 200, id = "readout", amplitude = 0.01)
resonator = ReadoutResonatorSingle(
    id = "readout_resonator", 
    intermediate_frequency=120e6,
    operations = {"readout": readout_pulse}, 
    opx_output = ports.LFFEMAnalogOutputPort("con1", lf_fem, port_id = 1), 
    opx_input = ports.LFFEMAnalogInputPort("con1", lf_fem, port_id = 2),
    sticky = StickyChannelAddon(duration = 16, digital = False), 
)

readout_pulse2 = pulses.SquareReadoutPulse(length = 200, id = "readout", amplitude = 0.01)
resonator2 = ReadoutResonatorSingle(
    id = "readout_resonator2", 
    intermediate_frequency=150e6,
    operations = {"readout": readout_pulse2}, 
    opx_output = ports.LFFEMAnalogOutputPort("con1", lf_fem, port_id = 1), 
    opx_input = ports.LFFEMAnalogInputPort("con1", lf_fem, port_id = 2),
    sticky = StickyChannelAddon(duration = 16, digital = False), 
)

# %%

#####################################
###### Create Virtual Gate Set ######
#####################################

# Create virtual gate set out of all the relevant HW channels.
# This function adds HW channels to machine.physical_channels, so no need to independently map
virtual_gate_set_name = "main_qpu"

machine.create_virtual_gate_set(
    virtual_channel_mapping = {
        "virtual_dot_1": p1,
        "virtual_dot_2": p2,
        "virtual_sensor_1": s1,
        "virtual_sensor_2": s2,
    },
    gate_set_id = virtual_gate_set_name
)

# %%
#########################################################
###### Register Quantum Dots, Sensors and Barriers ######
#########################################################

# Shortcut function to register QuantumDots, SensorDots, BarrierGates
machine.register_channel_elements(
    plunger_channels = [p1, p2], 
    barrier_channels = [],
    sensor_channels_resonators = [(s1, resonator), (s2, resonator2)], 
)

# %%
##################################################################
###### Connect the physical channels to the external source ######
##################################################################

qdac_connect = True
voltage_control_tab = None
if qdac_connect: 
    qdac_ip = "172.16.33.101"
    name = "QDAC"
    from qcodes import Instrument
    from qcodes_contrib_drivers.drivers.QDevil import QDAC2
    try:
        qdac = Instrument.find_instrument(name)
    except KeyError:
        qdac = QDAC2.QDac2(name, visalib='@py', address=f'TCPIP::{qdac_ip}::5025::SOCKET')
    external_voltage_mapping = {
        machine.quantum_dots["virtual_dot_1"].physical_channel: qdac.ch01.dc_constant_V, 
        machine.quantum_dots["virtual_dot_2"].physical_channel: qdac.ch02.dc_constant_V, 
        machine.sensor_dots["virtual_sensor_1"].physical_channel: qdac.ch08.dc_constant_V, 
        machine.sensor_dots["virtual_sensor_2"].physical_channel: qdac.ch09.dc_constant_V
    }
    machine.connect_to_external_source(external_voltage_mapping)

    from qcodes.parameters import DelegateParameter
    voltage_parameters = []
    physical_channels = machine.physical_channels
    for ch in list(physical_channels.values()): 
        voltage_parameters.append(DelegateParameter(
            name = ch.id, label = ch.id, source = ch.offset_parameter
        ))
    voltage_control_component = VoltageControlComponent(component_id="Voltage_Control",voltage_parameters=voltage_parameters,update_interval_ms=1000)
    voltage_control_tab = VoltageControlTabController(voltage_control_component = voltage_control_component)




# %%
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

# Select the scan mode (how the 2D grid is traversed in QUA)
# Options include: RasterScan, SpiralScan, SwitchRasterScan
scan_mode_dict = {
    "Switch_Raster_Scan": scan_modes.SwitchRasterScan(), 
    "Raster_Scan": scan_modes.RasterScan(), 
    "Spiral_Scan": scan_modes.SpiralScan(),
}

# Instantiate the OPXDataAcquirer.
# This component handles the QUA program generation, execution, and data fetching.
data_acquirer = OPXDataAcquirer(
    qmm=qmm,
    machine=machine,
    gate_set=machine.virtual_gate_sets[virtual_gate_set_name],  # Replace with your GateSet instance
    x_axis_name="plunger_1",  # Must appear in gate_set.valid_channel_names; Virtual gate names also valid
    y_axis_name="plunger_2",  # Must appear in gate_set.valid_channel_names; Virtual gate names also valid
    scan_modes=scan_mode_dict,
    result_type="I",  # "I", "Q", "amplitude", or "phase"
    available_readout_pulses=[readout_pulse, readout_pulse2] # Input a list of pulses. The default only reads out from the first pulse, unless the second one is chosen in the UI. 
)

# ### Add post-processing functions as needed. Default post-processing functions are x- and y- derivative functions. 
# import xarray as xr
# data_acquirer.add_processing_function("Log10", lambda da: np.log10(np.abs(da)))

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

save_path = r"C:\Users\ ..."
# Instantiate the main VideoModeComponent, providing the configured data_acquirer.
video_mode_component = VideoModeComponent(
    data_acquirer=data_acquirer,
    data_polling_interval_s=0.5,  # How often the dashboard polls for new data
    voltage_control_tab = voltage_control_tab, 
    save_path = save_path
)

# ### If virtual gates editing/adding is required. 
# from qua_dashboards.virtual_gates import VirtualLayerEditor, ui_update
# virtual_gating_component = VirtualLayerEditor(gateset = gate_set, component_id = 'Virtual Gates UI')

# Build the Dash application layout using the VideoModeComponent.
app = build_dashboard(
    components=[video_mode_component],
    title="OPX Video Mode Dashboard",  # Title for the web page
)
# ### Use ui_update to keep video_mode_component up to date with virtual_gating_component changes
#ui_update(app, video_mode_component)
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

"""
Example Script: Video Mode with OPX1k

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
    * Review `inner_loop_action` (`VirtualGateInnerLoopAction`) and ensure the
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
from qm import QuantumMachinesManager
from quam.components import (
    BasicQuam,
    Channel, 
    Octave,
    pulses,
    StickyChannelAddon,
    SingleChannel, 
    InOutSingleChannel
)
from qua_dashboards.video_mode.inner_loop_actions.virtual_gating_inner_loop_action import VirtualGateInnerLoopAction
from quam_builder.architecture.quantum_dots.virtual_gates.virtual_gate_set import VirtualGateSet
from qua_dashboards.core import build_dashboard
from qua_dashboards.utils import setup_logging, BasicParameter
from qua_dashboards.video_mode import (
    SweepAxis,
    OPXDataAcquirer,
    scan_modes,
    VideoModeComponent,
)
logger = setup_logging(__name__)

lffem1 = 3
lffem2 = 5
mwfem = 1


# %% Create QUAM Machine Configuration and Connect to Quantum Machines Manager (QMM)

# Initialize a basic QUAM machine object.
# This object will be used to define the quantum hardware configuration (channels, pulses, etc.)
# and generate the QUA configuration for the OPX.

from quam.core import quam_dataclass
from dataclasses import field
from typing import Optional, Dict
@quam_dataclass
class GateSetQuam(BasicQuam):
    channels: Dict[str, Channel] = field(default_factory=dict)
    octaves: Dict[str, Octave] = field(default_factory=dict)
    gate_set: Optional[VirtualGateSet] = None
machine = GateSetQuam()

machine.channels['ch1'] = SingleChannel(
    id = 'Plunger1',
    opx_output=("con1", lffem2, 6),  # OPX controller and port
    sticky=StickyChannelAddon(duration=1_000, digital=False),
    operations={"step": pulses.SquarePulse(amplitude=0.1, length=1000)},
)
machine.channels['ch2'] = SingleChannel(
    id = 'Plunger2', 
    opx_output=("con1", lffem1, 8),  # OPX controller and port
    sticky=StickyChannelAddon(duration=1_000, digital=False),
    operations={"step": pulses.SquarePulse(amplitude=0.1, length=1000)},
)
machine.channels['ch3'] = SingleChannel(
    id = 'Plunger3', 
    opx_output=("con1", lffem1, 7),  # OPX controller and port
    sticky=StickyChannelAddon(duration=1_000, digital=False),
    operations={"step": pulses.SquarePulse(amplitude=0.1, length=1000)},
)
length = 100
readout_pulse = pulses.SquareReadoutPulse(id="readout", length=length, amplitude=0.1)
# Define the readout pulse and the channel used for measurement
machine.channels["ch1_readout"] = InOutSingleChannel(
    id = 'Sensor1', 
    opx_output=("con1", lffem1, 1),  # Output for the readout pulse
    opx_input=("con1", lffem1, 1),  # Input for acquiring the measurement signal
    intermediate_frequency=200e6,  # Set IF for the readout channel
    operations={"readout": readout_pulse, 
                "step": pulses.SquarePulse(amplitude=0.1, length=length)},
    time_of_flight=32,  
)

### Right now, the .get_reference() is necessary to map the channels. .name is used to map consistently with the SweepAxis set up later
channels = {machine.channels['ch1'].name: machine.channels['ch1'].get_reference(),
            machine.channels['ch2'].name: machine.channels['ch2'].get_reference(),
            machine.channels['ch3'].name: machine.channels['ch3'].get_reference(),
            machine.channels['ch1_readout'].name: machine.channels['ch1_readout'].get_reference()}
readout = {'Resonator': machine.channels['ch1_readout'].get_reference()}

machine.gate_set = VirtualGateSet(id = 'Plungers', channels=channels)
machine.gate_set.add_layer(
    source_gates = ['vPlunger1', 'vPlunger2', 'vPlunger3', 'vSensor1'], 
    target_gates = ['Plunger1', 'Plunger2', 'Plunger3', 'Sensor1'], 
    matrix = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
)

# --- QMM Connection ---
# Replace with your actual OPX host and cluster name
qmm = QuantumMachinesManager(host=host_ip, cluster_name=cluster_name)

# Generate the QUA configuration from the QUAM machine object
config = machine.generate_config()

# Open a connection to the Quantum Machine (QM)
# This prepares the OPX with the generated configuration.
qm = qmm.open_qm(config, close_other_machines=True)


# %% Configure Video Mode Components

# Define BasicParameters for X and Y voltage offsets.
x_offset = BasicParameter(name="2DX Voltage Offset", initial_value=0.0)
y_offset = BasicParameter(name="2DY Voltage Offset", initial_value=0.0)
x_span = 0.06
y_span = 0.03
x_resolution = 101
y_resolution = 101

# Define the action to be performed at each point in the QUA scan (inner loop).
inner_loop_action = VirtualGateInnerLoopAction(
    x_element = 'vPlunger1', 
    y_element = 'vPlunger2',
    gateset=machine.gate_set,
    ramp_rate = 0,
    readout_pulse=readout_pulse
)

# Select the scan mode (how the 2D grid is traversed in QUA)
# Options include: RasterScan, SpiralScan, SwitchRasterScan
scan_mode = scan_modes.SwitchRasterScan()

# Instantiate the OPXQDACDataAcquirer.
# This component handles the QUA program generation, execution, and data fetching.
data_acquirer = OPXDataAcquirer(
    qmm=qmm,
    machine=machine,
    qua_inner_loop_action=inner_loop_action,
    scan_mode=scan_mode,
    x_axis=SweepAxis(name = 'vPlunger1', label = 'vPlunger1', span=x_span, points=x_resolution, offset_parameter=x_offset),
    y_axis=SweepAxis(name = 'vPlunger2', label = 'vPlunger2', span=y_span, points=y_resolution, offset_parameter=y_offset),
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
video_mode_component = VideoModeComponent(
    data_acquirer=data_acquirer,
    data_polling_interval_s=0.2, 
    machine = machine, 
)

app = build_dashboard(
    components=[video_mode_component],
    title="Combined Dashboard",
)


logger.info("Dashboard built. Starting Dash server on http://localhost:8050")
# Run the Dash server.
# `host="0.0.0.0"` makes it accessible on your network.
# `use_reloader=False` is often recommended for stability with background threads.
app.run(debug=True, host="127.0.0.1", port=8050, use_reloader=False)


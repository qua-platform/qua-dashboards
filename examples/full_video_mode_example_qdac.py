"""
Example Script: Video Mode with OPX with Virtual Gating

This script demonstrates how to use the VideoModeComponent with an OPXDataAcquirer
to perform live 2D scans on a quantum device. It sets up a QUA program to sweep
two DC voltage channels and measure a readout signal, displaying the results in a
real-time dashboard.

Quick How-to-Use:
1.  **Configure Hardware**:
    * Update `qmm = QuantumMachinesManager(...)` with your OPX host and cluster_name.
    * Currently, a BasicQuam() machine is defined. 
        Modify the `machine = BasicQuam()` section to define the QUA elements.
        (channels, pulses) that match your experimental setup.
        Ensure `ch1`, `ch2` etc (for sweeping) and `ch1_readout` (or your measurement
        channel) are correctly defined.
2.  **Add/Adjust your virtual gates**: 
    * This script assumes a single layer of virtual gates. Adjust and add virtual gates 
        as necessary. 
    * Be sure to adjust the virtual gating matrices to suit your experimental needs. 
        This can be adjusted via the UI.
3.  **Configure the DC Control**
    * Adjust the VoltageControlComponent to suit your experiment
    * Ensure that each Quam channel is mapped to the correct output in the VoltageControlComponent. 
    * This example assumes the use of a QM QDAC, however is flexible for any voltage source.
4.  **Define your Readout Pulses**: 
    * First instantiate the relevant readout pulses
    * When creating your readout Quam channel, ensure that each readout pulse is correctly 
        and uniquely mapped to your readout elements. 
    * Pass the readout pulses to the data_acquirer instance as a list.
5.  **Adjust Scan Parameters**:
    * Select a `scan_mode` (e.g., `SwitchRasterScan`, `RasterScan`).
    * Set `result_type` in `OPXDataAcquirer` (e.g., "I", "Q", "amplitude", "phase").
6.  **Set a save_path to save Quam State JSON in the right directory**
7.  **Run the Script**: Execute this Python file.
8.  **Open Dashboard**: Navigate to `http://localhost:8050` (or the address shown
    in your terminal) in a web browser to view the live video mode dashboard.

Note: The sections for "(Optional) Run program and acquire data" and "DEBUG: Generate QUA script"
and "Test simulation" are for direct execution/debugging and can be commented out
if you only intend to run the live dashboard.
"""

# %% Imports
from qm import QuantumMachinesManager
from quam.components import (
    BasicQuam,
    InOutSingleChannel,
    pulses,
    StickyChannelAddon,
)
from quam.core import QuamRoot
from typing import List
from qua_dashboards.core import build_dashboard
from qua_dashboards.utils import setup_logging
from qua_dashboards.video_mode import (
    OPXDataAcquirer,
    scan_modes,
    VideoModeComponent,
)
from qcodes_contrib_drivers.drivers.QDevil import QDAC2
from quam_builder.architecture.quantum_dots.components import VoltageGate
from quam_builder.architecture.quantum_dots.components import VirtualGateSet
from qua_dashboards.virtual_gates import VirtualLayerEditor, ui_update
from qua_dashboards.voltage_control import VoltageControlComponent
from qcodes.parameters import DelegateParameter


def connect_to_qdac(address): 
    qdac = QDAC2.QDac2('QDAC', visalib='@py', address=f'TCPIP::{address}::5025::SOCKET')
    return qdac

def setup_DC_channel(machine: QuamRoot, name: str, opx_output_port: int , qdac: QDAC2.QDac2, qdac_port: int, con = "con1", fem: int = None): 
    """
    Set up a DC Channel 

    Args: 
        machine: Your Quam machine instance.
        name: The channel name in your Quam.
        opx_ouput_port: The integer output port of your OPX.
        qdac: your QCodes Qdac instance. Replace with external source object (with its typing) if desired.
        qdac_port: Integer Qdac output port.
        con: QM cluster controller, defaults to "con1". 
        fem: If using an OPX1000, add integer FEM number. Defaults to None for OPX+. 
    """
    if fem is None: 
        opx_output = (con, opx_output_port)
    else: 
        opx_output = (con, fem, opx_output_port)

    machine.channels[name] = VoltageGate(
        opx_output = opx_output, # Output for channel
        sticky=StickyChannelAddon(duration=1_000, digital=False),  # For DC offsets
        operations={"half_max_square": pulses.SquarePulse(amplitude=0.25, length=1000)}, # This operation is necessary - although OPXDataAcquirer ensures this too
    )
    machine.channels[name].offset_parameter = qdac.channel(qdac_port).dc_constant_V
    return machine.channels[name].get_reference()


def setup_readout_channel(machine: QuamRoot, name: str, readout_pulse: pulses.ReadoutPulse, opx_output_port: int, opx_input_port: int, IF: float, con = "con1", fem: int = None):
    """
    Set up a Readout Channel 

    Args: 
        machine: Your Quam machine instance.
        name: The channel name in your Quam.
        readout_pulse: The Readout Pulse object to be passed to the OPXDataAcquirer.
        opx_ouput_port: The integer output port of your OPX.
        opx_input_port: The integer input port of your OPX.
        IF: The intermediate frequency of your Readout channel.
        con: QM cluster controller, defaults to "con1". 
        fem: If using an OPX1000, add integer FEM number. Defaults to None for OPX+. Assumed same FEM for output and input channels.
    """
    
    if fem is None: 
        opx_output = (con, opx_output_port)
        opx_input = (con, opx_input_port)
    else: 
        opx_output = (con, fem, opx_output_port)
        opx_input = con, fem, opx_input_port
    
    machine.channels[name] = InOutSingleChannel(
        opx_output=opx_output,  # Output for the readout pulse
        opx_input=opx_input,  # Input for acquiring the measurement signal
        intermediate_frequency=IF,  # Set IF for the readout channel
        operations={"readout": readout_pulse},  # Assign the readout pulse to this channel
        sticky=StickyChannelAddon(duration=1_000, digital=False),  # For DC offsets
        time_of_flight = 28
    )
    return machine.channels[name].get_reference()

def define_DC_params(machine: QuamRoot, gate_names: List[str]):
    """
    Defines gates using QDAC and a channel mapping dict. Provide a list of channel names existing in your Quam object instance.

    Currently assumes VoltageGate objects, using 'offset_parameter" attribute.
    """
    voltage_parameters = []
    for ch_name in gate_names:
        ch = machine.channels[ch_name]
        parameter = getattr(ch, "offset_parameter", None)
        if parameter is not None: 
            voltage_parameters.append(DelegateParameter(
                name = ch_name, label = ch_name, source = ch.offset_parameter
            ))
    return voltage_parameters

def main():
    logger = setup_logging(__name__)

    # Adjust the IP and cluster name here
    qm_ip = "127.0.0.1"
    cluster_name = "CS_1"
    qmm = QuantumMachinesManager(host=qm_ip, cluster_name=cluster_name)
    machine = BasicQuam()

    qdac_ip = "172.16.33.101"
    logger.info("Connecting to QDAC")
    qdac = connect_to_qdac(qdac_ip)

    # Define your readout pulses here. Each pulse should be uniquely mapped to your readout elements. 
    readout_pulse_ch1 = pulses.SquareReadoutPulse(id="readout", length=1500, amplitude=0.1)
    readout_pulse_ch2 = pulses.SquareReadoutPulse(id="readout", length=1500, amplitude=0.1)

    # Choose the FEM. For OPX+, keep fem = None. 
    fem = None

    channel_mapping = {
        "ch1": setup_DC_channel(machine, "ch1", 1, qdac, 1, fem = fem), 
        "ch2": setup_DC_channel(machine, "ch2", 2, qdac, 2, fem = fem), 
        "ch3": setup_DC_channel(machine, "ch3", 3, qdac, 3, fem = fem), 
        "ch1_readout": setup_readout_channel(machine, "ch1_readout", readout_pulse_ch1, opx_output_port = 4, opx_input_port = 1, IF = 1.7e8, fem = fem), 
        "ch2_readout": setup_readout_channel(machine, "ch2_readout", readout_pulse_ch2, opx_output_port = 4, opx_input_port = 1, IF = 2.3e8, fem = fem), 
    }

    # Adjust or add your virtual gates here. This example assumes a single virtual gating layer, add more if necessary. 
    logger.info("Creating VirtualGateSet")
    virtual_gate_set = VirtualGateSet(id = "Plungers", channels = channel_mapping)

    virtual_gate_set.add_layer(
        source_gates = ["V1", "V2"], # Pick the virtual gate names here 
        target_gates = ["ch1", "ch2"], # Must be a subset of gates in the gate_set
        matrix = [[1, 0.2], [0.2, 1]] # Any example matrix
    )

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
        gate_set=virtual_gate_set,  # Replace with your GateSet instance
        x_axis_name="ch1",  # Must appear in gate_set.valid_channel_names; Virtual gate names also valid
        y_axis_name="ch2",  # Must appear in gate_set.valid_channel_names; Virtual gate names also valid
        scan_modes=scan_mode_dict,
        result_type="I",  # "I", "Q", "amplitude", or "phase"
        available_readout_pulses=[readout_pulse_ch1, readout_pulse_ch2] # Input a list of pulses. The default only reads out from the first pulse, unless the second one is chosen in the UI. 
    )

    video_mode_component = VideoModeComponent(
        data_acquirer=data_acquirer,
        data_polling_interval_s=0.5,  # How often the dashboard polls for new data
        save_path = r"C:\Users\..."
    )

    virtual_gating_component = VirtualLayerEditor(gateset = virtual_gate_set, component_id = 'virtual-gates-ui')

    voltage_parameters = define_DC_params(machine, ["ch1", "ch2", "ch3"])

    # Currently no Quam channel to combine readout + DC control. Relevant QUAM components to come in the future
    voltage_parameters.extend([
        DelegateParameter(name = "ch1_readout", label = "ch1_readout", source = qdac.channel(4).dc_constant_V), 
        DelegateParameter(name = "ch2_readout", label = "ch2_readout", source = qdac.channel(5).dc_constant_V)
        ])

    voltage_controller = VoltageControlComponent(
        component_id="Voltage_Control",
        voltage_parameters=voltage_parameters,
        update_interval_ms=1000,
    )

    app = build_dashboard(
        components=[video_mode_component, voltage_controller, virtual_gating_component],
        title="OPX Video Mode Dashboard",  # Title for the web page
    )
    # Helper function to keep UI updated with Virtual Layer changes
    ui_update(app, video_mode_component)

    logger.info("Dashboard built. Starting Dash server on http://localhost:8050")
    app.run(debug=True, host="0.0.0.0", port=8050, use_reloader=False)


if __name__ == "__main__":
    main()

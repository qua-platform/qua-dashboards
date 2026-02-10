"""
Example Script: Video Mode with OPX with Virtual Gating using BaseQuamQD

BaseQuamQD is a Quam infrastructure provided by Quantum Machines to aid in the calibration
of quantum dot devices. This script demonstrates how to use this infrastructure to create an
instance of video mode with the correct parameters, to perform live 2D scans on a quantum dot 
device. 

Quick How-to-Use:
1.  **Configure Hardware**:
    * Update the `host_ip` and `cluster_name` with your OPX IP and cluster name. 
    * Currently, this example defines an entirely new BaseQuamQD and populates it. If you 
    already have a quam_state you intend to use, then simply load that Quam state. 
        Modify the `machine = BaseQuamQD()` section to define the QUA elements.
        (channels, pulses) that match your experimental setup.
        Ensure `plunger_1`, `plunger_2` etc (for sweeping) and `sensor_1` (or your measurement
        channel) are correctly defined.
2.  **Add/Adjust your virtual gates**:
    * This script adds two virtualization layers. The first is the cross-compensation layer, which 
    includes all the channels in your machine. The goal of this layer is to create an orthogonalized 
    voltage space for the device components (quantum dots, barrier gates, sensor dots). The matrix
    can be updated sub-matrix-wise using the command `update_cross_compensation_submatrix()`.
    * An optional second layer is defined as the detuning axes of the QuantumDot components. 
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
"""

# %% Imports
from qm import QuantumMachinesManager
from quam.components import (
    pulses,
    BasicQuam, 
)
from quam_builder.architecture.quantum_dots.components import VirtualGateSet
from qua_dashboards.core import build_dashboard
from qua_dashboards.utils import setup_logging
from qua_dashboards.video_mode import (
    OPXDataAcquirer,
    scan_modes,
    VideoModeComponent,
)
from qua_dashboards.virtual_gates import VirtualLayerEditor, ui_update
from qua_dashboards.utils import setup_DC_channel, setup_readout_channel

def main():
    logger = setup_logging(__name__)

    ###########################################################################
    ############ CREATE QUAM MACHINE (SKIP IF EXISTING QUAM STATE) ############
    ###########################################################################

    # ### If Quam state exists, instead simply load it below: 
    # machine = BasicQuam.load()

    # Adjust the IP and cluster name here
    qm_ip = "172.16.33.101"
    cluster_name = "CS_2"

    qmm = QuantumMachinesManager(host=qm_ip, cluster_name=cluster_name)
    machine = BasicQuam()

    # Define your readout pulses here. Each pulse should be uniquely mapped to your readout elements. 
    readout_pulse_ch1 = pulses.SquareReadoutPulse(id="readout", length=100, amplitude=0.1)
    readout_pulse_ch2 = pulses.SquareReadoutPulse(id="readout", length=100, amplitude=0.1)

    # Choose the FEM. For OPX+, keep fem = None. 
    fem = None

    # Set up the readout channels
    machine.channels["ch1_readout"] = setup_readout_channel(name = "ch1_readout", readout_pulse=readout_pulse_ch1, opx_output_port = 6, opx_input_port = 1, IF = 150e6, fem = fem)
    machine.channels["ch2_readout"] = setup_readout_channel(name = "ch2_readout", readout_pulse=readout_pulse_ch2, opx_output_port = 6, opx_input_port = 1, IF = 250e6, fem = fem)


    channel_mapping = {
        "ch1": setup_DC_channel(name = "ch1", opx_output_port = 1, qdac_port = 1, fem = fem), 
        "ch2": setup_DC_channel(name = "ch2", opx_output_port = 2, qdac_port = 2, fem = fem), 
        "ch3": setup_DC_channel(name = "ch3", opx_output_port = 3, qdac_port = 3, fem = fem), 
        "ch1_readout_DC": setup_DC_channel(name = "ch1_readout_DC", opx_output_port = 4, qdac_port = 3, fem = fem), 
        "ch2_readout_DC": setup_DC_channel(name = "ch2_readout_DC", opx_output_port = 5, qdac_port = 3, fem = fem), 
    }
    for ch_name, ch in channel_mapping.items(): 
        machine.channels[ch_name] = ch

    # Adjust or add your virtual gates here. This example assumes a single virtual gating layer, add more if necessary. 
    logger.info("Creating VirtualGateSet")
    virtual_gate_set = VirtualGateSet(id = "Plungers", channels = {ch_name: ch.get_reference() for ch_name, ch in channel_mapping.items()})

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
        available_readout_pulses=[readout_pulse_ch1, readout_pulse_ch2], # Input a list of pulses. The default only reads out from the first pulse, unless the second one is chosen in the UI. 
        acquisition_interval_s=0.05, 
        voltage_control_component=None
    )

    virtual_gating_component = VirtualLayerEditor(gateset = virtual_gate_set, component_id = 'virtual-gates-ui', dc_set = None)

    video_mode_component = VideoModeComponent(
        data_acquirer=data_acquirer,
        data_polling_interval_s=0.1,  # How often the dashboard polls for new data
        voltage_control_tab = None,
        save_path = r"C:\Users\..."
    )
    components = [video_mode_component, virtual_gating_component]

    app = build_dashboard(
        components = components,
        title = "OPX Video Mode Dashboard",  # Title for the web page
    )
    # Helper function to keep UI updated with Virtual Layer changes
    ui_update(app, video_mode_component, voltage_control_component=None)

    logger.info("Dashboard built. Starting Dash server on http://localhost:8050")
    app.run(debug=True, host="0.0.0.0", port=8050, use_reloader=False)

if __name__ == "__main__":
    main()

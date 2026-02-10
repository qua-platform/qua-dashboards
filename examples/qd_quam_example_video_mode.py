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
)
from qua_dashboards.core import build_dashboard
from qua_dashboards.utils import setup_logging
from qua_dashboards.video_mode import (
    OPXDataAcquirer,
    scan_modes,
    VideoModeComponent,
)
from quam_builder.architecture.quantum_dots.qpu import BaseQuamQD
from qua_dashboards.virtual_gates import VirtualLayerEditor, ui_update
from qua_dashboards.voltage_control import VoltageControlComponent
from qua_dashboards.utils import setup_DC_channel, setup_readout_channel

def main():
    logger = setup_logging(__name__)

    ###########################################################################
    ############ CREATE QUAM MACHINE (SKIP IF EXISTING QUAM STATE) ############
    ###########################################################################

    # ### If Quam state exists, instead simply load it below: 
    # machine = BaseQuamQD.load()

    # Adjust the IP and cluster name here
    qm_ip = "172.16.33.115"
    cluster_name = "CS_4"

    # If connecting to qdac, set qdac_connect = True, and the qdac_ip.
    qdac_ip = "172.16.33.101"
    qdac_connect = True

    qmm = QuantumMachinesManager(host=qm_ip, cluster_name=cluster_name)
    machine = BaseQuamQD()

    # Define your readout pulses here. Each pulse should be uniquely mapped to your readout elements.
    readout_pulse_ch1 = pulses.SquareReadoutPulse(
        id="readout", length=100, amplitude=0.1
    )
    readout_pulse_ch2 = pulses.SquareReadoutPulse(
        id="readout", length=100, amplitude=0.1
    )

    # Choose the FEM. For OPX+, keep fem = None.
    fem = 5

    # Set up the DC channels
    p1 = setup_DC_channel(name="plunger_1", opx_output_port=1, qdac_port=1, fem=fem)
    p2 = setup_DC_channel(name="plunger_2", opx_output_port=2, qdac_port=2, fem=fem)
    p3 = setup_DC_channel(name="plunger_3", opx_output_port=3, qdac_port=3, fem=fem)
    p4 = setup_DC_channel(name="plunger_4", opx_output_port=8, qdac_port=8, fem=fem)
    s1 = setup_DC_channel(name="sensor_1", opx_output_port=4, qdac_port=4, fem=fem)
    s2 = setup_DC_channel(name="sensor_2", opx_output_port=5, qdac_port=5, fem=fem)
    b1 = setup_DC_channel(name="barrier_1", opx_output_port=6, qdac_port=6, fem=fem)
    b2 = setup_DC_channel(name="barrier_2", opx_output_port=7, qdac_port=7, fem=fem)

    # Set up the readout channels
    sensor_readout_channel_1 = setup_readout_channel(
        name="readout_resonator_1",
        readout_pulse=readout_pulse_ch1,
        opx_output_port=6,
        opx_input_port=1,
        IF=150e6,
        fem=fem,
    )
    sensor_readout_channel_2 = setup_readout_channel(
        name="readout_resonator_2",
        readout_pulse=readout_pulse_ch2,
        opx_output_port=6,
        opx_input_port=1,
        IF=250e6,
        fem=fem,
    )

    # Adjust or add your virtual gates here. This example assumes a single virtual gating layer, add more if necessary.
    logger.info("Creating VirtualGateSet")
    machine.create_virtual_gate_set(
        gate_set_id="main_qpu",
        virtual_channel_mapping={
            "virtual_dot_1": p1,
            "virtual_dot_2": p2,
            "virtual_dot_3": p3,
            "virtual_dot_4": p4,
            "virtual_barrier_1": b1,
            "virtual_barrier_2": b2,
            "virtual_sensor_1": s1,
            "virtual_sensor_2": s2,
        },
        adjust_for_attenuation=False,
    )

    machine.register_channel_elements(
        plunger_channels = [p1, p2, p3, p4],
        barrier_channels = [b1, b2],
        sensor_resonator_mappings = {
            s1: sensor_readout_channel_1, 
            s2: sensor_readout_channel_2,
        },
    )

    # Register the quantum dot pairs
    machine.register_quantum_dot_pair(
        id = "dot1_dot2_pair",
        quantum_dot_ids = ["virtual_dot_1", "virtual_dot_2"], 
        sensor_dot_ids = ["virtual_sensor_1"], 
        barrier_gate_id = "virtual_barrier_1"
    )

    machine.register_quantum_dot_pair(
        id = "dot3_dot4_pair",
        quantum_dot_ids = ["virtual_dot_3", "virtual_dot_4"], 
        sensor_dot_ids = ["virtual_sensor_1"],
        barrier_gate_id = "virtual_barrier_2"
    )

    if qdac_connect:
        logger.info("Connecting to QDAC")
        machine.network.update({"qdac_ip": qdac_ip})
        machine.connect_to_external_source(external_qdac=True)
        machine.create_virtual_dc_set("main_qpu")

    # Update Cross Capacitance matrix values
    machine.update_cross_compensation_submatrix(
        virtual_names=["virtual_barrier_1", "virtual_barrier_2"],
        channels=[p3],
        matrix=[[0.1, 0.5]],
        target = "both"
    )

    machine.update_cross_compensation_submatrix(
        virtual_names=["virtual_dot_1", "virtual_dot_2", "virtual_dot_3"],
        channels=[p1, p2, p3],
        matrix=[
            [1, 0.1, 0.1],
            [0.2, 1, 0.6],
            [0.1, 0.3, 1],
        ],
        target = "both"
    )

    machine.update_cross_compensation_submatrix(
        virtual_names=["virtual_dot_1", "virtual_dot_2", "virtual_dot_3"],
        channels=[b1, b2, s1, s2],
        matrix=[
            [0.15, 0.4, 0.1 ],
            [0.1 , 0.2, 0.2 ],
            [0.2 , 0.2, 0.1 ],
            [0.2 , 0.3, 0.25],
        ],
        target = "both"
    )

    # Define the detuning axes for both QuantumDotPairs
    machine.quantum_dot_pairs["dot1_dot2_pair"].define_detuning_axis(
        matrix = [[1,-1]], 
        detuning_axis_name = "dot1_dot2_pair_epsilon"
    )

    machine.quantum_dot_pairs["dot3_dot4_pair"].define_detuning_axis(
        matrix = [[1,-1]], 
        detuning_axis_name = "dot3_dot4_pair_epsilon"
    )

    ################################################
    ############ INSTANTIATE VIDEO MODE ############
    ################################################

    scan_mode_dict = {
        "Switch_Raster_Scan": scan_modes.SwitchRasterScan(),
        "Raster_Scan": scan_modes.RasterScan(),
        "Spiral_Scan": scan_modes.SpiralScan(),
    }

    virtual_gating_component = VirtualLayerEditor(
        gateset=machine.virtual_gate_sets["main_qpu"], component_id="virtual-gates-ui", dc_set = machine.virtual_dc_sets["main_qpu"]
    )

    # External Voltage Control Component, if qdac_connect = True
    voltage_control_tab, voltage_control_component = None, None
    if qdac_connect:
        voltage_control_component = VoltageControlComponent(
            component_id="Voltage_Control",
            dc_set = machine.virtual_dc_sets["main_qpu"],
            update_interval_ms=1000,
            preselected_gates=["plunger_1", "plunger_2", "virtual_dot_1", "virtual_dot_2"]
        )
        from qua_dashboards.video_mode.tab_controllers import (
            VoltageControlTabController,
        )

        voltage_control_tab = VoltageControlTabController(
            voltage_control_component=voltage_control_component
        )


    # Instantiate the OPXDataAcquirer.
    # This component handles the QUA program generation, execution, and data fetching.
    data_acquirer = OPXDataAcquirer(
        qmm=qmm,
        machine=machine,
        gate_set=machine.virtual_gate_sets[
            "main_qpu"
        ],  # Replace with your GateSet instance
        x_axis_name="virtual_dot_1",  # Must appear in gate_set.valid_channel_names; Virtual gate names also valid
        y_axis_name="virtual_dot_2",  # Must appear in gate_set.valid_channel_names; Virtual gate names also valid
        scan_modes=scan_mode_dict,
        result_type="I",  # "I", "Q", "amplitude", or "phase"
        available_readout_pulses=[
            readout_pulse_ch1,
            readout_pulse_ch2,
        ],  # Input a list of pulses. The default only reads out from the first pulse, unless the second one is chosen in the UI.
        acquisition_interval_s=0.01,
        voltage_control_component=voltage_control_component,
    )

    video_mode_component = VideoModeComponent(
        data_acquirer=data_acquirer,
        data_polling_interval_s=0.2,  # How often the dashboard polls for new data
        voltage_control_tab=voltage_control_tab,
        save_path=r"C:\Users\...",
    )
    components = [video_mode_component, virtual_gating_component]

    app = build_dashboard(
        components=components,
        title="OPX Video Mode Dashboard",  # Title for the web page
    )
    # Helper function to keep UI updated with Virtual Layer changes
    ui_update(app, video_mode_component, voltage_control_component)

    logger.info("Dashboard built. Starting Dash server on http://localhost:8050")
    app.run(debug=True, host="0.0.0.0", port=8050, use_reloader=False)


if __name__ == "__main__":
    main()

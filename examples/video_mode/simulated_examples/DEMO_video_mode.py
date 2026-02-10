"""
Example Script: Simulated Video Mode with OPX output and Virtual Gating

This script demonstrates how to use the VideoModeComponent with a SimulationDataAcquirerOPXOutput
to perform simulated 2D scans on a quantum device, while outputting a real OPX signal to inspect
on the scope. It sets up a QUA program to sweep two DC voltage channels and measure a readout signal, 
displaying the results in a real-time dashboard.

Quick How-to-Use:
1.  **Configure Hardware**:
    * Update `qmm = QuantumMachinesManager(...)` with your OPX host and cluster_name.
    * Currently, a BaseQuamQD() machine is defined.
        Modify the `machine = BaseQuamQD()` section to define the QUA elements.
        (channels, pulses) that match your experimental setup.
        Ensure `plunger_1`, `plunger_2` etc (for sweeping) and `sensor_1` (or your measurement
        channel) are correctly defined.
2.  **Add/Adjust your virtual gates**:
    * This script uses `BaseQuamQD`, which assumes a first, cross-compensation layer 
        of virtual gates, and an optional second layer of detuning axes. 
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
    * Set `result_type` in `SimulationDataAcquirerOPXOutput` (e.g., "I", "Q", "amplitude", "phase").
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
    pulses,
)
from qua_dashboards.core import build_dashboard
from qua_dashboards.utils import setup_logging
from qua_dashboards.video_mode import (
    scan_modes,
    VideoModeComponent,
)
from qua_dashboards.video_mode.data_acquirers.simulation_data_acquirer import SimulationDataAcquirerOPXOutput
from quam_builder.architecture.quantum_dots.qpu import BaseQuamQD
from qua_dashboards.virtual_gates import VirtualLayerEditor, ui_update
from qua_dashboards.voltage_control import VoltageControlComponent
from qua_dashboards.utils import define_DC_params, setup_DC_channel, setup_readout_channel

def main():
    logger = setup_logging(__name__)

    # Adjust the IP and cluster name here
    qm_ip = "172.16.33.115"
    cluster_name = "CS_3"

    # If connecting to qdac, set qdac_connect = True, and the qdac_ip.
    qdac_ip = "172.16.33.101"
    qdac_connect = True

    qmm = QuantumMachinesManager(host=qm_ip, cluster_name=cluster_name)
    machine = BaseQuamQD()

    # Define your readout pulses here. Each pulse should be uniquely mapped to your readout elements.
    readout_pulse_ch1 = pulses.SquareReadoutPulse(
        id="readout", length=100, amplitude=0.1
    )

    # Choose the FEM. For OPX+, keep fem = None.
    fem = 5

    # Set up the DC channels
    p1 = setup_DC_channel(name="plunger_1", opx_output_port=6, qdac_port=1, fem=fem)
    p2 = setup_DC_channel(name="plunger_2", opx_output_port=2, qdac_port=2, fem=fem)
    s1 = setup_DC_channel(name="sensor_1", opx_output_port=4, qdac_port=4, fem=fem)

    # Set up the readout channels
    sensor_readout_channel_1 = setup_readout_channel(
        name="readout_resonator_1",
        readout_pulse=readout_pulse_ch1,
        opx_output_port=3,
        opx_input_port=1,
        IF=150e6,
        fem=fem,
    )

    # Adjust or add your virtual gates here. This example assumes a single virtual gating layer, add more if necessary.
    logger.info("Creating VirtualGateSet")
    machine.create_virtual_gate_set(
        gate_set_id="main_qpu",
        virtual_channel_mapping={
            "virtual_dot_1": p1,
            "virtual_dot_2": p2,
            "virtual_sensor_1": s1,
        },
        adjust_for_attenuation=False,
    )

    machine.register_channel_elements(
        plunger_channels = [p1, p2],
        barrier_channels = [],
        sensor_resonator_mappings = {
            s1: sensor_readout_channel_1, 
        },
    )

    # Register the quantum dot pairs
    machine.register_quantum_dot_pair(
        id = "dot1_dot2_pair",
        quantum_dot_ids = ["virtual_dot_1", "virtual_dot_2"], 
        sensor_dot_ids = ["virtual_sensor_1"], 
        barrier_gate_id = None
    )

    if qdac_connect:
        logger.info("Connecting to QDAC")
        machine.network.update({"qdac_ip": qdac_ip})
        machine.connect_to_external_source(external_qdac=True)
        machine.create_virtual_dc_set("main_qpu")

    # Define the detuning axes for both QuantumDotPairs
    machine.quantum_dot_pairs["dot1_dot2_pair"].define_detuning_axis(
        matrix = [[1,-1]], 
        detuning_axis_name = "dot1_dot2_pair_epsilon"
    )

    scan_mode_dict = {
        "Switch_Raster_Scan": scan_modes.SwitchRasterScan(),
        "Raster_Scan": scan_modes.RasterScan(),
        "Spiral_Scan": scan_modes.SpiralScan(),
    }

    virtual_gating_component = VirtualLayerEditor(
        gateset=machine.virtual_gate_sets["main_qpu"], component_id="virtual-gates-ui", dc_set = machine.virtual_dc_sets["main_qpu"]
    )

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


    matrix = [
    [0.12, 0.02, 0.005], 
    [0.02, 0.11, 0.005],

    ]
    import numpy as np
    machine.update_cross_compensation_submatrix(
        virtual_names = ["virtual_dot_1", "virtual_dot_2"], 
        channels = [p1, p2, s1], 
        matrix = np.linalg.pinv(matrix).tolist()
    )
    from qarray import ChargeSensedDotArray, WhiteNoise, TelegraphNoise, LatchingModel
    Cdd = [
        [0.10, 0.04],
        [0.04, 0.10],
    ]
    Cgd = matrix   
    Cds = [
        [0.015, 0.012], 
    ]

    Cgs = [
        [0.002, 0.002, 0.10],
    ]

    model = ChargeSensedDotArray(
        Cdd=Cdd,
        Cgd=Cgd,
        Cds=Cds,
        Cgs=Cgs,
        coulomb_peak_width=0.05,
        T=50,
        algorithm="default",
        implementation="jax", 
        noise_model=WhiteNoise(amplitude=5e-3) + TelegraphNoise(
            amplitude=2e-3, 
            p01=1e-3, 
            p10=1e-3 
        ),
        latching_model=LatchingModel(
            n_dots=2, 
            p_leads=0.98,
            p_inter=0.01 
        ),
    )
    from qua_dashboards.video_mode.inner_loop_actions.simulators import QarraySimulator
    simulator = QarraySimulator(
        gate_set = machine.virtual_gate_sets["main_qpu"], 
        dc_set = machine.virtual_dc_sets["main_qpu"],
        model = model,
        sensor_gate_names = ("virtual_dot_1", "virtual_dot_2"), 
        n_charges = [1, 1, 3],
    )


    data_acquirer = SimulationDataAcquirerOPXOutput(
        qmm = qmm,
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
        ],  # Input a list of pulses. The default only reads out from the first pulse, unless the second one is chosen in the UI.
        acquisition_interval_s=0.01,
        voltage_control_component=voltage_control_component,
        simulator = simulator,
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

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
    InOutSingleChannel,
    pulses,
    StickyChannelAddon,
)
from quam.components.ports import (
    LFFEMAnalogOutputPort,
    LFFEMAnalogInputPort,
    OPXPlusAnalogOutputPort,
    OPXPlusAnalogInputPort,
)
from quam.core import QuamRoot
from typing import List, Optional
from qua_dashboards.core import build_dashboard
from qua_dashboards.utils import setup_logging
from qua_dashboards.video_mode import (
    OPXDataAcquirer,
    scan_modes,
    VideoModeComponent,
    SimulationDataAcquirer,
)
from quam_builder.architecture.quantum_dots.components import (
    VoltageGate,
    VirtualGateSet,
    ReadoutResonatorSingle,
    QdacSpec,
)
from quam_builder.architecture.quantum_dots.qpu import BaseQuamQD
from qua_dashboards.virtual_gates import VirtualLayerEditor, ui_update
from qua_dashboards.voltage_control import VoltageControlComponent


def setup_DC_channel(
    name: str, opx_output_port: int, qdac_port: int, con="con1", fem: int = None
):
    """
    Set up a DC Channel

    Args:
        name: The channel name in your Quam.
        opx_ouput_port: The integer output port of your OPX.
        qdac_port: Integer Qdac output port.
        con: QM cluster controller, defaults to "con1".
        fem: If using an OPX1000, add integer FEM number. Defaults to None for OPX+.
    """
    if fem is None:
        opx_output = OPXPlusAnalogOutputPort(
            controller_id=con,
            port_id=opx_output_port,
        )
    else:
        opx_output = LFFEMAnalogOutputPort(
            controller_id=con,
            fem_id=fem,
            port_id=opx_output_port,
            upsampling_mode="pulse",
        )

    channel = VoltageGate(
        id=name,
        opx_output=opx_output,  # Output for channel
        sticky=StickyChannelAddon(duration=1_000, digital=False),  # For DC offsets
        attenuation = 10,
    )
    qdac_spec = QdacSpec(qdac_output_port = qdac_port)
    channel.qdac_spec = qdac_spec
    if qdac_port is None:
        channel.offset_parameter = None
    return channel


def setup_readout_channel(
    name: str,
    readout_pulse: pulses.ReadoutPulse,
    opx_output_port: int,
    opx_input_port: int,
    IF: float,
    con="con1",
    fem: int = None,
):
    """
    Set up a Readout Channel

    Args:
        name: The channel name in your Quam.
        readout_pulse: The Readout Pulse object to be passed to the OPXDataAcquirer.
        opx_ouput_port: The integer output port of your OPX.
        opx_input_port: The integer input port of your OPX.
        IF: The intermediate frequency of your Readout channel.
        con: QM cluster controller, defaults to "con1".
        fem: If using an OPX1000, add integer FEM number. Defaults to None for OPX+. Assumed same FEM for output and input channels.
    """

    if fem is None:
        opx_output = OPXPlusAnalogOutputPort(
            controller_id=con,
            port_id=opx_output_port,
        )
        opx_input = OPXPlusAnalogInputPort(
            controller_id=con,
            port_id=opx_input_port,
        )
    else:
        opx_output = LFFEMAnalogOutputPort(
            controller_id=con,
            fem_id=fem,
            port_id=opx_output_port,
            upsampling_mode="mw",
        )
        opx_input = LFFEMAnalogInputPort(
            controller_id=con,
            fem_id=fem,
            port_id=opx_input_port,
        )

    channel = ReadoutResonatorSingle(
        id=name,
        opx_output=opx_output,  # Output for the readout pulse
        opx_input=opx_input,  # Input for acquiring the measurement signal
        intermediate_frequency=IF,  # Set IF for the readout channel
        operations={
            "readout": readout_pulse
        },  # Assign the readout pulse to this channel
        time_of_flight=28,
    )
    return channel


def define_DC_params(machine: QuamRoot, gate_names: List[str]):
    """
    Defines gates using QDAC and a channel mapping dict. Provide a list of channel names existing in your Quam object instance.

    Currently assumes VoltageGate objects, using 'offset_parameter" attribute.
    """
    from qcodes.parameters import DelegateParameter

    voltage_parameters = []
    for ch_name in gate_names:
        ch = machine.physical_channels[ch_name]
        parameter = getattr(ch, "offset_parameter", None)
        if parameter is not None:
            voltage_parameters.append(
                DelegateParameter(
                    name=ch_name, label=ch_name, source=ch.offset_parameter
                )
            )
    return voltage_parameters


def main():
    logger = setup_logging(__name__)

    # Adjust the IP and cluster name here
    qm_ip = "172.16.33.115"
    cluster_name = "CS_4"

    # If connecting to qdac, set qdac_connect = True, and the qdac_ip.
    qdac_ip = "172.16.33.101"
    qdac_connect = True

    # qmm = QuantumMachinesManager(host=qm_ip, cluster_name=cluster_name)
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
    compensation_matrix = [  [1,0,0,0,0,0, 0.020406, 0.020406], 
                            [0,1,0,0,0,0, 0.029189, 0.029189], 
                            [0,0,1,0,0,0, 0.007986, 0.007986], 
                            [0,0,0,1,0,0, 0.010645, 0.010645], 
                            [0,0,0,0,1,0, 0.010643, 0.010643], 
                            [0,0,0,0,0,1, 0.0905586, 0.0905586], 
                            [0.020406, 0.029189, 0.007986, 0.010645, 0.010643, 0.0905586, 1.0, 0.0], 
                            [0.020406, 0.029189, 0.007986, 0.010645, 0.010643, 0.0905586, 0.0, 1.0] ]

    # compensation_matrix = np.eye(8).tolist()

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
        compensation_matrix=compensation_matrix,
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
    # machine.update_cross_compensation_submatrix(
    #     virtual_names=["virtual_barrier_1", "virtual_barrier_2"],
    #     channels=[p3],
    #     matrix=[[0.05, 0.05]],
    #     target = "both"
    # )

    # machine.update_cross_compensation_submatrix(
    #     virtual_names=["virtual_dot_1", "virtual_dot_2", "virtual_dot_3"],
    #     channels=[p1, p2, p3],
    #     matrix=[
    #         [1, 0.1, 0.0],
    #         [0.1, 1, 0.1],
    #         [0.0, 0.1, 1],
    #     ],
    #     target = "both"
    # )

    # machine.update_cross_compensation_submatrix(
    #     virtual_names=["virtual_dot_1", "virtual_dot_2", "virtual_dot_3"],
    #     channels=[b1, b2, s1, s2],
    #     matrix=[
    #         [0.08, 0.04, 0.00 ],
    #         [0.00 , 0.04, 0.08 ],
    #         [-0.4 , -0.4, -0.4],
    #         [-0.5 , -0.4, -0.4],
    #     ],
    #     target = "both"
    # )

    # Define the detuning axes for both QuantumDotPairs
    # machine.quantum_dot_pairs["dot1_dot2_pair"].define_detuning_axis(
    #     matrix = [[1,-1]], 
    #     detuning_axis_name = "dot1_dot2_pair_epsilon",
    #     set_dc_virtual_axis = qdac_connect
    # )

    # machine.quantum_dot_pairs["dot3_dot4_pair"].define_detuning_axis(
    #     matrix = [[1,-1]], 
    #     detuning_axis_name = "dot3_dot4_pair_epsilon",
    #     set_dc_virtual_axis = qdac_connect
    # )

    scan_mode_dict = {
        "Switch_Raster_Scan": scan_modes.SwitchRasterScan(),
        "Raster_Scan": scan_modes.RasterScan(),
        "Spiral_Scan": scan_modes.SpiralScan(),
    }

    virtual_gating_component = VirtualLayerEditor(
        gateset=machine.virtual_gate_sets["main_qpu"], component_id="virtual-gates-ui", dc_set = machine.virtual_dc_sets["main_qpu"] if qdac_connect else None
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



    from qarray import ChargeSensedDotArray, WhiteNoise, TelegraphNoise, LatchingModel
    Cdd = [
        [0.12, 0.08, 0.00, 0.00, 0.00, 0.00],
        [0.08, 0.13, 0.08, 0.00, 0.00, 0.00],
        [0.00, 0.08, 0.12, 0.08, 0.00, 0.00],
        [0.00, 0.00, 0.08, 0.12, 0.08, 0.00],
        [0.00, 0.00, 0.00, 0.08, 0.12, 0.08],
        [0.00, 0.00, 0.00, 0.00, 0.08, 0.11],
    ]
    Cgd = [
        [0.13, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.11, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.09, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.13, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.13, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00],
    ]
    Cds = [
        [0.002, 0.002, 0.002, 0.002, 0.002, 0.002],
        [0.002, 0.002, 0.002, 0.002, 0.002, 0.002],
    ]
    Cgs = [
        [0.001, 0.002, 0.000, 0.000, 0.000, 0.000, 0.100, 0.000],
        [0.001, 0.002, 0.000, 0.000, 0.000, 0.000, 0.000, 0.100],
    ]
    model = ChargeSensedDotArray(
        Cdd=Cdd,
        Cgd=Cgd,
        Cds=Cds,
        Cgs=Cgs,
        coulomb_peak_width=0.9,
        T=50.0,
        algorithm="default",
        implementation="jax",
        # noise_model=WhiteNoise(amplitude=1.0e-5) + TelegraphNoise(
        #     amplitude=5e-4, p01=5e-3, p10=5e-3
        # ),
        # latching_model=LatchingModel(n_dots=6, p_leads=0.95, p_inter=0.005),
    )
    from qua_dashboards.video_mode.inner_loop_actions.simulators import QarraySimulator

    sensor_plunger_bias_mv = [-5.0e-3, -5.0e-3]
    base_point = {
        "virtual_dot_1": -5.0e-3,
        "virtual_dot_2": -5.0e-3,
        "virtual_sensor_1": sensor_plunger_bias_mv[0],
        "virtual_sensor_2": sensor_plunger_bias_mv[1],
    }

    simulator = QarraySimulator(
        gate_set = machine.virtual_gate_sets["main_qpu"], 
        dc_set = machine.virtual_dc_sets["main_qpu"] if qdac_connect else None,
        model = model,
        sensor_gate_names = ("virtual_sensor_1", "virtual_sensor_2"), 
        base_point = base_point,
    )


    data_acquirer = SimulationDataAcquirer(
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
        simulator = simulator,
    )

    for axis in data_acquirer.sweep_axes["Voltage"]:
        if axis.name in ("virtual_dot_1", "virtual_dot_2"):
            axis.span = 0.03
            axis.points = 101

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

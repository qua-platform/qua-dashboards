"""
Example Script: Video Mode with OPX with Virtual Gating using LossDiVincenzoQuam

BaseQuamQD is a Quam infrastructure provided by Quantum Machines to aid in the calibration
of quantum dot devices. This script demonstrates how to use this infrastructure to create an
instance of video mode with the correct parameters, to perform live 2D scans on a quantum dot 
device. 

Quick How-to-Use:
1.  **Load the machine**
    * Video Mode is built on the gate set. Make sure that the gate set ID is correct
2.  **Adjust Scan Parameters**:
    * Select a `scan_mode` (e.g., `SwitchRasterScan`, `RasterScan`).
    * Set `result_type` in `OPXDataAcquirer` (e.g., "I", "Q", "amplitude", "phase").
3.  **Set a save_path to save Quam State JSON in the right directory**
4.  **Run the Script**: Execute this Python file.
5.  **Open Dashboard**: Navigate to `http://localhost:8050` (or the address shown
    in your terminal) in a web browser to view the live video mode dashboard.
"""

# %% Imports
from qua_dashboards.core import build_dashboard
from qua_dashboards.utils import setup_logging
from qua_dashboards.video_mode import (
    OPXDataAcquirer,
    scan_modes,
    VideoModeComponent,
)
from quam_builder.architecture.quantum_dots.qpu import LossDiVincenzoQuam
from qua_dashboards.virtual_gates import VirtualLayerEditor, ui_update
from qua_dashboards.voltage_control import VoltageControlComponent

def main():
    logger = setup_logging(__name__)

    ###########################################
    ############ LOAD QUAM MACHINE ############
    ###########################################

    dc_control = True
    machine = LossDiVincenzoQuam.load()
    gate_set_name = "main_qpu"
    # Instantiate the qmm once the virtual gate set is created
    qmm = machine.connect(skip_dacs = not dc_control)

    ################################################
    ############ INSTANTIATE VIDEO MODE ############
    ################################################

    scan_mode_dict = {
        "Switch_Raster_Scan": scan_modes.SwitchRasterScan(),
        "Raster_Scan": scan_modes.RasterScan(),
        "Spiral_Scan": scan_modes.SpiralScan(),
    }

    virtual_gating_component = VirtualLayerEditor(
        gateset=machine.virtual_gate_sets[gate_set_name], component_id="virtual-gates-ui", dc_set = machine.virtual_dc_sets[gate_set_name]
    )

    # External Voltage Control Component, if qdac_connect = True
    voltage_control_tab, voltage_control_component = None, None
    if dc_control:
        voltage_control_component = VoltageControlComponent(
            component_id="Voltage_Control",
            dc_set = machine.virtual_dc_sets[gate_set_name],
            update_interval_ms=1000,
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
            gate_set_name
        ],  # Replace with your GateSet instance
        x_axis_name=machine.virtual_gate_sets[gate_set_name].valid_channel_names[0], 
        y_axis_name=machine.virtual_gate_sets[gate_set_name].valid_channel_names[1], 
        scan_modes=scan_mode_dict,
        result_type="I",  # "I", "Q", "amplitude", or "phase"
        available_readout_pulses=[s.readout_resonator.operations["readout"] for s in machine.sensor_dots.values()],  # Input a list of pulses. The default only reads out from the first pulse, unless the second one is chosen in the UI.
        acquisition_interval_s=0.01,
        voltage_control_component=voltage_control_component,
        use_buffered_stream=False,
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

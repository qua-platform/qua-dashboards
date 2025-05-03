# File: my_experiment_scripts/config/config_video_mode_quam.py
import logging

# --- Hardware/Library Imports ---
# Assumes hardware_setup.py exists in the same directory
from . import hardware_setup as hw

# Imports for necessary classes from the library
try:
    from qua_dashboards.components.video_mode import VideoModeComponent
    from qua_dashboards.video_mode import SweepAxis, VoltageParameter, scan_modes

    # Note: BasicInnerLoopActionQuam needs to be adapted to take params/shared_objects
    # or be instantiated here based on machine structure. Let's assume it takes params.
    from qua_dashboards.video_mode.inner_loop_actions import BasicInnerLoopActionQuam

    # Imports for QUAM setup
    from quam.components import (
        BasicQuAM,
        SingleChannel,
        InOutSingleChannel,
        pulses,
        StickyChannelAddon,
    )
except ImportError as e:
    logging.error(
        f"Failed to import library components: {e}. Ensure qua_dashboards is installed."
    )
    # Raise error here as the config cannot function without the library
    raise e

logger = logging.getLogger(__name__)


# --- Configuration Function ---
def get_components():
    """
    Instantiates and returns components for Video Mode using OPXQuamDataAcquirer.
    """
    logger.info("Configuring components for QUAM Video Mode...")

    # --- Shared Objects Setup ---
    qmm = hw.get_qmm(host="192.168.8.4", cluster_name="Cluster_1")  # Get QMM instance

    # Define the QUAM machine structure (simplified)
    logger.debug("Defining QUAM machine...")
    machine = BasicQuAM()

    # Drive Channels
    machine.channels["ch1"] = SingleChannel(
        opx_output=("con1", 1),
        sticky=StickyChannelAddon(duration=1_000, digital=False),
        operations={
            "step": pulses.SquarePulse(amplitude=0.1, length=1000)
        },  # Used by BasicInnerLoopActionQuam ramp?
    )
    machine.channels["ch2"] = SingleChannel(
        opx_output=("con1", 2),
        sticky=StickyChannelAddon(duration=1_000, digital=False),
        operations={
            "step": pulses.SquarePulse(amplitude=0.1, length=1000)
        },  # Used by BasicInnerLoopActionQuam ramp?
    )

    # Readout Pulse and Channel (only one needed for the action)
    # Use a unique ID if reusing the name 'readout' is problematic
    readout_pulse_id = "vm_readout_pulse"
    readout_pulse_obj = pulses.SquareReadoutPulse(
        id=readout_pulse_id, length=1000, amplitude=0.1
    )
    machine.pulses[readout_pulse_id] = readout_pulse_obj  # Register pulse

    readout_channel_id = "measure_ch"  # Give the readout channel a clear name
    machine.channels[readout_channel_id] = InOutSingleChannel(
        opx_output=("con1", 3),  # Example OPX output
        opx_input=("con1", 1),  # Example OPX input
        intermediate_frequency=0,
        operations={readout_pulse_id: readout_pulse_obj},  # Associate the pulse object
    )
    logger.debug(f"QUAM Machine defined with channels: {list(machine.channels.keys())}")

    # --- Component Parameters ---
    logger.debug("Defining component parameters...")
    # Voltage Offsets
    x_offset = VoltageParameter(name="X Voltage Offset", initial_value=0.0)
    y_offset = VoltageParameter(name="Y Voltage Offset", initial_value=0.0)

    # Inner Loop Action Parameters (passed to VMC to instantiate the action)
    inner_loop_action_params = {
        # Pass channel *names* and pulse *id*
        "x_element_ch_name": "ch1",
        "y_element_ch_name": "ch2",
        "readout_pulse_id": readout_pulse_id,
        "readout_channel_name": readout_channel_id,  # Need channel name too
        "ramp_rate": 1_000,
        "use_dBm": True,
    }

    # Axis Parameters (passed to VMC to instantiate SweepAxis)
    axis_params = {
        "x_axis": {
            "name": "x",
            "span": 0.03,
            "points": 51,
            "offset_parameter": x_offset,
        },
        "y_axis": {
            "name": "y",
            "span": 0.03,
            "points": 51,
            "offset_parameter": y_offset,
        },
    }

    # Scan Mode (instantiate or pass params)
    scan_mode = scan_modes.SwitchRasterScan()

    # Data Acquirer Specific Parameters (passed to VMC)
    data_acquirer_specific_params = {
        "result_type": "I",
        "num_averages": 1,  # Example, adjust as needed
    }

    # Assemble parameters for VideoModeComponent constructor
    component_params = {
        "data_acquirer_type": "opx_quam",  # Flag for VMC
        "scan_mode": scan_mode,
        "inner_loop_action_type": BasicInnerLoopActionQuam,  # Pass the class type
        "inner_loop_action_params": inner_loop_action_params,
        "axis_params": axis_params,
        "data_acquirer_params": data_acquirer_specific_params,
        "update_interval": 1.0,  # From original script's VideoModeApp call
    }

    # Shared objects needed by VideoModeComponent
    shared_objects = {
        "qmm": qmm,
        "machine": machine,
        # VMC will likely call machine.generate_config() internally when needed
    }
    logger.debug("Configuration parameters defined.")

    # --- Instantiate Component ---
    logger.info("Instantiating VideoModeComponent...")
    vmc = VideoModeComponent(component_params, shared_objects)

    # Return component instances in a list
    return [vmc]


# --- Optional: Direct execution block ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)  # Set higher level for direct run
    logger.info(
        f"Running configuration script '{os.path.basename(__file__)}' directly..."
    )

    try:
        # Import the builder here, only needed when running directly
        from qua_dashboards.core.dashboard_builder import build_dashboard

        components_to_run = get_components()

        logger.info("Building dashboard...")
        app = build_dashboard(
            components=components_to_run, title="Video Mode (QUAM - Direct Run)"
        )

        logger.info("Starting Dash server (Direct Run)...")
        # use_reloader=False is often important here
        app.run(debug=True, port=8053, use_reloader=False)

    except ImportError as e:
        logger.error(
            f"ImportError in main block: {e}. Cannot launch directly. Is qua_dashboards installed?"
        )
    except Exception as e:
        logger.exception("Failed to configure or launch the dashboard directly.")

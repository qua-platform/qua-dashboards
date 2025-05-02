import os
import logging
from qua_dashboards.components.video_mode import VideoModeComponent, SweepAxis


def get_components():
    """Instantiates and returns components for Random Video Mode"""
    axis_params = {
        "x_axis": SweepAxis(name="X", span=0.1, points=51),
        "y_axis": SweepAxis(name="Y", span=0.1, points=101),
    }
    data_acquirer_specific_params = {
        "num_averages": 5,
        "acquire_time": 0.1,
    }
    component_params = {
        "data_acquirer_type": "random",
        "axis_params": axis_params,
        "data_acquirer_params": data_acquirer_specific_params,
        "update_interval": 0.1,
    }
    shared_objects = {}

    vmc = VideoModeComponent(component_params, shared_objects)
    return [vmc]


if __name__ == "__main__":
    # This block only executes when the script is run directly

    # Setup basic logging for direct execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__ + ".main")

    logger.info(
        f"Running configuration script '{os.path.basename(__file__)}' directly..."
    )

    try:
        # Import the builder here, only needed when running directly
        from qua_dashboards.core.dashboard_builder import build_dashboard

        # Get the components defined in this file
        components_to_run = get_components()

        logger.info("Building dashboard...")
        app = build_dashboard(
            components=components_to_run, title="Video Mode (Random - Direct Run)"
        )

        logger.info("Starting Dash server (Direct Run)...")
        # Use default port or specify one; debug=True is common for direct runs
        app.run(debug=True, port=8052, use_reloader=False)

    except ImportError as e:
        logger.error(
            f"ImportError in main block: {e}. Cannot launch directly. Is qua_dashboards installed?"
        )
    except Exception:
        logger.exception("Failed to configure or launch the dashboard directly.")

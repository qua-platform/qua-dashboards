import logging
import sys

# --- Imports from qua_dashboards ---
# Assumes qua_dashboards is installed and accessible in PYTHONPATH.
from qua_dashboards.core.dashboard_builder import build_dashboard
from qua_dashboards.video_mode import (
    SweepAxis,
    RandomDataAcquirer,
    VideoModeComponent,
)


def main():
    """
    Sets up and runs a Dash dashboard with VideoModeComponent using
    RandomDataAcquirer.
    """
    # 1. Configure Logging
    # Sets up basic logging to the console.
    # Adjust level (e.g., logging.INFO) and format as needed for debugging.
    logging.basicConfig(
        level=logging.DEBUG,  # Set to INFO for less verbose output
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    # Optionally, quiet down overly verbose third-party loggers.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("hpack").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info("Starting Video Mode application with RandomDataAcquirer.")

    # 2. Define SweepAxis objects
    # These objects define the properties of the X and Y axes for the scan.
    x_axis = SweepAxis(
        name="Horizontal Axis",
        label="X Coordinate",
        units="µm",  # Example unit
        span=10.0,  # Total span of the sweep (e.g., from -5 µm to +5 µm)
        points=61,  # Number of points along this axis
    )
    logger.debug(f"X-axis defined: {x_axis}")

    y_axis = SweepAxis(
        name="Vertical Axis",
        label="Y Coordinate",
        units="µm",  # Example unit
        span=8.0,  # Total span of the sweep
        points=51,  # Number of points along this axis
    )
    logger.debug(f"Y-axis defined: {y_axis}")

    # 3. Instantiate RandomDataAcquirer
    # This data acquirer generates random 2D data for demonstration.
    random_acquirer = RandomDataAcquirer(
        component_id="random-data-source-001",  # Unique ID for this acquirer
        x_axis=x_axis,
        y_axis=y_axis,
        acquire_time=0.03,  # Simulated time (s) per raw data frame
        num_software_averages=5,  # Number of raw frames to average
        acquisition_interval_s=0.5,  # Target interval (s) for new averaged frames
    )
    logger.info(f"RandomDataAcquirer instance created: {random_acquirer.component_id}")

    # 4. Instantiate VideoModeComponent
    # This is the main component for the video mode UI.
    # It uses the RandomDataAcquirer defined above.
    # Default tab controllers (LiveView and Annotation) will be used.
    video_mode_component = VideoModeComponent(
        component_id=VideoModeComponent.DEFAULT_COMPONENT_ID,  # Using the shared constant
        data_acquirer=random_acquirer,
        data_polling_interval_s=random_acquirer.acquisition_interval_s,  # How often the UI polls for new data updates
    )
    logger.info(
        f"VideoModeComponent instance created: {video_mode_component.component_id}"
    )

    # 5. Build and Run the Dashboard Application
    # The build_dashboard function takes a list of top-level components.
    logger.info("Building the dashboard...")
    app = build_dashboard(
        components=[video_mode_component],  # Only one top-level component here
        title="Video Mode Simulation (Random Data)",
    )

    logger.info("Dashboard built. Starting Dash server on http://localhost:8050")
    # debug=True enables Dash's debugging tools and auto-reloader.
    # use_reloader=False is often helpful during development to avoid
    # multiple initializations of components if the reloader is too aggressive.
    app.run(debug=True, host="0.0.0.0", port=8050, use_reloader=False)


if __name__ == "__main__":
    main()

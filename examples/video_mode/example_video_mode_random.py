"""
Example Script: Video Mode with RandomDataAcquirer (Simulation)

This script demonstrates how to use the VideoModeComponent with a RandomDataAcquirer.
This setup is ideal for simulating and testing video mode dashboards without needing
a live connection to an OPX or other hardware. It allows you to understand the
dashboard's functionality, test UI interactions, and develop custom components
in a controlled environment.

Core Components Used:
- SweepAxis: Defines the parameters for each axis in the 2D scan (name, label,
             units, span, number of points).
- RandomDataAcquirer: A data acquirer that generates random data for the 2D scan,
                      simulating a real data acquisition process.
- VideoModeComponent: The main Dash component that orchestrates the video mode
                      display, taking a data acquirer as input and rendering
                      the live plot and controls.
- build_dashboard: A utility function from qua_dashboards.core to construct
                   a Dash application layout with the provided components.

How to Run:
1. Ensure you have `qua-dashboards` and its dependencies installed.
   (e.g., `pip install qua-dashboards`)
2. Save this script as a Python file (e.g., `run_random_video_mode.py`).
3. Run the script from your terminal: `python run_random_video_mode.py`
4. Open your web browser and navigate to `http://127.0.0.1:8050/` (or the
   address shown in your terminal).

You should see a dashboard titled "Video Mode Simulation (Random Data)"
displaying a 2D plot that updates with new random data periodically. You will
also have controls to adjust the parameters of the X and Y axes (Span and Points)
and the RandomDataAcquirer (Software Averages, Simulated Acquire Time).
"""

from qua_dashboards.core import build_dashboard
from qua_dashboards.utils import setup_logging
from qua_dashboards.video_mode import (
    SweepAxis,
    RandomDataAcquirer,
    VideoModeComponent,
)


def get_video_mode_component() -> VideoModeComponent:
    """
    Creates and returns a VideoModeComponent instance configured with a RandomDataAcquirer.

    This function encapsulates the setup of the sweep axes and the data acquirer,
    making it easy to reuse this configuration or import it into other scripts.

    Returns:
        VideoModeComponent: The configured video mode component.
    """
    # Define the X-axis for the 2D scan.
    x_axis = SweepAxis(
        name="x",  # Internal identifier for the axis.
        label="X Coordinate",  # Display label in the UI.
        units="µm",  # Physical units of the axis.
        span=10.0,  # The total range of the X-axis sweep.
        points=61,  # The number of points (pixels) along the X-axis.
    )

    # Define the Y-axis for the 2D scan.
    y_axis = SweepAxis(
        name="y",  # Internal identifier for the axis.
        label="Y Coordinate",  # Display label in the UI.
        units="µm",  # Physical units of the axis.
        span=8.0,  # The total range of the Y-axis sweep.
        points=51,  # The number of points (pixels) along the Y-axis.
    )

    # Instantiate the RandomDataAcquirer.
    # This acquirer simulates data fetching by generating random 2D arrays.
    random_acquirer = RandomDataAcquirer(
        component_id="random-data-acquirer",  # Unique ID for Dash elements.
        x_axis=x_axis,
        y_axis=y_axis,
        acquire_time=0.03,  # Simulated delay (seconds) for acquiring one raw frame.
        num_software_averages=5,  # Number of raw frames to average for display.
        acquisition_interval_s=0.1,  # Target time (seconds) between acquiring raw frames.
    )

    # Instantiate the VideoModeComponent.
    # This is the main UI component that displays the live 2D plot and controls.
    video_mode_component = VideoModeComponent(
        component_id=VideoModeComponent.DEFAULT_COMPONENT_ID,  # Uses a default ID.
        data_acquirer=random_acquirer,  # The source of the data.
        # How often the frontend asks for new data (linked to acquirer's interval).
        data_polling_interval_s=random_acquirer.acquisition_interval_s,
    )
    return video_mode_component


def main() -> None:
    """
    Sets up logging, creates the VideoModeComponent, builds the dashboard,
    and runs the Dash application server.
    """
    # Configure logging for the application.
    logger = setup_logging(__name__)
    logger.info("Starting Video Mode application with RandomDataAcquirer (Simulation).")

    # Get the configured VideoModeComponent.
    video_mode_component = get_video_mode_component()
    logger.info(
        f"VideoModeComponent instance created: {video_mode_component.component_id}"
    )

    logger.info("Building the dashboard...")
    # Use build_dashboard to create the Dash app layout.
    app = build_dashboard(
        components=[video_mode_component],  # List of dashboard components to include.
        title="Video Mode Simulation (Random Data)",  # Browser window title.
    )

    logger.info("Dashboard built. Starting Dash server on http://localhost:8050")
    # Run the Dash server.
    app.run(
        debug=True,  # Enables helpful Dash debugging features.
        host="0.0.0.0",  # Makes the server accessible on your local network.
        port=8050,  # Sets the server port.
        use_reloader=False,  # Often recommended for stability with background threads.
    )


if __name__ == "__main__":
    # This ensures that main() is called only when the script is executed directly.
    main()

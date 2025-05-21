from qua_dashboards.core import build_dashboard
from qua_dashboards.utils import setup_logging
from qua_dashboards.video_mode import (
    SweepAxis,
    RandomDataAcquirer,
    VideoModeComponent,
)


def get_video_mode_component():
    """
    Creates and returns a VideoModeComponent instance with a RandomDataAcquirer.
    This can be imported and used externally without starting the dashboard.
    """
    # 2. Define SweepAxis objects
    x_axis = SweepAxis(
        name="x",
        label="X Coordinate",
        units="µm",
        span=10.0,
        points=61,
    )
    y_axis = SweepAxis(
        name="y",
        label="Y Coordinate",
        units="µm",
        span=8.0,
        points=51,
    )
    # 3. Instantiate RandomDataAcquirer
    random_acquirer = RandomDataAcquirer(
        component_id="random-data-acquirer",
        x_axis=x_axis,
        y_axis=y_axis,
        acquire_time=0.03,
        num_software_averages=5,
        acquisition_interval_s=0.5,
    )
    # 4. Instantiate VideoModeComponent
    video_mode_component = VideoModeComponent(
        component_id=VideoModeComponent.DEFAULT_COMPONENT_ID,
        data_acquirer=random_acquirer,
        data_polling_interval_s=random_acquirer.acquisition_interval_s,
    )
    return video_mode_component


def main():
    """
    Sets up and runs a Dash dashboard with VideoModeComponent using
    RandomDataAcquirer.
    """
    logger = setup_logging(__name__)
    logger.info("Starting Video Mode application with RandomDataAcquirer.")

    video_mode_component = get_video_mode_component()
    logger.info(
        f"VideoModeComponent instance created: {video_mode_component.component_id}"
    )

    logger.info("Building the dashboard...")
    app = build_dashboard(
        components=[video_mode_component],
        title="Video Mode Simulation (Random Data)",
    )

    logger.info("Dashboard built. Starting Dash server on http://localhost:8050")
    app.run(debug=True, host="0.0.0.0", port=8050, use_reloader=False)


if __name__ == "__main__":
    main()

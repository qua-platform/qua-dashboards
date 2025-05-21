# Guide: Understanding and Using the Random Video Mode Example

This guide explains the `example_video_mode_random.py` script, which demonstrates how to use the `VideoModeComponent` with a `RandomDataAcquirer`.
This setup is ideal for simulating and testing video mode dashboards without needing a live connection to an OPX.

## Core Components

The example primarily uses three components from the `qua_dashboards` library:

1.  `SweepAxis`: Defines the parameters for each axis in the 2D scan (e.g.
    , name, label, units, span, and number of points).
2.  `RandomDataAcquirer`: A data acquirer that generates random data for the 2D scan, simulating a real data acquisition process.
    It's useful for development and testing when an OPX is not available.
3.  `VideoModeComponent`: The main Dash component that orchestrates the video mode display, taking a data acquirer as input and rendering the live plot and controls.
4.  `build_dashboard`: A utility function to construct a Dash application layout with the provided components.

## Code Breakdown

Let's break down the `example_video_mode_random.py` script step-by-step.

### 1. Imports

The script begins by importing the necessary classes and functions:

```python
from qua_dashboards.core import build_dashboard
from qua_dashboards.utils import setup_logging
from qua_dashboards.video_mode import (
    SweepAxis,
    RandomDataAcquirer,
    VideoModeComponent,
)
```

* `build_dashboard`: Used to assemble the final Dash application.
* `setup_logging`: A utility to configure logging for the application.
* `SweepAxis`: To define the properties of the X and Y axes for our 2D scan.
* `RandomDataAcquirer`: The component that will generate simulated random data.
* `VideoModeComponent`: The core UI component for displaying the video mode.

### 2. `get_video_mode_component()` Function

This function encapsulates the creation and configuration of the `VideoModeComponent`.
This is useful if you want to import and use this component in another script without immediately running a dashboard.

```python
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
```

* **Defining Sweep Axes**:
    * Two `SweepAxis` objects, `x_axis` and `y_axis`, are created.
    * Each axis is configured with a `name` (internal identifier), `label` (for display), `units`, `span` (range of the sweep), and `points` (resolution).
* **Instantiating `RandomDataAcquirer`**:
    * A `RandomDataAcquirer` is initialized.
        It takes the defined `x_axis` and `y_axis` as inputs.
    * `component_id`: A unique identifier for this acquirer instance.
    * `acquire_time`: Simulates the time it takes to acquire one frame of data (0.03 seconds in this case).
    * `num_software_averages`: Specifies how many raw data snapshots are averaged to produce one displayed frame.
    * `acquisition_interval_s`: The target interval for acquiring a new raw snapshot.
* **Instantiating `VideoModeComponent`**:
    * The `VideoModeComponent` is created, using the `random_acquirer` to get its data.
    * `component_id`: A unique identifier for the video mode UI component.
        `VideoModeComponent.DEFAULT_COMPONENT_ID` provides a default value.
    * `data_polling_interval_s`: Sets how often the frontend should request new data from the backend.
        It's linked to the acquirer's `acquisition_interval_s`.

### 3. `main()` Function

This is the primary function that sets up and runs the Dash dashboard.

```python
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
```

* **Setup Logging**: `setup_logging(__name__)` initializes the logging system.
* **Get `VideoModeComponent`**: It calls `get_video_mode_component()` to get the pre-configured video mode UI.
* **Build Dashboard**:
    * `build_dashboard` creates the Dash application.
    * It takes a list of `components` (in this case, just our `video_mode_component`) and a `title` for the web page.
* **Run Application**:
    * `app.run(...)` starts the Dash server.
    * `debug=True` enables Dash's debug mode.
    * `host="0.0.0.0"` makes the server accessible from other devices on the network.
    * `port=8050` specifies the port number.
    * `use_reloader=False` is often recommended for stability, especially when running within certain environments or with background threads.

### 4. Script Execution

The standard Python entry point:

```python
if __name__ == "__main__":
    main()
```

This ensures that the `main()` function is called only when the script is executed directly.

## How to Run

1.  Ensure you have `qua-dashboards` and its dependencies installed.
2.  Save the code as a Python file (e.g., `run_random_video_mode.py`).
3.  Run the script from your terminal: `python run_random_video_mode.py`
4.  Open your web browser and navigate to `http://127.0.0.1:8050/` (or the address shown in your terminal).

You should see a dashboard titled "Video Mode Simulation (Random Data)" displaying a 2D plot that updates with new random data periodically.
You will also have controls to adjust the parameters of the X and Y axes (Span and Points) and the RandomDataAcquirer (Software Averages, Simulated Acquire Time).

This example provides a solid foundation for understanding how to integrate data acquisition (even simulated) with the `VideoModeComponent` to create interactive dashboards.
You can adapt the `RandomDataAcquirer` or replace it with a real data acquirer (like `OPXDataAcquirer`) for actual experiments.

import numpy as np
import logging

from qua_dashboards.voltage_control import VoltageControlComponent
from qua_dashboards.core.base_component import BaseComponent
from qua_dashboards.core.dashboard_builder import build_dashboard
from qua_dashboards.video_mode import VideoModeComponent, SweepAxis, RandomDataAcquirer

# --- Configure Logging ---
# Recommended to see component initialization messages
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
# Silence overly verbose libraries if needed
logging.getLogger("hpack.hpack").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
# You might want to set qua_dashboards logger level specifically
# logging.getLogger("qua_dashboards").setLevel(logging.INFO)


class SimpleVoltageSource:  # No explicit inheritance from VoltageParameterProtocol needed for runtime check
    def __init__(self, name: str, label: str, units: str, initial_value: float = 0.0):
        # Direct instance attributes
        self.name: str = name
        self.label: str = label
        self.units: str = units
        self._value: float = initial_value

    def get_latest(self) -> float:
        return self._value

    def set(self, value: float) -> None:
        self._value = value


# Example Usage:
def get_components_voltage_control():
    voltage_parameters = [
        SimpleVoltageSource("vg1", "Gate 1", "V", initial_value=0),
        SimpleVoltageSource("vg2", "Gate 2", "V", initial_value=np.random.randn()),
        SimpleVoltageSource(
            "vds", "Drain-Source", "V", initial_value=np.random.randn()
        ),
        SimpleVoltageSource("vgs", "Gate-Source", "V", initial_value=np.random.randn()),
    ]
    voltage_controller = VoltageControlComponent(
        component_id="v_ctrl",
        voltage_parameters=voltage_parameters,
    )
    return [voltage_controller]


def get_components_video_mode() -> list[BaseComponent]:
    """
    Function to instantiate and configure the dashboard components.
    This follows the pattern potentially used by a launcher script.
    """
    # 1. Configure Axes
    x_axis = SweepAxis(name="X", span=0.1, points=51, units="V")
    y_axis = SweepAxis(name="Y", span=0.1, points=101, units="V")

    # 2. Configure Data Acquirer
    # Ensure RandomDataAcquirer inherits correctly (e.g., from BaseUpdatableComponent)
    # and calls super().__init__(component_id=...) in its __init__.
    # Provide a unique component_id if its controls need namespacing.
    data_acquirer = RandomDataAcquirer(
        x_axis=x_axis,
        y_axis=y_axis,
        num_averages=5,
        acquire_time=0.1,
        component_id="random-acquirer-1",  # Explicit ID for the acquirer component part
    )

    # 3. Instantiate the Video Mode Component
    # Pass the configured data acquirer and give this component instance a unique ID.
    # NOTE: Ensure VideoModeComponent's __init__ calls super().__init__(component_id=...)
    video_mode_comp = VideoModeComponent(
        data_acquirer=data_acquirer,
        component_id="video-mode-main",  # Unique ID for this VideoMode instance
        update_interval_sec=1,
        include_update_button=True,
        layout_columns=9,
        save_path="./video_mode_component_output",  # Optional: specify different save path
    )

    # 4. Create a list of components for the dashboard
    # In this case, just the VideoModeComponent
    dashboard_components = [video_mode_comp]

    return dashboard_components


def get_components() -> list[BaseComponent]:
    """
    Combines the components from both video mode and voltage control examples.
    """
    return get_components_voltage_control() + get_components_video_mode()


def main():
    """
    Builds and runs the dashboard.
    """
    components_to_run = get_components()

    # Build the dashboard application using the builder function
    app = build_dashboard(
        components=components_to_run, title="Video Mode Dashboard (Component Example)"
    )

    # Run the Dash server
    # use_reloader=False is often safer, especially if components interact with hardware.
    # Change port if default 8050 is in use.
    print("Starting Video Mode dashboard...")
    print("Access at: http://127.0.0.1:8050/ (or your local IP)")
    app.run(debug=True, port=8050, use_reloader=False, host="0.0.0.0")


if __name__ == "__main__":
    main()

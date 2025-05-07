import numpy as np

from qua_dashboards.core.dashboard_builder import build_dashboard
from qua_dashboards.voltage_control import VoltageControlComponent


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
def get_components():
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

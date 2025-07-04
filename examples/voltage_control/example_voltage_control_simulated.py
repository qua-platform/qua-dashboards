import numpy as np
import logging

from qua_dashboards.core import build_dashboard
from qua_dashboards.voltage_control import VoltageControlComponent
from qua_dashboards.utils import setup_logging, BasicParameter


def define_gates_simple():
    """Defines gates using SimpleVoltageSource for demo purposes."""
    return [
        BasicParameter("vg1", "Gate 1", "V", initial_value=0, delay=0.2),
        BasicParameter("vg2", "Gate 2", "V", initial_value=np.random.randn(), delay=0.2),
        BasicParameter("vds", "Drain-Source", "V", initial_value=np.random.randn(), delay=0.2),
        BasicParameter("vgs", "Gate-Source", "V", initial_value=np.random.randn(), delay=0.2),
    ]


def get_voltage_control_component(callback=None):
    """Returns a VoltageControlComponent with demo gates."""
    voltage_parameters = define_gates_simple()
    voltage_controller = VoltageControlComponent(
        component_id="v_ctrl",
        voltage_parameters=voltage_parameters,
        callback_on_param_change = callback
    )
    return voltage_controller

def example_callback(parameter):
    print("Parameter", parameter.name, "was changed to value", parameter.get_latest())

def main():
    logger = setup_logging(__name__)
    logger.info("Starting Voltage Control dashboard (Simple demo mode)")
    component = get_voltage_control_component(callback=example_callback)
    app = build_dashboard(
        components=[component],
        title="Voltage Control Dashboard (Simple Demo)",
    )
    logger.info("Dashboard built. Access at: http://127.0.0.1:8050/")
    app.run(debug=True, port=8050, use_reloader=False, host="0.0.0.0")


if __name__ == "__main__":
    main()

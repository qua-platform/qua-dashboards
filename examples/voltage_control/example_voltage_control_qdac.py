# %% Imports
from qua_dashboards.core import build_dashboard
from qua_dashboards.voltage_control import VoltageControlComponent
from qua_dashboards.utils import setup_logging

# QCoDeS parameters
from qcodes_contrib_drivers.drivers.QDevil import QDAC2
from qcodes.parameters import DelegateParameter


def define_gates_qdac(qdac, channel_mapping):
    """Defines gates using QDAC and a channel mapping dict."""
    voltage_parameters = []
    for ch_id, ch_attrs in channel_mapping.items():
        channel_voltage_param = qdac.channel(ch_id).dc_constant_V
        name = ch_attrs.get("name", channel_voltage_param.name)
        label = ch_attrs.get("label", channel_voltage_param.label)
        gate = DelegateParameter(source=channel_voltage_param, name=name, label=label)
        voltage_parameters.append(gate)
    return voltage_parameters


def get_voltage_control_component_qdac(qdac, channel_mapping):
    """Returns a VoltageControlComponent for QDAC with the given channel mapping."""
    voltage_parameters = define_gates_qdac(qdac, channel_mapping)
    voltage_controller = VoltageControlComponent(
        component_id="v_ctrl",
        voltage_parameters=voltage_parameters,
    )
    return voltage_controller


def main():
    logger = setup_logging(__name__)
    logger.info("Starting Voltage Control dashboard (QDAC mode)")

    # QDAC setup
    qdac_addr = "192.168.8.36"
    qdac = QDAC2.QDac2(
        "QDAC", visalib="@py", address=f"TCPIP::{qdac_addr}::5025::SOCKET"
    )
    gate_channel_attributes = {
        2: {"name": "VP1", "label": "Plunger gate 1"},
        3: {"name": "VP2", "label": "Plunger gate 2"},
        4: {"name": "VB1", "label": "Bias gate 1"},
        5: {"name": "VB2", "label": "Bias gate 2"},
    }
    component = get_voltage_control_component_qdac(qdac, gate_channel_attributes)
    app = build_dashboard(
        components=[component],
        title="Voltage Control (QDAC)",
    )
    logger.info("Dashboard built. Access at: http://127.0.0.1:8050/")
    app.run(debug=True, port=8050, use_reloader=False, host="0.0.0.0")


if __name__ == "__main__":
    main()

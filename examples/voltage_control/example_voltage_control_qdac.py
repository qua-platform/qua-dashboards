# %% Imports
from qua_dashboards.core.dashboard_builder import build_dashboard
from qua_dashboards.voltage_control import VoltageControlComponent

# QCoDeS parameters
from qcodes_contrib_drivers.drivers.QDevil import QDAC2
from qcodes.parameters import DelegateParameter

# %% QDAC setup

# Load instrument driver
qdac_addr = "192.168.8.36"
qdac = QDAC2.QDac2("QDAC", visalib="@py", address=f"TCPIP::{qdac_addr}::5025::SOCKET")

# Define relevant channels and attributes (e.g. custom name, label, units)
gate_channel_attributes = {
    2: {"name": "VP1", "label": "Plunger gate 1"},
    3: {"name": "VP2", "label": "Plunger gate 2"},
    4: {"name": "VB1", "label": "Bias gate 1"},
    5: {"name": "VB2", "label": "Bias gate 2"},
}

# Initialize gates
voltage_parameters = []
for ch_id, ch_attrs in gate_channel_attributes.items():
    channel_voltage_param = qdac.channel(ch_id).dc_constant_V
    name = ch_attrs.pop("name", channel_voltage_param.name)
    gate = DelegateParameter(source=channel_voltage_param, name=name, **ch_attrs)
    voltage_parameters.append(gate)


# %% Build dashboard
def get_components():
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
    app = build_dashboard(components=components_to_run, title="Voltage Control")

    # Run the Dash server
    print("Starting Video Mode dashboard - Access at: http://127.0.0.1:8050/")
    app.run(debug=True, port=8050, use_reloader=False, host="0.0.0.0")


if __name__ == "__main__":
    main()

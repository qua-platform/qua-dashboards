# %% Imports
from qua_dashboards.core import build_dashboard
from typing import Dict
from quam.components import SingleChannel, BasicQuam
from qua_dashboards.virtual_gates import VirtualLayerEditor, ui_update
from qua_dashboards.utils import setup_logging
from quam_builder.architecture.quantum_dots.components.virtual_gate_set import VirtualGateSet


def setup_single_channel(name, output, controller = 'con1', fem = 5, operations: Dict = None):
    channel = SingleChannel(
        id = name, 
        opx_output = (controller, fem, output), 
        operations = operations
    )
    return channel

def example_quam(num_of_gates = 3):
    machine = BasicQuam()
    gatelist = [f"Plunger{i+1}" for i in range(num_of_gates)]
    channel_mapping = {}
    for chan in gatelist:
        machine.channels[f"ch{gatelist.index(chan)+1}"] = setup_single_channel(name = chan, output = gatelist.index(chan)+1)
        channel_mapping[chan] = machine.channels[f"ch{gatelist.index(chan)+1}"].get_reference()

    return machine, channel_mapping

def setup_gateset():
    machine, channel_mapping = example_quam(num_of_gates = 4)
    gateset = VirtualGateSet(id = 'Plungers', channels = channel_mapping)
    return machine, gateset

def main():
    logger = setup_logging(__name__)
    logger.info("Starting Virtual Gating UI")
    machine, gateset = setup_gateset()

    component = VirtualLayerEditor(gateset = gateset, component_id = 'Virtual Gates UI')
    app = build_dashboard(
        components=[component],
        title="Virtual Layer UI",
    )
    ui_update(app, component)
    logger.info("Dashboard built. Access at: http://127.0.0.1:8050/")
    app.run(debug=True, port=8050, use_reloader=False, host="127.0.0.1")

    
    

if __name__ == "__main__":
    main()

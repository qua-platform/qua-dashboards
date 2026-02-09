from quam.core import QuamRoot
from typing import List, Optional

__all__ = ["define_DC_params", "connect_to_qdac"]

def define_DC_params(machine: QuamRoot, gate_names: List[str]):
    """
    Defines gates using QDAC and a channel mapping dict. Provide a list of channel names existing in your Quam object instance.

    Currently assumes VoltageGate objects, using 'offset_parameter" attribute.
    """
    from qcodes.parameters import DelegateParameter

    voltage_parameters = []
    for ch_name in gate_names:
        ch = machine.physical_channels[ch_name]
        parameter = getattr(ch, "offset_parameter", None)
        if parameter is not None:
            voltage_parameters.append(
                DelegateParameter(
                    name=ch_name, label=ch_name, source=ch.offset_parameter
                )
            )
    return voltage_parameters

def connect_to_qdac(address): 
    from qcodes_contrib_drivers.drivers.QDevil import QDAC2
    qdac = QDAC2.QDac2('QDAC', visalib='@py', address=f'TCPIP::{address}::5025::SOCKET')
    return qdac

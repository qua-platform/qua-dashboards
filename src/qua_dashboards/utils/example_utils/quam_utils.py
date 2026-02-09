from quam.components import (
    pulses,
    StickyChannelAddon,
)
from quam.components.ports import (
    LFFEMAnalogOutputPort,
    LFFEMAnalogInputPort,
    OPXPlusAnalogOutputPort,
    OPXPlusAnalogInputPort,
)

from quam_builder.architecture.quantum_dots.components import (
    VoltageGate,
    ReadoutResonatorSingle,
    QdacSpec,
)

__all__ = ["setup_DC_channel", "setup_readout_channel"]

def setup_DC_channel(
    name: str, opx_output_port: int, qdac_port: int, con="con1", fem: int = None
):
    """
    Set up a DC Channel

    Args:
        name: The channel name in your Quam.
        opx_ouput_port: The integer output port of your OPX.
        qdac_port: Integer Qdac output port.
        con: QM cluster controller, defaults to "con1".
        fem: If using an OPX1000, add integer FEM number. Defaults to None for OPX+.
    """
    if fem is None:
        opx_output = OPXPlusAnalogOutputPort(
            controller_id=con,
            port_id=opx_output_port,
        )
    else:
        opx_output = LFFEMAnalogOutputPort(
            controller_id=con,
            fem_id=fem,
            port_id=opx_output_port,
            upsampling_mode="pulse",
        )

    channel = VoltageGate(
        id=name,
        opx_output=opx_output,  # Output for channel
        sticky=StickyChannelAddon(duration=1_000, digital=False),  # For DC offsets
        attenuation = 10,
    )
    qdac_spec = QdacSpec(qdac_output_port = qdac_port)
    channel.qdac_spec = qdac_spec
    if qdac_port is None:
        channel.offset_parameter = None
    return channel


def setup_readout_channel(
    name: str,
    readout_pulse: pulses.ReadoutPulse,
    opx_output_port: int,
    opx_input_port: int,
    IF: float,
    con="con1",
    fem: int = None,
):
    """
    Set up a Readout Channel

    Args:
        name: The channel name in your Quam.
        readout_pulse: The Readout Pulse object to be passed to the OPXDataAcquirer.
        opx_ouput_port: The integer output port of your OPX.
        opx_input_port: The integer input port of your OPX.
        IF: The intermediate frequency of your Readout channel.
        con: QM cluster controller, defaults to "con1".
        fem: If using an OPX1000, add integer FEM number. Defaults to None for OPX+. Assumed same FEM for output and input channels.
    """

    if fem is None:
        opx_output = OPXPlusAnalogOutputPort(
            controller_id=con,
            port_id=opx_output_port,
        )
        opx_input = OPXPlusAnalogInputPort(
            controller_id=con,
            port_id=opx_input_port,
        )
    else:
        opx_output = LFFEMAnalogOutputPort(
            controller_id=con,
            fem_id=fem,
            port_id=opx_output_port,
            upsampling_mode="mw",
        )
        opx_input = LFFEMAnalogInputPort(
            controller_id=con,
            fem_id=fem,
            port_id=opx_input_port,
        )

    channel = ReadoutResonatorSingle(
        id=name,
        opx_output=opx_output,  # Output for the readout pulse
        opx_input=opx_input,  # Input for acquiring the measurement signal
        intermediate_frequency=IF,  # Set IF for the readout channel
        operations={
            "readout": readout_pulse
        },  # Assign the readout pulse to this channel
        time_of_flight=28,
    )
    return channel

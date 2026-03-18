from .voltage_control_utils import define_DC_params, connect_to_qdac

__all__ = [
    "setup_DC_channel",
    "setup_readout_channel",
    "define_DC_params",
    "connect_to_qdac",
]


def setup_DC_channel(*args, **kwargs):
    from .quam_utils import setup_DC_channel as _setup_DC_channel
    return _setup_DC_channel(*args, **kwargs)


def setup_readout_channel(*args, **kwargs):
    from .quam_utils import setup_readout_channel as _setup_readout_channel
    return _setup_readout_channel(*args, **kwargs)
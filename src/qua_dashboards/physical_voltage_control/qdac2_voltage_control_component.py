"""
QDAC-II convenience wrapper for PhysicalVoltageControlComponent.
"""

from typing import Mapping

from .physical_voltage_control_component import PhysicalVoltageControlComponent


class QDAC2VoltageControlComponent(PhysicalVoltageControlComponent):
    def __init__(
        self,
        component_id: str,
        qdac,
        device_mapping: Mapping[str, int],
        dc_set=None,
        step_size: float = 1e-3,
        update_interval_ms: int = 1000,
    ):
        self.qdac = qdac
        self.device_mapping = dict(device_mapping)

        n_channels = qdac.n_channels()

        voltage_parameters = {}
        channel_labels = {}

        for name, channel_number in self.device_mapping.items():
            if not isinstance(channel_number, int):
                raise TypeError(
                    f"QDAC-II channel for '{name}' must be an integer, got {channel_number!r}."
                )

            if not 1 <= channel_number <= n_channels:
                raise ValueError(
                    f"Invalid QDAC-II channel {channel_number} for '{name}'. "
                    f"Valid range is 1-{n_channels}."
                )

            voltage_parameters[name] = qdac.channel(channel_number).dc_constant_V
            channel_labels[name] = f"ch{channel_number:02d}"

        super().__init__(
            component_id=component_id,
            voltage_parameters=voltage_parameters,
            dc_set=dc_set,
            step_size=step_size,
            update_interval_ms=update_interval_ms,
            channel_labels=channel_labels,
        )
"""
Shared manager for physical DC voltages.

The QCoDeS parameter cache is treated as the runtime source of truth.

If a physical gate also exists in the QUAM VirtualGateSet,
the corresponding `current_external_voltage` is synchronized
whenever that voltage is read or written through this manager.
"""

import logging
from typing import Mapping


logger = logging.getLogger(__name__)


class PhysicalVoltageManager:
    """
    Synchronize physical DC voltage state between QCoDeS and QUAM.

    Parameters
    ----------
    voltage_parameters:
        Mapping from logical physical gate names to QCoDeS-like parameters.

        Example
        -------
        {
            "P1_gate": qdac.ch01.dc_constant_V,
            "P2_gate": qdac.ch02.dc_constant_V,
        }

    gate_set:
        Optional VirtualGateSet.

        If a gate exists in gate_set.channels and has the attribute
        `current_external_voltage`, that value is kept synchronized
        with the QCoDeS parameter cache.
    """

    def __init__(
        self,
        voltage_parameters: Mapping[str, object],
        gate_set=None,
    ):
        self.voltage_parameters = dict(voltage_parameters)
        self.gate_set = gate_set

    def _sync_quam(
        self,
        gate_name: str,
        value: float,
    ) -> None:
        """
        Mirror a physical voltage into the QUAM runtime state.

        Gates which exist only on the external DAC are simply ignored.
        """

        if self.gate_set is None:
            return

        if gate_name not in self.gate_set.channels:
            return

        channel = self.gate_set.channels[gate_name]

        if hasattr(channel, "current_external_voltage"):
            channel.current_external_voltage = float(value)

    def get_latest(
        self,
        gate_name: str,
        *,
        fallback_to_hardware: bool = True,
    ) -> float:
        """
        Return the latest QCoDeS cached voltage.

        If the cache is empty and fallback_to_hardware=True,
        perform a real parameter get.

        The corresponding QUAM state is synchronized afterwards.
        """

        if gate_name not in self.voltage_parameters:
            raise KeyError(
                f"No physical voltage parameter configured "
                f"for '{gate_name}'."
            )

        param = self.voltage_parameters[gate_name]

        value = param.get_latest()

        if value is None:
            if not fallback_to_hardware:
                raise RuntimeError(
                    f"No cached voltage available for "
                    f"'{gate_name}'."
                )

            value = param.get()

        value = float(value)

        self._sync_quam(
            gate_name,
            value,
        )

        return value

    def get_from_hardware(
        self,
        gate_name: str,
    ) -> float:
        """
        Force a real hardware read.

        This updates the QCoDeS cache through param.get()
        and then mirrors the value into QUAM.
        """

        if gate_name not in self.voltage_parameters:
            raise KeyError(
                f"No physical voltage parameter configured "
                f"for '{gate_name}'."
            )

        param = self.voltage_parameters[gate_name]

        value = float(param.get())

        self._sync_quam(
            gate_name,
            value,
        )

        return value

    def set(
        self,
        gate_name: str,
        value: float,
    ) -> float:
        """
        Set a physical voltage through QCoDeS and synchronize QUAM.
        """

        if gate_name not in self.voltage_parameters:
            raise KeyError(
                f"No physical voltage parameter configured "
                f"for '{gate_name}'."
            )

        value = float(value)

        param = self.voltage_parameters[gate_name]

        # Write to QDAC / external hardware through QCoDeS.
        param.set(value)

        # QCoDeS cache is now updated.
        # Mirror the same value into QUAM.
        self._sync_quam(
            gate_name,
            value,
        )

        logger.info(
            "Set %s to %.6f V",
            gate_name,
            value,
        )

        return value

    def increment(
        self,
        gate_name: str,
        delta: float,
    ) -> float:
        """
        Increment a physical voltage relative to the latest QCoDeS value.
        """

        current = self.get_latest(gate_name)

        target = current + float(delta)

        return self.set(
            gate_name,
            target,
        )

    def sync_from_cache(self) -> dict[str, float]:
        """
        Synchronize all available QCoDeS cached values into QUAM.

        No real hardware query is performed for parameters whose
        cache is empty.
        """

        values = {}

        for gate_name in self.voltage_parameters:
            try:
                value = self.get_latest(
                    gate_name,
                    fallback_to_hardware=False,
                )

                values[gate_name] = value

            except Exception:
                # An empty cache is fine here.
                continue

        return values

    def sync_from_hardware(self) -> dict[str, float]:
        """
        Force a hardware query for all configured physical gates.

        This can be slow for remote instruments and should normally
        be used only explicitly, for example at session startup.
        """

        values = {}

        for gate_name in self.voltage_parameters:
            try:
                value = self.get_from_hardware(
                    gate_name
                )

                values[gate_name] = value

            except Exception:
                logger.exception(
                    "Failed to refresh %s from hardware",
                    gate_name,
                )

        return values
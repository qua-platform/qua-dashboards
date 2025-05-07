"""
Defines the protocol for voltage parameter objects that VoltageControlComponent interacts with.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class VoltageParameterProtocol(Protocol):
    """
    A protocol defining the expected interface for a voltage parameter.

    The VoltageControlComponent will interact with objects that adhere to this protocol.
    """

    @property
    def name(self) -> str:
        """A unique string identifier for this voltage parameter."""
        ...

    @property
    def label(self) -> str:
        """A user-friendly string label for display purposes."""
        ...

    @property
    def units(self) -> str:
        """A string indicating the physical units of the voltage (e.g., "V", "mV")."""
        ...

    def get_latest(self) -> float:
        """
        Returns the latest known or actual voltage value.

        This method is called periodically by the VoltageControlComponent for display.
        """
        ...

    def set(self, value: float) -> None:
        """
        Sets the voltage to the specified value.

        Args:
            value: The new voltage value to set.
        """
        ...

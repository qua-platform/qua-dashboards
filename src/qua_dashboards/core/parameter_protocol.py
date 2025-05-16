"""
Defines the protocol for parameter objects that can be used in control components.
"""

from typing import Protocol, runtime_checkable

__all__ = ["ParameterProtocol"]


@runtime_checkable
class ParameterProtocol(Protocol):
    """
    A protocol defining the expected interface for a generic parameter.
    """

    @property
    def name(self) -> str:
        """A unique string identifier for this parameter."""
        ...

    @property
    def label(self) -> str:
        """A user-friendly string label for display purposes."""
        ...

    @property
    def units(self) -> str:
        """
        A string indicating the physical units of the parameter (e.g.,
        "V", "mV", "Hz").
        """
        ...

    def get_latest(self) -> float:
        """
        Returns the latest known or actual value of the parameter.
        """
        ...

    def set(self, value: float) -> None:
        """
        Sets the parameter to the specified value.

        Args:
            value: The new value to set.
        """
        ...

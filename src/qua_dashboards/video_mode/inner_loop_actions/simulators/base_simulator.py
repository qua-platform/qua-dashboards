import numpy as np
from typing import Sequence

from abc import ABC, abstractmethod
from quam_builder.architecture.quantum_dots.components import GateSet

__all__ = ["BaseSimulator"]

class BaseSimulator(ABC): 
    """
    Base class for a simulator backend. 
    """
    def __init__(
        self, 
        gate_set: GateSet
    ): 
        self.gate_set = gate_set

    @abstractmethod
    def measure_data(        
        self,
        x_axis_name: str,
        y_axis_name: str,
        x_vals: Sequence[float],
        y_vals: Sequence[float],
        n_readout_channels: int,
        ): 
        """
        The measure function. By default takes the x and y sweeps, and returns a 2D map of simulated data.
        
        Args: 
            x_axis_name: The name of the x axis element
            y_axis_name: The name of the y axis element
            x_vals: The x-axis sweep values
            y_vals: The y-axis sweep values
            n_readout_channels: The number of readout channels 

        Returns: 
            I and Q 2D plots, both of dimensions (len(x_vals), len(y_vals))
        """

        pass

        



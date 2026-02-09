import numpy as np
from typing import Sequence, Any

from qua_dashboards.video_mode.inner_loop_actions.simulators.base_simulator import BaseSimulator

__all__ = ["RandomSimulator"]

class RandomSimulator(BaseSimulator): 
    """
    Random simulator which returns randomised data. 
    """

    def __init__(
        self, 
        **kwargs: Any,
    ): 
        super().__init__(**kwargs)
        
    def measure_data(        
        self,
        x_axis_name: str,
        y_axis_name: str,
        x_vals: Sequence[float],
        y_vals: Sequence[float],
        n_readout_channels: int,
        ):

        I, Q = np.random.rand(int(n_readout_channels), len(x_vals), len(y_vals)), np.random.rand(int(n_readout_channels), len(x_vals), len(y_vals))
        return I, Q
        
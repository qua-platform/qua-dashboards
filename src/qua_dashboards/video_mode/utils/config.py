from dataclasses import dataclass, field
import numpy as np

__all__ = ["SensorTuningConfig"]

@dataclass
class GeneralConfig:
    method: str = "align-linear"            # "align-linear" (faster, based on finding a linear compensation factor) or "autograd" (slower, based on fitting a sum of Gaussian basis functions)
    num_measurements: int = 6               # > 0, number of gate values
    N: int = 200                            # > 0, number of points in the sensor ramp (resolution)
    delta_central_point: float = 0.015      # > 0, min-max range of amplitude values are computed by central point -/+ delta_central_point

    def __post_init__(self):
        if self.method not in ("align-linear","autograd"):
            raise ValueError(f"Config: Invalid method: {self.method}")
        if self.num_measurements <= 0:
            raise ValueError(f"Config: Invalid num_measurements value: {self.num_measurements}")
        if self.N <= 0:
            raise ValueError(f"Config: Invalid N value: {self.N}")
        if self.delta_central_point <= 0:
            raise ValueError(f"Config: Invalid delta_central_point value: {self.delta_central_point}")


@dataclass
class AutogradConfig:  # "autograd" method
    min_w0: float = 0.0                 # min_w0 < max_w0
    max_w0: float = 0.7
    num_trials: int = 4                 # >0
    max_iterations: int = 1000000       # >0
    epsilon: float = 1.e-4              # tolerance, pick small value >0

    def __post_init__(self):
        if self.max_w0 < self.min_w0:
            raise ValueError(f"Config: max_w0 = {self.max_w0} < min_w0 = {self.min_w0}")
        if self.num_trials <= 0:
            raise ValueError(f"Config: Invalid num_trials value: {self.num_trials}")
        if self.max_iterations <= 0:
            raise ValueError(f"Config: Invalid max_iterations value: {self.max_iterations}")
        if self.epsilon <= 0:
            raise ValueError(f"Config: Invalid epsilon (tolerance) value: {self.epsilon}")   


@dataclass
class AlignLinearConfig:  # "align-linear" method
    w_min: float = -2.0                 # w_min < w_max
    w_max: float = 0.2
    sigma_gaussian: float = 1.0         # >0
    normalize_rows: bool = True
    use_mse: bool = False

    def __post_init__(self):
        if self.max_w0 < self.min_w0:
            raise ValueError(f"Config: max_w0 = {self.max_w0} < min_w0 = {self.min_w0}")
        if self.sigma_gaussian <= 0:
            raise ValueError(f"Config: Invalid sigma_gaussian value: {self.sigma_gaussian}")


@dataclass
class SensorTuningConfig:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    autograd: AutogradConfig = field(default_factory=AutogradConfig)
    align_linear: AlignLinearConfig = field(default_factory=AlignLinearConfig)

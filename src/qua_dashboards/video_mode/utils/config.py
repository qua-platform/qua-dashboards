from dataclasses import dataclass, field
import numpy as np

__all__ = ["TransformationMatrixConfig"]

@dataclass(frozen=True)
class GradientConfig:
    sigmaX_blur: float = 1.0    # >=0
    sigmaY_blur: float = 1.0    # >=0
    ksize_sobelX: int = 5       # >0, odd number
    ksize_sobelY: int = 5       # >0, odd number
    frac: float = 0.7           # between 0 and 1

    def __post_init__(self):
        if self.sigmaX_blur < 0:
            raise ValueError(f"Config: Invalid sigmaX_blur value: {self.sigmaX_blur}")
        if self.sigmaY_blur < 0:
            raise ValueError(f"Config: Invalid sigmaY_blur value: {self.sigmaY_blur}")
        if self.ksize_sobelX <= 0 or self.ksize_sobelX % 2 == 0:
            raise ValueError(f"Config: Invalid ksize_sobelX value: {self.ksize_sobelX}")
        if self.ksize_sobelY <= 0 or self.ksize_sobelY % 2 == 0:
            raise ValueError(f"Config: Invalid ksize_sobelY value: {self.ksize_sobelY}")
        if self.frac > 1 or self.frac < 0:
            raise ValueError(f"Config: Invalid frac value: {self.frac}")        

@dataclass(frozen=True)
class OptimizationConfig:
    max_iterations: int = 1000000      # >0
    epsilon: float = 1e-4              # tolerance: pick small value >0

    def __post_init__(self):
        if self.max_iterations <= 0:
            raise ValueError(f"Config: Invalid max_iterations value: {self.max_iterations}")
        if self.epsilon <= 0:
            raise ValueError(f"Config: Invalid epsilon (tolerance) value: {self.epsilon}")        
        

@dataclass(frozen=True)
class ModelConfig:
    scale: str = "per-dimension"                        # "overall" or "per-dimension"
    likelihood: str = "with-reg"                        # "with-reg" or "without-reg", i.e. with or without regularization term that ensures alignment of the normals with the correct axes
    w: tuple = (0.8, 0.1, 0.1)                          # Weights for the Gaussian components (mixing coefficients), sum w_i = 1
    init_params: tuple = (0.1, 0.0, 0.0, 0.1, 0.0)      # Initial guess for GMM: p1, p2 horizontal and vertical, tau = log_sigma = log(1.0) = 0.0
    reg_param: float = 1000.0                           # Regularization parameter >0

    def __post_init__(self):
        if self.scale not in ("overall","per-dimension"):
            raise ValueError(f"Config: Invalid scale parameter: {self.scale}")
        if self.likelihood not in ("with-reg","without-reg"):
            raise ValueError(f"Config: Invalid likelihood parameter: {self.likelihood}")
        if len(self.w) != 3 or not abs(sum(self.w) - 1.0) < 1e-8:
            raise ValueError(f"Config: Invalid weights w: {self.w}")
        if len(self.init_params) != 5:
            raise ValueError(f"Config: Invalid init_params: {self.init_params}")
        if self.reg_param <= 0:
            raise ValueError(f"Config: Invalid reg_param: {self.reg_param}")
        

@dataclass
class TransformationMatrixConfig:
    gradient: GradientConfig = field(default_factory=GradientConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
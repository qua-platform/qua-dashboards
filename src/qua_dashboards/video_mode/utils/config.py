from dataclasses import dataclass, field
import numpy as np

__all__ = ["TransformationMatrixConfig"]

@dataclass(frozen=True)
class GradientConfig:
    sigmaX_blur: float = 1.0
    sigmaY_blur: float = 1.0
    ksize_sobelX: int = 5
    ksize_sobelY: int = 5
    frac: float = 0.7

@dataclass(frozen=True)
class OptimizationConfig:
    max_iterations: int = 1000000
    epsilon: float = 1e-4

@dataclass(frozen=True)
class ModelConfig:
    scale: str = "per-dimension"                        # "overall" or "per-dimension"
    likelihood: str = "with-reg"                        # "with-reg" or "without-reg", i.e. with or without regularization term that ensures alignment of the normals with the correct axes
    w: tuple = (0.8, 0.1, 0.1)                          # Weights for the Gaussian components (mixing coefficients)
    init_params: tuple = (0.1, 0.0, 0.0, 0.1, 0.0)      # Initial guess for GMM: p1, p2 horizontal and vertical, tau = log_sigma = log(1.0) = 0.0

@dataclass
class TransformationMatrixConfig:
    gradient: GradientConfig = field(default_factory=GradientConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
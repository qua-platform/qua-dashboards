from qua_dashboards.core import BaseUpdatableComponent
from qua_dashboards.utils.qua_types import QuaVariableFloat


import numpy as np
from matplotlib import axes, figure, pyplot as plt
from matplotlib.ticker import MultipleLocator


from abc import ABC, abstractmethod
from typing import Generator, Sequence, Tuple


class ScanMode(BaseUpdatableComponent, ABC):
    """Abstract base class for scan modes, e.g. raster scan, spiral scan, etc.

    The scan mode is used to generate the scan pattern for the video mode.
    """

    def __init__(self, component_id: str = "scan-mode"):
        super().__init__(component_id=component_id)

    @abstractmethod
    def get_idxs(self, x_points: int, y_points: int) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def plot_scan(
        self, x_points: int, y_points: int
    ) -> Tuple[figure.Figure, axes.Axes]:
        idxs_x, idxs_y = self.get_idxs(x_points, y_points)

        u = np.diff(idxs_x)
        v = np.diff(idxs_y)
        pos_x = idxs_x[:-1] + u / 2
        pos_y = idxs_y[:-1] + v / 2
        norm = np.sqrt(u**2 + v**2)

        fig, ax = plt.subplots()
        ax.plot(idxs_x, idxs_y, marker="o")
        ax.quiver(pos_x, pos_y, u / norm, v / norm, angles="xy", zorder=5, pivot="mid")

        ax.xaxis.grid(True, which="both")
        ax.xaxis.set_minor_locator(MultipleLocator(abs(np.max(u))))
        ax.yaxis.grid(True, which="both")
        ax.yaxis.set_minor_locator(MultipleLocator(abs(np.max(v))))
        plt.show()

        return fig, ax

    @abstractmethod
    def scan(
        self, x_vals: Sequence[float], y_vals: Sequence[float]
    ) -> Generator[Tuple[QuaVariableFloat, QuaVariableFloat], None, None]:
        pass

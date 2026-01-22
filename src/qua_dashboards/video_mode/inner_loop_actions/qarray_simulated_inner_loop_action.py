from typing import List, Optional, Sequence

import numpy as np

from qua_dashboards.video_mode.inner_loop_actions.simulated_inner_loop_action import (
    SimulatedInnerLoopAction,
)

try:
    from qarray import ChargeSensedDotArray, WhiteNoise, TelegraphNoise, LatchingModel
except ImportError as exc:  # pragma: no cover - runtime dependency
    ChargeSensedDotArray = None  # type: ignore[assignment]
    WhiteNoise = None  # type: ignore[assignment]
    TelegraphNoise = None  # type: ignore[assignment]
    LatchingModel = None  # type: ignore[assignment]
    _QARRAY_IMPORT_ERROR = exc
else:
    _QARRAY_IMPORT_ERROR = None

__all__ = ["QarraySimulatedInnerLoopAction"]


class QarraySimulatedInnerLoopAction(SimulatedInnerLoopAction):
    """Simulated inner loop action backed by qarray charge sensor simulations."""

    def __init__(
        self,
        *args,
        model: Optional["ChargeSensedDotArray"] = None,
        x_gate_idx: int = 0,
        y_gate_idx: int = 1,
        sensor_gate_voltage_mv: float = 14.875,
        base_bias_mv: Optional[Sequence[float]] = None,
        n_charges: Optional[Sequence[int]] = None,
        optimize_sensor_gate: bool = True,
        sensor_sweep_min_mv: float = -10.0,
        sensor_sweep_max_mv: float = 10.0,
        sensor_sweep_points: int = 200,
        compensation_vector: Optional[Sequence[float]] = None,
        sweep_scale: float = 1e3,
        **kwargs,
    ):
        if _QARRAY_IMPORT_ERROR is not None:
            raise ImportError(
                "qarray is required for QarraySimulatedInnerLoopAction. "
                "Install it in your environment to use this simulation mode."
            ) from _QARRAY_IMPORT_ERROR

        super().__init__(*args, **kwargs)

        if model is None:
            model = self._build_default_model()

        self.model = model
        self.x_gate_idx = x_gate_idx
        self.y_gate_idx = y_gate_idx
        self.sensor_gate_voltage_mv = sensor_gate_voltage_mv
        self.base_bias_mv = self._init_base_bias(
            base_bias_mv=base_bias_mv,
            n_charges=n_charges,
            optimize_sensor_gate=optimize_sensor_gate,
            sensor_sweep_min_mv=sensor_sweep_min_mv,
            sensor_sweep_max_mv=sensor_sweep_max_mv,
            sensor_sweep_points=sensor_sweep_points,
        )
        self.compensation_vector = (
            np.asarray(compensation_vector, dtype=float)
            if compensation_vector is not None
            else np.zeros(7, dtype=float)
        )
        self.sweep_scale = float(sweep_scale)

    def _init_base_bias(
        self,
        *,
        base_bias_mv: Optional[Sequence[float]],
        n_charges: Optional[Sequence[int]],
        optimize_sensor_gate: bool,
        sensor_sweep_min_mv: float,
        sensor_sweep_max_mv: float,
        sensor_sweep_points: int,
    ) -> np.ndarray:
        if base_bias_mv is not None:
            return np.asarray(base_bias_mv, dtype=float)

        if n_charges is None:
            return np.zeros(7, dtype=float)

        base = np.asarray(self.model.optimal_Vg(n_charges=n_charges), dtype=float)
        if optimize_sensor_gate:
            z, _ = self.model.do1d_open(
                7,
                float(sensor_sweep_min_mv),
                float(sensor_sweep_max_mv),
                int(sensor_sweep_points),
            )
            sensor_sweep = np.linspace(
                sensor_sweep_min_mv, sensor_sweep_max_mv, int(sensor_sweep_points)
            )
            base[-1] = sensor_sweep[int(np.argmax(z))]
        return base

    @staticmethod
    def _build_default_model() -> "ChargeSensedDotArray":
        return ChargeSensedDotArray(
            Cdd=[
                [0.12, 0.08, 0.00, 0.00, 0.00, 0.00],
                [0.08, 0.13, 0.08, 0.00, 0.00, 0.00],
                [0.00, 0.08, 0.12, 0.08, 0.00, 0.00],
                [0.00, 0.00, 0.08, 0.12, 0.08, 0.00],
                [0.00, 0.00, 0.00, 0.08, 0.12, 0.08],
                [0.00, 0.00, 0.00, 0.00, 0.08, 0.11],
            ],
            Cgd=[
                [0.13, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                [0.00, 0.11, 0.00, 0.00, 0.00, 0.00, 0.00],
                [0.00, 0.00, 0.09, 0.00, 0.00, 0.00, 0.00],
                [0.00, 0.00, 0.00, 0.13, 0.00, 0.00, 0.00],
                [0.00, 0.00, 0.00, 0.00, 0.13, 0.00, 0.00],
                [0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00],
            ],
            Cds=[[0.002, 0.002, 0.002, 0.002, 0.002, 0.002]],
            Cgs=[[0.001, 0.002, 0.000, 0.000, 0.000, 0.000, 0.100]],
            coulomb_peak_width=0.9,
            T=50.0,
            algorithm="default",
            implementation="jax",
            noise_model=WhiteNoise(amplitude=1.0e-2) + TelegraphNoise(
                amplitude=5e-4, p01=5e-3, p10=5e-3
            ),
            latching_model=LatchingModel(n_dots=6, p_leads=0.95, p_inter=0.005),
        )

    def _sensor_scan(
        self,
        vp1_val: float,
        vp2_vals: Sequence[float],
    ) -> np.ndarray:
        compensation = self.compensation_vector
        base = np.array(self.base_bias_mv, dtype=float)
        base[self.x_gate_idx] += vp1_val
        base[6] += self.sensor_gate_voltage_mv + vp1_val * compensation[self.x_gate_idx]

        v_add_template = np.zeros(7, dtype=float)
        v_add_template[self.y_gate_idx] = 1.0
        v_add_template[6] = compensation[self.y_gate_idx]
        v_add = np.asarray(vp2_vals, dtype=float)[:, None] * v_add_template

        inputs = base + v_add
        z, _ = self.model.charge_sensor_open(-inputs)
        return np.asarray(z).squeeze()

    def __call__(self, x: Sequence[float], y: Sequence[float]) -> List[np.ndarray]:
        x_vals = np.asarray(x, dtype=float) * self.sweep_scale
        y_vals = np.asarray(y, dtype=float) * self.sweep_scale

        zs = [self._sensor_scan(vp1_val, y_vals) for vp1_val in x_vals]
        zs = np.asarray(zs)

        result: List[np.ndarray] = []
        for _ in self.selected_readout_channels:
            result.extend([zs, np.zeros_like(zs)])
        return result

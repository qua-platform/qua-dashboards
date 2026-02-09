import numpy as np
from typing import Sequence, Any, List, Dict, Optional

from qua_dashboards.video_mode.inner_loop_actions.simulators.base_simulator import BaseSimulator

__all__ = ["QarraySimulator"]

class QarraySimulator(BaseSimulator): 
    """
    Simulator using Qarray to simulate a QD structure.  
    """
    def __init__(
        self, *, 
        gate_set, 
        dc_set,
        model, 
        sensor_gate_names: Sequence[str],
        base_point: dict[str, float] | None = None, 
        voltage_scale: float = 1e3,
    ):
        super().__init__(gate_set=gate_set)
        self.model = model
        self.base_point = dict(base_point or {})

        self.sensor_gate_name = sensor_gate_names[0]
        self.sensor_gate_names = tuple(sensor_gate_names)
        self.sensor_index_by_name = {name: i for i, name in enumerate(self.sensor_gate_names)}
        self.dc_set = dc_set
        self.voltage_scale = voltage_scale

        self.qarray_gate_order = self.infer_qarray_gate_order()
        if dc_set is not None:
            _ = dc_set.all_current_voltages


    def infer_qarray_gate_order(self) -> List[str]: 
        gate_set = self.gate_set
        if hasattr(gate_set, "layers") and gate_set.layers: 
            tg = gate_set.layers[0].target_gates
            return tg
        else: 
            return sorted(gate_set.channels.keys())
    
    def point_to_vg(self, point: Dict[str, float]) -> np.ndarray: 
        physical_voltages = self.gate_set.resolve_voltages(point, allow_extra_entries=True)
        return np.array([float(physical_voltages.get(name, 0.0)) for name in self.qarray_gate_order], dtype = float)
    
    # def get_physical_grid(self, x_axis_name, y_axis_name, x_vals, y_vals): 
    #     """x_vals and y_vals are 1d sweeps"""
    #     base = dict(self.dc_set._current_levels)
    #     base.update(self.base_point)

    #     x_vals = np.asarray(x_vals, float)
    #     y_vals = np.asarray(y_vals, float)
    #     x_vals = x_vals - x_vals[len(x_vals)//2]
    #     y_vals = y_vals - y_vals[len(y_vals)//2]

    #     n_gates = len(self.qarray_gate_order)

    #     grids = [np.zeros((len(y_vals), len(x_vals)), float) for _ in range(n_gates)]
    #     for iy, dy in enumerate(y_vals):
    #         for ix, dx in enumerate(x_vals):
    #             point = dict(base)
    #             point[x_axis_name] = point.get(x_axis_name, 0.0) + dx
    #             point[y_axis_name] = point.get(y_axis_name, 0.0) + dy
    #             phys = self.gate_set.resolve_voltages(
    #                 point,
    #                 allow_extra_entries=True,
    #             )
    #             for gi, gate_name in enumerate(self.qarray_gate_order):
    #                 grids[gi][iy, ix] = float(phys.get(gate_name, 0.0))

    #     return grids

    def get_physical_grid(self, x_axis_name, y_axis_name, x_vals, y_vals):
        """Vectorized version - much faster for large grids."""
        # base = dict(
        #     self.dc_set._current_levels
        #     if self.dc_set is not None
        #     else zip(self.qarray_gate_order, np.zeros(len(self.qarray_gate_order)))
        # )
        if self.dc_set is not None:
            base = dict(self.dc_set._current_levels)
        else:
            base = dict(self.base_point)
        
        print(base)
        print(self.qarray_gate_order)
        
        x_vals = np.asarray(x_vals, float)
        y_vals = np.asarray(y_vals, float)
        x_vals = x_vals - x_vals[len(x_vals)//2]
        y_vals = y_vals - y_vals[len(y_vals)//2]
        
        base_phys = {name: base.get(name, 0.0) for name in self.qarray_gate_order}
        
        delta_x = self.gate_set.resolve_voltages(
            {**{k: 0.0 for k in base}, x_axis_name: 1.0}, 
            allow_extra_entries=True
        )
        delta_y = self.gate_set.resolve_voltages(
            {**{k: 0.0 for k in base}, y_axis_name: 1.0}, 
            allow_extra_entries=True
        )
        
        n_gates = len(self.qarray_gate_order)
        base_vec = np.array([base_phys.get(g, 0.0) for g in self.qarray_gate_order])
        dx_vec = np.array([delta_x.get(g, 0.0) for g in self.qarray_gate_order])
        dy_vec = np.array([delta_y.get(g, 0.0) for g in self.qarray_gate_order])
        X, Y = np.meshgrid(x_vals, y_vals)
        grids = (base_vec[None, None, :] 
                + X[:, :, None] * dx_vec[None, None, :] 
                + Y[:, :, None] * dy_vec[None, None, :])
        
        return [grids[:, :, i] for i in range(n_gates)]
    
    def grids_to_vg(self, grids): 
        grids = [np.asarray(g, float) for g in grids]
        vg = np.stack(grids, axis=-1)
        return vg


    # def measure_data(
    #     self,
    #     x_axis_name: str,
    #     y_axis_name: str,
    #     x_vals: Sequence[float],
    #     y_vals: Sequence[float],
    #     n_readout_channels: int,
    # ):
    #     """
    #     Run the qarray simulation and return I/Q data.
        
    #     Returns:
    #         I: array of shape (n_readout_channels, y_pts, x_pts)
    #         Q: array of shape (n_readout_channels, y_pts, x_pts)
    #     """
        
    #     grids = self.get_physical_grid(x_axis_name, y_axis_name, x_vals, y_vals)
    #     y_pts, x_pts = grids[0].shape
        
    #     slices = []
    #     for ix in range(x_pts):
    #         vg_slice = np.column_stack([g[:, ix] for g in grids])
    #         vg_slice_mv = vg_slice * self.voltage_scale 
            
    #         raw = self.model.charge_sensor_open(vg_slice_mv)
            
    #         if isinstance(raw, tuple):
    #             raw = raw[0]
    #         slices.append(np.asarray(raw).squeeze())
        

    #     zs = np.asarray(slices)

    #     z = zs.T
        
    #     if z.ndim == 2:
    #         z = z[np.newaxis, ...]
        
    #     n_sensors = z.shape[0]
        
    #     I_list = []
    #     Q_list = []
    #     for ch_idx in range(n_readout_channels):
    #         sensor_idx = ch_idx % n_sensors
    #         signal = z[sensor_idx]
            
    #         I_ch = signal
    #         Q_ch = np.zeros((y_pts, x_pts), dtype=signal.dtype)
            
    #         I_list.append(I_ch)
    #         Q_list.append(Q_ch)
        
    #     return np.stack(I_list, axis=0), np.stack(Q_list, axis=0)
    


    def measure_data(
        self,
        x_axis_name: str,
        y_axis_name: str,
        x_vals: Sequence[float],
        y_vals: Sequence[float],
        n_readout_channels: int,
    ):
        grids = self.get_physical_grid(x_axis_name, y_axis_name, x_vals, y_vals)
        y_pts, x_pts = grids[0].shape
        
        vg = np.stack(grids, axis=-1).reshape(-1, len(grids)) * self.voltage_scale
        
        raw = self.model.charge_sensor_open(vg)  # Model has NO noise
        if isinstance(raw, tuple):
            raw = raw[0]
        raw = np.asarray(raw)
        
        if raw.ndim == 1:
            raw = raw[:, np.newaxis]
        
        n_sensors = raw.shape[-1]
        z = raw.reshape(y_pts, x_pts, n_sensors)
        z = np.moveaxis(z, -1, 0)
        
        # Add noise HERE - no sequential artifacts
        z = z + np.random.normal(0, 0.01, z.shape)
        
        I = np.stack([z[ch_idx % n_sensors] for ch_idx in range(n_readout_channels)])
        Q = np.zeros_like(I)
        
        return I, Q
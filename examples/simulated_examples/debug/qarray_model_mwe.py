"""
Minimal Qarray model example: generate one frame and plot it.
"""

import numpy as np
import matplotlib.pyplot as plt

from qarray import ChargeSensedDotArray, WhiteNoise, TelegraphNoise, LatchingModel


if __name__ == "__main__":
    Cdd = [
        [0.12, 0.08, 0.00, 0.00, 0.00, 0.00],
        [0.08, 0.13, 0.08, 0.00, 0.00, 0.00],
        [0.00, 0.08, 0.12, 0.08, 0.00, 0.00],
        [0.00, 0.00, 0.08, 0.12, 0.08, 0.00],
        [0.00, 0.00, 0.00, 0.08, 0.12, 0.08],
        [0.00, 0.00, 0.00, 0.00, 0.08, 0.11],
    ]
    Cgd = [
        [0.13, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.11, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.09, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.13, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.13, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00],
    ]
    Cds = [
        [0.002, 0.002, 0.002, 0.002, 0.002, 0.002],
        [0.002, 0.002, 0.002, 0.002, 0.002, 0.002],
    ]
    Cgs = [
        [0.001, 0.002, 0.000, 0.000, 0.000, 0.000, 0.100, 0.000],
        [0.001, 0.002, 0.000, 0.000, 0.000, 0.000, 0.000, 0.100],
    ]
    model = ChargeSensedDotArray(
        Cdd=Cdd,
        Cgd=Cgd,
        Cds=Cds,
        Cgs=Cgs,
        coulomb_peak_width=0.9,
        T=50.0,
        algorithm="default",
        implementation="jax",
        noise_model=WhiteNoise(amplitude=1.0e-4) + TelegraphNoise(
            amplitude=5e-4, p01=5e-3, p10=5e-3
        ),
        latching_model=LatchingModel(n_dots=6, p_leads=0.95, p_inter=0.005),
    )

    n_charges = [1, 3, 0, 0, 0, 0, 5, 5]
    base_bias_mv = np.asarray(model.optimal_Vg(n_charges=n_charges), dtype=float)
    base_bias_mv += np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5., -5.], dtype=float)
    voltage_scale = 1e3
    x_vals = np.linspace(-0.05, 0.05, 101)
    y_vals = np.linspace(-0.05, 0.05, 101)

    x_vals = x_vals - x_vals[len(x_vals) // 2]
    y_vals = y_vals - y_vals[len(y_vals) // 2]

    z_rows = []
    n_open_rows = []
    for x in x_vals:
        vg_slice = np.zeros((len(y_vals), len(base_bias_mv)), dtype=float)
        vg_slice[:, 0] = x
        vg_slice[:, 1] = y_vals
        vg_slice_mv = vg_slice * voltage_scale + base_bias_mv

        raw = model.charge_sensor_open(vg_slice_mv)
        if isinstance(raw, tuple):
            raw, n_open = raw[0], raw[1]
            n_open_rows.append(np.asarray(n_open))
        else:
            n_open = None
        z_rows.append(np.asarray(raw))

    z = np.asarray(z_rows)
    if z.ndim == 2:
        z = z[None, ...]
    if z.ndim == 3:
        # (x, sensors, y) -> (sensors, y, x)
        z = np.swapaxes(z, 0, 2)

    n_gate = len(base_bias_mv)

    n_open_arr = np.asarray(n_open_rows) if n_open_rows else None
    if n_open_arr is not None and n_open_arr.ndim == 3:
        # (x, y, n_dot) -> (n_dot, y, x)
        n_open_arr = np.transpose(n_open_arr, (2, 1, 0))

    if z.shape[0] == 1:
        plt.figure(figsize=(6, 5))
        plt.imshow(
            z[0],
            origin="lower",
            aspect="auto",
            extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
        )
        plt.colorbar(label="Signal (arb.)")
        plt.xlabel("virtual_dot_1")
        plt.ylabel("virtual_dot_2")
        plt.title("Qarray model single frame (sensor 0)")
        plt.tight_layout()
        plt.show()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        for i in range(2):
            im = axes[i].imshow(
                z[i],
                origin="lower",
                aspect="auto",
                extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
            )
            axes[i].set_title(f"Sensor {i}")
            axes[i].set_xlabel("virtual_dot_1")
            if i == 0:
                axes[i].set_ylabel("virtual_dot_2")
            fig.colorbar(im, ax=axes[i], shrink=0.9)
        fig.suptitle("Qarray model single frame")
        fig.tight_layout()
        plt.show()

    for gate_idx in (n_gate - 2, n_gate - 1):
        vg_1d = np.tile(base_bias_mv, (len(x_vals), 1))
        vg_1d[:, gate_idx] += 0.1 * voltage_scale * x_vals
        signal_1d, _n_open_1d = model.charge_sensor_open(vg_1d)
        signal_1d = np.asarray(signal_1d)

        plt.figure(figsize=(6, 4))
        for sensor_idx in range(signal_1d.shape[1]):
            plt.plot(x_vals, signal_1d[:, sensor_idx], label=f"sensor {sensor_idx}")
        plt.title(f"1D sweep of gate {gate_idx}")
        plt.xlabel("sweep value")
        plt.ylabel("signal (arb.)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    if n_open_arr is not None:
        n_dot = n_open_arr.shape[0]
        ncols = min(3, n_dot)
        nrows = int(np.ceil(n_dot / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4 * ncols, 3.5 * nrows),
            sharex=True,
            sharey=True,
        )
        axes = np.atleast_1d(axes).ravel()
        for i in range(n_dot):
            im = axes[i].imshow(
                n_open_arr[i],
                origin="lower",
                aspect="auto",
                extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
            )
            axes[i].set_title(f"n_open dot {i}")
            axes[i].set_xlabel("virtual_dot_1")
            if i % ncols == 0:
                axes[i].set_ylabel("virtual_dot_2")
            fig.colorbar(im, ax=axes[i], shrink=0.85)
        for j in range(n_dot, len(axes)):
            axes[j].axis("off")
        fig.suptitle("Qarray model n_open")
        fig.tight_layout()
        plt.show()


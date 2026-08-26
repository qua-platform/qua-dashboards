"""
Minimal QarraySimulator example: generate one frame and plot it.
"""

# pylint: disable=unexpected-keyword-arg

from quam.components import StickyChannelAddon
from quam.components.ports import OPXPlusAnalogOutputPort
from quam_builder.architecture.quantum_dots.components import VoltageGate
from quam_builder.architecture.quantum_dots.qpu import BaseQuamQD

from qua_dashboards.video_mode.inner_loop_actions.simulators import QarraySimulator

from qarray import ChargeSensedDotArray, WhiteNoise, TelegraphNoise, LatchingModel

import numpy as np
import matplotlib.pyplot as plt


def setup_dc_channel(name: str, opx_output_port: int, con: str = "con1") -> VoltageGate:
    opx_output = OPXPlusAnalogOutputPort(controller_id=con, port_id=opx_output_port)
    return VoltageGate(
        id=name,
        opx_output=opx_output,
        sticky=StickyChannelAddon(duration=1_000, digital=False),
        attenuation=10,
    )


def main() -> None:
    machine = BaseQuamQD()

    p1 = setup_dc_channel("plunger_1", 1)
    p2 = setup_dc_channel("plunger_2", 2)
    p3 = setup_dc_channel("plunger_3", 3)
    p4 = setup_dc_channel("plunger_4", 4)
    p5 = setup_dc_channel("plunger_5", 5)
    p6 = setup_dc_channel("plunger_6", 6)
    p7 = setup_dc_channel("plunger_7", 7)
    p8 = setup_dc_channel("plunger_8", 8)

    machine.create_virtual_gate_set(
        gate_set_id="main_qpu",
        virtual_channel_mapping={
            "virtual_dot_1": p1,
            "virtual_dot_2": p2,
            "virtual_dot_3": p3,
            "virtual_dot_4": p4,
            "virtual_dot_5": p5,
            "virtual_dot_6": p6,
            "virtual_sensor_1": p7,
            "virtual_sensor_2": p8,
        },
        adjust_for_attenuation=False,
    )

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

    sensor_plunger_bias_mv = [-5.0, -5.0]
    base_point = {
        "virtual_sensor_1": sensor_plunger_bias_mv[0] / 1e3,
        "virtual_sensor_2": sensor_plunger_bias_mv[1] / 1e3,
    }

    simulator = QarraySimulator(
        gate_set=machine.virtual_gate_sets["main_qpu"],
        dc_set=None,
        model=model,
        sensor_gate_names=("virtual_sensor_1", "virtual_sensor_2"),
        # n_charges=[1, 3, 0, 0, 0, 0, 5, 5],
        voltage_scale=1e3,
        base_point=base_point,
    )

    x_vals = np.linspace(-0.05, 0.05, 101)
    y_vals = np.linspace(-0.05, 0.05, 101)
    x_vals = x_vals - x_vals[len(x_vals) // 2]
    y_vals = y_vals - y_vals[len(y_vals) // 2]

    I_data, _Q = simulator.measure_data(
        "virtual_dot_1",
        "virtual_dot_2",
        x_vals,
        y_vals,
        n_readout_channels=1,
    )
    
    fig, ax = plt.subplots(figsize=(6, 5), ncols=1)
    ax.imshow(I_data[0], origin="lower", aspect="auto", extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]])
    ax.set_xlabel("virtual_dot_1")
    ax.set_ylabel("virtual_dot_2")
    ax.set_title("QarraySimulator single frame")
    ax.tight_layout()
    # ax[1].imshow(I_data[1], origin="lower", aspect="auto", extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]])
    # ax[1].set_xlabel("virtual_dot_1")
    # ax[1].set_ylabel("virtual_dot_2")
    # ax[1].set_title("QarraySimulator single frame")
    ax.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

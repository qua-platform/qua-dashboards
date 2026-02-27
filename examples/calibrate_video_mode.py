"""
Video Mode Buffer Calibration
=============================

This script builds a 2D calibration map of ideal buffer_frames values across
a grid of (nx, ny) resolutions.  Run it once; the results are saved to JSON
and can be loaded in subsequent sessions so the OPXDataAcquirer automatically
picks the right buffer size for each resolution.

Usage
-----
1. Run this script with your normal machine/acquirer setup.
2. The calibration sweeps every (nx, ny) in the grid, recompiling the QUA
   program each time, then saves the results to CALIBRATION_PATH.
3. In your main video-mode script, load the calibration and pass it to the
   acquirer (see "Normal session" section at the bottom).
"""

"""
Example Script: Video Mode with OPX with Virtual Gating

This script demonstrates how to use the VideoModeComponent with an OPXDataAcquirer
to perform live 2D scans on a quantum device. It sets up a QUA program to sweep
two DC voltage channels and measure a readout signal, displaying the results in a
real-time dashboard.

Quick How-to-Use:
1.  **Configure Hardware**:
    * Update `qmm = QuantumMachinesManager(...)` with your OPX host and cluster_name.
    * Currently, a BasicQuam() machine is defined. 
        Modify the `machine = BasicQuam()` section to define the QUA elements.
        (channels, pulses) that match your experimental setup.
        Ensure `ch1`, `ch2` etc (for sweeping) and `ch1_readout` (or your measurement
        channel) are correctly defined.
2.  **Add/Adjust your virtual gates**: 
    * This script assumes a single layer of virtual gates. Adjust and add virtual gates 
        as necessary. 
    * Be sure to adjust the virtual gating matrices to suit your experimental needs. 
        This can be adjusted via the UI.
3.  **Configure the DC Control**
    * Adjust the VoltageControlComponent to suit your experiment
    * Ensure that each Quam channel is mapped to the correct output in the VoltageControlComponent. 
    * This example assumes the use of a QM QDAC, however is flexible for any voltage source.
4.  **Define your Readout Pulses**: 
    * First instantiate the relevant readout pulses
    * When creating your readout Quam channel, ensure that each readout pulse is correctly 
        and uniquely mapped to your readout elements. 
    * Pass the readout pulses to the data_acquirer instance as a list.
5.  **Adjust Scan Parameters**:
    * Select a `scan_mode` (e.g., `SwitchRasterScan`, `RasterScan`).
    * Set `result_type` in `OPXDataAcquirer` (e.g., "I", "Q", "amplitude", "phase").
6.  **Set a save_path to save Quam State JSON in the right directory**
7.  **Run the Script**: Execute this Python file.
8.  **Open Dashboard**: Navigate to `http://localhost:8050` (or the address shown
    in your terminal) in a web browser to view the live video mode dashboard.
"""

# %% Imports
from qm import QuantumMachinesManager
from quam.components import (
    BasicQuam,
    pulses,
)
import numpy as np
from qua_dashboards.utils import setup_logging
from qua_dashboards.video_mode import (
    OPXDataAcquirer,
    scan_modes,
)
from quam_builder.architecture.quantum_dots.components import VirtualGateSet
from quam_builder.architecture.quantum_dots.components import VirtualDCSet
from qua_dashboards.voltage_control import VoltageControlComponent

from qua_dashboards.utils import setup_DC_channel, setup_readout_channel, connect_to_qdac 

from qua_dashboards.video_mode.calibrations import Calibrations

def main():
    logger = setup_logging(__name__)

    ###########################################################################
    ############ CREATE QUAM MACHINE (SKIP IF EXISTING QUAM STATE) ############
    ###########################################################################

    # ### If Quam state exists, instead simply load it below: 
    # machine = BasicQuam.load()

    # Adjust the IP and cluster name here
    qm_ip = "172.16.33.101"
    cluster_name = "CS_2"

    qmm = QuantumMachinesManager(host=qm_ip, cluster_name=cluster_name)
    machine = BasicQuam()

    # Define your readout pulses here. Each pulse should be uniquely mapped to your readout elements. 
    readout_pulse_ch1 = pulses.SquareReadoutPulse(id="readout", length=30000, amplitude=0.1)
    readout_pulse_ch2 = pulses.SquareReadoutPulse(id="readout", length=30000, amplitude=0.1)

    # Choose the FEM. For OPX+, keep fem = None. 
    fem = None

    # Set up the readout channels
    machine.channels["ch1_readout"] = setup_readout_channel(name = "ch1_readout", readout_pulse=readout_pulse_ch1, opx_output_port = 6, opx_input_port = 1, IF = 150e6, fem = fem)
    machine.channels["ch2_readout"] = setup_readout_channel(name = "ch2_readout", readout_pulse=readout_pulse_ch2, opx_output_port = 6, opx_input_port = 1, IF = 250e6, fem = fem)

    channel_mapping = {
        "ch1": setup_DC_channel(name = "ch1", opx_output_port = 1, qdac_port = 1, fem = fem), 
        "ch2": setup_DC_channel(name = "ch2", opx_output_port = 2, qdac_port = 2, fem = fem), 
        "ch3": setup_DC_channel(name = "ch3", opx_output_port = 3, qdac_port = 3, fem = fem), 
        "ch1_readout_DC": setup_DC_channel(name = "ch1_readout_DC", opx_output_port = 4, qdac_port = 3, fem = fem), 
        "ch2_readout_DC": setup_DC_channel(name = "ch2_readout_DC", opx_output_port = 5, qdac_port = 3, fem = fem), 
    }
    for ch_name, ch in channel_mapping.items(): 
        machine.channels[ch_name] = ch

    # Adjust or add your virtual gates here. This example assumes a single virtual gating layer, add more if necessary. 
    logger.info("Creating VirtualGateSet")
    virtual_gate_set = VirtualGateSet(id = "Plungers", channels = {ch_name: ch.get_reference() for ch_name, ch in channel_mapping.items()})

    virtual_gate_set.add_layer(
        source_gates = ["V1", "V2"], # Pick the virtual gate names here 
        target_gates = ["ch1", "ch2"], # Must be a subset of gates in the gate_set
        matrix = [[1, 0.2], [0.2, 1]] # Any example matrix
    )

    scan_mode_dict = {
        "Switch_Raster_Scan": scan_modes.SwitchRasterScan(), 
        "Raster_Scan": scan_modes.RasterScan(), 
        "Spiral_Scan": scan_modes.SpiralScan(),
    }

    # Instantiate the OPXDataAcquirer.
    # This component handles the QUA program generation, execution, and data fetching.
    data_acquirer = OPXDataAcquirer(    
        qmm=qmm,
        machine=machine,
        gate_set=virtual_gate_set,  # Replace with your GateSet instance
        x_axis_name="ch1",  # Must appear in gate_set.valid_channel_names; Virtual gate names also valid
        y_axis_name="ch2",  # Must appear in gate_set.valid_channel_names; Virtual gate names also valid
        scan_modes=scan_mode_dict,
        result_type="I",  # "I", "Q", "amplitude", or "phase"
        available_readout_pulses=[readout_pulse_ch1, readout_pulse_ch2], # Input a list of pulses. The default only reads out from the first pulse, unless the second one is chosen in the UI. 
        acquisition_interval_s=0.05, 
        voltage_control_component=None, 
    )


    # ---------------------------------------------------------------------------
    # 1. Where to save the calibration
    # ---------------------------------------------------------------------------
    CALIBRATION_PATH = "video_mode_calibration.json"

    # ---------------------------------------------------------------------------
    # 2. Run calibration
    # ---------------------------------------------------------------------------
    # The grid runs nx and ny independently from 20 to 460 in steps of 40,
    # giving an 12×12 = 144-point grid.
    # Total time ≈ n_points × (warmup_s + n_fetch_samples × fetch_time + tiny plot overhead)
    # With warmup_s=3 and ~1 s fetch at 200×200 that is roughly 20–40 min.
    # Reduce the grid or increase the step for a faster (coarser) calibration.

    cal = Calibrations(
        acquirer=data_acquirer,
        nx_vals=np.arange(20, 461, 40),   # 12 values: 20, 60, ..., 460
        ny_vals=np.arange(20, 461, 40),
    )
    cal.run(
        n_fetch_samples=5,       # fresh-frame intervals per repeat
        n_plot_samples=10,       # figure_from_data() calls per repeat
        n_repeats=3,             # valid repeats per (nx, ny) point
        fetch_max_s=4.0,         # retry if fetch exceeds this (seconds/frame)
        max_retry_per_repeat=3,  # max retry budget per repeat target
    )
    cal.save(CALIBRATION_PATH)

    # ---------------------------------------------------------------------------
    # 3. Inspect results (optional)
    # ---------------------------------------------------------------------------
    cal = Calibrations.load(CALIBRATION_PATH)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, data, title in zip(
        axes,
        [cal.fetch_times * 1e3, cal.plot_times * 1e3, cal.ideal_buffers],
        ["Fetch time (ms)", "Plot time (ms)", "Ideal buffer_frames"],
    ):
        finite = np.asarray(data, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size > 0:
            vmin = np.percentile(finite, 5)
            vmax = np.percentile(finite, 95)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
                vmin = None
                vmax = None
        else:
            vmin = None
            vmax = None
        im = ax.imshow(
            data,
            origin="lower",
            extent=[cal.ny_vals[0], cal.ny_vals[-1], cal.nx_vals[0], cal.nx_vals[-1]],
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("ny")
        ax.set_ylabel("nx")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

    # # ---------------------------------------------------------------------------
    # # 4. Normal session — load calibration and pass to the acquirer
    # # ---------------------------------------------------------------------------
    # cal = Calibrations.load(CALIBRATION_PATH)

    # acquirer = OPXDataAcquirer(
    #     qmm=qmm,
    #     machine=machine,
    #     x_axis_name="P1",
    #     y_axis_name="P2",
    #     scan_modes={"Raster": RasterScan()}, # pyright: ignore[reportUndefinedVariable]
    #     gate_set=machine.gate_set,
    #     calibrations=cal.to_dict(),   # <-- pass the dict here
    #     buffer_frames=20,             # fallback if calibration lookup fails
    #     ...
    # )

# The acquirer will automatically interpolate the calibration map and use the
# ideal buffer_frames whenever it (re)compiles the QUA program.

if __name__ == "__main__":
    main()

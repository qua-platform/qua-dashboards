# Examples

Quick guide to every example shipped in this repo. Use the "When to use" blurb to pick the closest starting point; each bullet links to the script.

## Full dashboards (top-level)
- [`example_video_mode_full.py`](example_video_mode_full.py) – End-to-end OPX video-mode dashboard with optional QDAC DC control, virtual gates, and live plotting. When to use: you want the complete stack (VideoModeComponent + VirtualLayerEditor + VoltageControl) running on lab hardware.
- [`hybrid_opx_qdac_example.py`](hybrid_opx_qdac_example.py) – Hybrid sweeps where X is OPX-driven and Y is stepped on the QDAC (triggered by OPX). When to use: bias-tee setups or experiments needing long dwell on one axis while still streaming frames.
- [`qd_quam_example_video_mode.py`](qd_quam_example_video_mode.py) – Same video-mode flow built on `BaseQuamQD` with cross-compensation and detuning layers ready-made. When to use: you already structure devices with the QUAM quantum-dot template and want video mode + voltage control without rebuilding gatesets.

## Video mode (OPX-based)
- [`video_mode/example_video_mode_opx.py`](video_mode/example_video_mode_opx.py) – Minimal OPX-only video mode. You plug in your own `GateSet`; ideal smoke test for OPX connectivity and QUA generation.
- [`video_mode/example_video_mode_2_opx.py`](video_mode/example_video_mode_2_opx.py) – OPX video mode using `BasicQuam` + `VirtualGateSet` boilerplate. Leaner than the full example but already wires up sweep axes and readout pulses.

## Video mode (simulated backends)
- [`video_mode/simulated_examples/example_video_mode_random.py`](video_mode/simulated_examples/example_video_mode_random.py) – UI/data-path smoke test with `SimulationDataAcquirer` + `RandomSimulator`; runs fully offline, optional QDAC offsets if available.
- [`video_mode/simulated_examples/example_video_mode_simulation.py`](video_mode/simulated_examples/example_video_mode_simulation.py) – More realistic simulated scan using `QarraySimulator` + `BaseQuamQD` virtual gates. Great for developing scan logic and UI without booking hardware time.

## Data dashboard
- [`data_dashboard/example_data_dashboard.py`](data_dashboard/example_data_dashboard.py) – Demonstrates sending scalars, dicts, `xarray` arrays/datasets, and Matplotlib figures to the Data Dashboard client, including short real-time loops.

## Virtual gates
- [`virtual_gates/example_virtual_gates.py`](virtual_gates/example_virtual_gates.py) – Stand-alone `VirtualLayerEditor` UI over simulated channels; practice adding/editing virtual layers before touching hardware or embedding in a larger dashboard.

## Voltage control
- [`voltage_control/example_voltage_control_simulated.py`](voltage_control/example_voltage_control_simulated.py) – Simulated voltage sources wired to `VoltageControlComponent`; quick way to test-drive the GUI and polling cadence with no instruments.
- [`voltage_control/example_voltage_control_qdac.py`](voltage_control/example_voltage_control_qdac.py) – QCoDeS-backed QDAC control dashboard; template for real gate-bias tuning and integration with other components.

## Debug helpers
- [`debug/qarray_model_mwe.py`](debug/qarray_model_mwe.py) – Minimal `qarray` model to generate and plot a single frame (plus 1D slices) straight from the analytical charge-sensor model.
- [`debug/qarray_simulator_mwe.py`](debug/qarray_simulator_mwe.py) – Minimal `QarraySimulator` usage: builds a virtual gate set and grabs one simulated frame for plotting; handy for understanding simulator outputs.

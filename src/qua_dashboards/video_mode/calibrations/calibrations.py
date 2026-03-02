import json
import logging
import time
from math import ceil
from typing import Optional, TYPE_CHECKING

import numpy as np
import xarray as xr

from qua_dashboards.core import ModifiedFlags
from qua_dashboards.video_mode.utils.plot_utils import figure_from_data

if TYPE_CHECKING:
    from qua_dashboards.video_mode.data_acquirers.opx_data_acquirer import OPXDataAcquirer

logger = logging.getLogger(__name__)

__all__ = ["Calibrations"]


class Calibrations:
    """Calibrates ideal buffer_frames on the live acquisition pipeline.

    For each (nx, ny), calibration runs the normal start_acquisition ->
    get_latest_data path, measures per-frame acquire+fetch time from fresh frame
    arrivals, measures plotting time on the same frame payload, and sets
    ideal_buffer = ceil(fetch_time / plot_time).
    """

    def __init__(
        self,
        acquirer: Optional["OPXDataAcquirer"] = None,
        nx_vals: Optional[np.ndarray] = None,
        ny_vals: Optional[np.ndarray] = None,
    ):
        self.acquirer = acquirer
        self.nx_vals: np.ndarray = nx_vals if nx_vals is not None else np.arange(20, 461, 40)
        self.ny_vals: np.ndarray = ny_vals if ny_vals is not None else np.arange(20, 461, 40)

        # Populated after run()
        self.fetch_times: Optional[np.ndarray] = None   # shape (len(nx_vals), len(ny_vals)), seconds
        self.plot_times: Optional[np.ndarray] = None    # same, seconds per frame
        self.ideal_buffers: Optional[np.ndarray] = None  # same, integer counts

    # ------------------------------------------------------------------
    # Measurement helpers
    # ------------------------------------------------------------------

    def _current_fresh_seq(self) -> int:
        """Sequence that advances only when a truly fresh frame is consumed."""
        return int(getattr(self.acquirer, "_fresh_frame_seq", 0))

    def _wait_for_new_seq(
        self,
        prev_seq: int,
        timeout_s: float,
        poll_s: float = 0.005,
    ) -> tuple[int, Optional[xr.DataArray]]:
        """Block until get_latest_data() reports a strictly newer seq.

        Returns the new seq and the most recent data payload (which may be None
        for transient 'pending' states right after recompile).
        """
        t0 = time.perf_counter()
        while True:
            out = self.acquirer.get_latest_data()
            err = out.get("error")
            if err is not None:
                raise RuntimeError(f"Acquirer error during calibration: {err}")
            # Critical: _latest_seq increments every acquisition loop cycle, even
            # when perform_actual_acquisition() returns a repeated cached frame.
            # For calibration we need true fresh-frame cadence, so we use the
            # acquirer's dedicated fresh-frame sequence.
            seq = self._current_fresh_seq()
            data = out.get("data")
            if seq > prev_seq:
                return int(seq), data
            if time.perf_counter() - t0 > timeout_s:
                raise TimeoutError(
                    f"Timed out waiting for new frame seq (prev={prev_seq}, timeout={timeout_s}s)."
                )
            time.sleep(poll_s)

    def _measure_acquire_fetch_time(
        self,
        warmup_frames: int = 3,
        n_samples: int = 10,
        timeout_per_frame_s: float = 60.0,
    ) -> tuple[float, xr.DataArray]:
        """Measure per-frame acquisition+fetch time from the live threaded path.

        Uses start_acquisition() + get_latest_data() seq increments (same path as
        real Video Mode updates), not direct result_handles polling.

        Returns:
            (seconds per frame, latest frame data)
        """
        prev_seq = self._current_fresh_seq()
        latest_data: Optional[xr.DataArray] = None

        # Warm up: ignore initial compile/start transients.
        for _ in range(max(0, warmup_frames)):
            prev_seq, data = self._wait_for_new_seq(prev_seq, timeout_per_frame_s)
            if data is not None:
                latest_data = data

        t0 = time.perf_counter()
        for _ in range(max(1, n_samples)):
            prev_seq, data = self._wait_for_new_seq(prev_seq, timeout_per_frame_s)
            if data is not None:
                latest_data = data
        elapsed = time.perf_counter() - t0
        per_frame = elapsed / max(1, n_samples)

        # If the sampled frames were all transient/pending, wait briefly for any
        # valid frame payload to use in plot timing.
        if latest_data is None:
            t0_data = time.perf_counter()
            while time.perf_counter() - t0_data < timeout_per_frame_s:
                out = self.acquirer.get_latest_data()
                err = out.get("error")
                if err is not None:
                    raise RuntimeError(f"Acquirer error during calibration: {err}")
                data = out.get("data")
                if data is not None:
                    latest_data = data
                    break
                time.sleep(0.01)
            if latest_data is None:
                raise TimeoutError(
                    "No valid frame payload received for plot timing after seq advanced."
                )
        return per_frame, latest_data

    def _measure_plot_time(self, frame_data: xr.DataArray, n_samples: int = 10) -> float:
        """Average wall-clock time (seconds) to build a figure from real frame data."""
        times = []
        for _ in range(n_samples):
            t0 = time.perf_counter()
            figure_from_data(frame_data)
            times.append(time.perf_counter() - t0)
        return float(np.mean(times))

    # ------------------------------------------------------------------
    # Main calibration sweep
    # ------------------------------------------------------------------

    def run(
        self,
        warmup_frames: int = 3,
        n_fetch_samples: int = 10,
        n_plot_samples: int = 10,
        n_repeats: int = 3,
        fetch_max_s: float = 4.0,
        max_retry_per_repeat: int = 3,
    ) -> None:
        """Sweep the (nx, ny) resolution grid and fill fetch_times, plot_times,
        and ideal_buffers.

        Args:
            warmup_frames: Number of initial frames to ignore after start/recompile.
            n_fetch_samples: Number of per-frame intervals to average for
                acquisition+fetch timing.
            n_plot_samples: Number of figure_from_data() calls to average per point.
            n_repeats: Number of valid measurements to average per (nx, ny).
            fetch_max_s: Reject/retry measurements with fetch_time above this.
            max_retry_per_repeat: Max retries allowed for each repeat target.
        """
        if self.acquirer is None:
            raise ValueError("acquirer must be set before calling run().")

        nx_n = len(self.nx_vals)
        ny_n = len(self.ny_vals)
        self.fetch_times = np.zeros((nx_n, ny_n))
        self.plot_times = np.zeros((nx_n, ny_n))

        total = nx_n * ny_n
        done = 0

        for i, nx in enumerate(self.nx_vals):
            for j, ny in enumerate(self.ny_vals):
                done += 1
                print(f"[{done}/{total}] Calibrating nx={int(nx)}, ny={int(ny)} ...", flush=True)

                # Use the exact live measurement pipeline:
                # start_acquisition -> perform_actual_acquisition -> get_latest_data.
                # We only reconfigure between runs.
                self.acquirer.stop_acquisition()

                # Set resolution directly on the axis objects
                self.acquirer.x_axis.points = int(nx)
                self.acquirer.y_axis.points = int(ny)

                self.acquirer._clear_queue()
                self.acquirer._compiled_stream_vars = None
                self.acquirer._compilation_flags |= ModifiedFlags.PROGRAM_MODIFIED
                fetch_runs = []
                plot_runs = []
                repeat_idx = 0
                attempts = 0
                max_attempts = max(1, n_repeats) * max(1, max_retry_per_repeat)

                while repeat_idx < max(1, n_repeats) and attempts < max_attempts:
                    attempts += 1
                    self.acquirer.start_acquisition()
                    try:
                        fetch_t, latest_frame = self._measure_acquire_fetch_time(
                            warmup_frames=warmup_frames,
                            n_samples=n_fetch_samples,
                        )
                        plot_t = self._measure_plot_time(latest_frame, n_plot_samples)
                    except Exception as e:
                        logger.warning(
                            "Calibration retry (%s/%s) at nx=%s ny=%s due to exception: %s",
                            attempts,
                            max_attempts,
                            int(nx),
                            int(ny),
                            e,
                        )
                        continue
                    finally:
                        self.acquirer.stop_acquisition()

                    if (not np.isfinite(fetch_t)) or fetch_t <= 0 or fetch_t > fetch_max_s:
                        logger.warning(
                            "Calibration retry (%s/%s) at nx=%s ny=%s due to fetch outlier: %.3fs",
                            attempts,
                            max_attempts,
                            int(nx),
                            int(ny),
                            float(fetch_t),
                        )
                        continue
                    if (not np.isfinite(plot_t)) or plot_t <= 0:
                        logger.warning(
                            "Calibration retry (%s/%s) at nx=%s ny=%s due to invalid plot time: %.6fs",
                            attempts,
                            max_attempts,
                            int(nx),
                            int(ny),
                            float(plot_t),
                        )
                        continue

                    fetch_runs.append(fetch_t)
                    plot_runs.append(plot_t)
                    repeat_idx += 1

                if fetch_runs:
                    fetch_t = float(np.mean(fetch_runs))
                    plot_t = float(np.mean(plot_runs))
                else:
                    logger.warning(
                        "No valid calibration samples at nx=%s ny=%s after %s attempts; storing NaN.",
                        int(nx),
                        int(ny),
                        max_attempts,
                    )
                    fetch_t = np.nan
                    plot_t = np.nan

                self.fetch_times[i, j] = fetch_t
                self.plot_times[i, j] = plot_t
                print(
                    f"    fetch={fetch_t*1e3:.1f} ms  plot={plot_t*1e3:.1f} ms  "
                    f"ideal_buffer={ceil(fetch_t / max(plot_t, 1e-6)) if np.isfinite(fetch_t) and np.isfinite(plot_t) else 'NaN'}  "
                    f"(valid={len(fetch_runs)}/{max(1, n_repeats)})",
                    flush=True,
                )

        # Compute ideal buffer sizes
        ratio = self.fetch_times / np.maximum(self.plot_times, 1e-6)
        ratio = np.where(np.isfinite(ratio), ratio, np.nan)
        ratio = np.where(ratio > 0, ratio, np.nan)
        self.ideal_buffers = np.ceil(np.nan_to_num(ratio, nan=20.0)).astype(int)
        self.ideal_buffers = np.clip(self.ideal_buffers, 1, 200)

        print("Calibration complete.", flush=True)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_ideal_buffer(self, nx: int, ny: int) -> int:
        """Return the interpolated ideal buffer size for resolution (nx, ny).

        Falls back to 20 if calibration has not been run.
        """
        if self.ideal_buffers is None:
            logger.warning("No calibration data available. Returning default of 20.")
            return 20

        from scipy.interpolate import RegularGridInterpolator

        interp = RegularGridInterpolator(
            (self.nx_vals, self.ny_vals),
            self.ideal_buffers.astype(float),
            method="linear",
            bounds_error=False,
            fill_value=None,  # nearest-edge extrapolation at boundaries
        )
        val = float(interp([[float(nx), float(ny)]])[0])
        return max(1, int(round(val)))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise calibration data to a plain dict (JSON-compatible)."""
        if self.ideal_buffers is None:
            raise ValueError("No calibration data to serialise. Run calibration first.")
        return {
            "nx_vals": self.nx_vals.tolist(),
            "ny_vals": self.ny_vals.tolist(),
            "fetch_times": self.fetch_times.tolist(),
            "plot_times": self.plot_times.tolist(),
            "ideal_buffers": self.ideal_buffers.tolist(),
        }

    def save(self, path: str) -> None:
        """Save calibration data to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Calibration saved to {path}", flush=True)

    @classmethod
    def load(cls, path: str, acquirer: Optional["OPXDataAcquirer"] = None) -> "Calibrations":
        """Load calibration data from a JSON file.

        The returned object can be used for lookup (get_ideal_buffer / to_dict)
        without an acquirer.  Pass acquirer only if you intend to run further
        calibration sweeps.
        """
        with open(path) as f:
            data = json.load(f)
        cal = cls(acquirer=acquirer)
        cal.nx_vals = np.array(data["nx_vals"])
        cal.ny_vals = np.array(data["ny_vals"])
        cal.fetch_times = np.array(data["fetch_times"])
        cal.plot_times = np.array(data["plot_times"])
        cal.ideal_buffers = np.array(data["ideal_buffers"])
        return cal

import abc
import logging
import threading
import queue
import time
from typing import Any, Optional, List, Dict

from dash.dash import Dash
import numpy as np

from qua_dashboards.core import BaseUpdatableComponent, ModifiedFlags
from qua_dashboards.utils.dash_utils import create_input_field

logger = logging.getLogger(__name__)

__all__ = ["BaseDataAcquirer"]


class BaseDataAcquirer(BaseUpdatableComponent, abc.ABC):
    """
    Abstract base class for data acquirers with threaded acquisition and
    software averaging capabilities.

    This class provides the core mechanism for running data acquisition in a
    separate thread, collecting multiple raw data snapshots, and performing
    a rolling average on them. It makes minimal assumptions about the raw
    data structure itself, expecting numerical data or numpy arrays if default
    averaging is used. Subclasses implement the actual raw data acquisition.
    """

    def __init__(
        self,
        *,
        component_id: str,
        num_software_averages: int = 1,
        acquisition_interval_s: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the BaseDataAcquirer.

        Args:
            component_id: Unique ID for Dash component namespacing.
            num_software_averages: Number of raw data snapshots to average for
                each processed data point.
            acquisition_interval_s: Target interval for acquiring a new raw snapshot.
            **kwargs: Additional keyword arguments for BaseUpdatableComponent.
        """
        super().__init__(component_id=component_id, **kwargs)

        self.num_software_averages: int = max(1, num_software_averages)
        self.acquisition_interval_s: float = acquisition_interval_s

        self._data_history_raw: List[
            Any
        ] = []  # Stores raw snapshots from perform_actual_acquisition
        self._latest_processed_data: Optional[Any] = None  # Stores the averaged data

        self._data_lock: threading.Lock = threading.Lock()
        self._acquisition_thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()
        self._error_queue: queue.Queue = queue.Queue()
        self._acquisition_status: str = "stopped"  # stopped, running, error

    @abc.abstractmethod
    def perform_actual_acquisition(self) -> Any:
        """
        The core raw data acquisition method implemented by subclasses.

        Called repeatedly by the background thread. Should return a single
        raw data snapshot. This method can be blocking.

        Returns:
            The acquired raw data snapshot. Structure depends on the subclass.
            For averaging, this data should be numerically summable/dividable,
            or subclasses should override _perform_averaging.
        """
        pass

    def get_layout(self) -> List[Any]:
        """
        Get the layout for the data acquirer.
        """
        return []

    def register_callbacks(self, app: Dash, **kwargs: Any) -> None:
        pass

    def _perform_averaging(self, history: List[Any]) -> Optional[Any]:
        """
        Performs averaging on the collected raw data history.
        Default implementation assumes data are numpy arrays or numeric.
        Subclasses can override if a different averaging method is needed.
        """
        if not history:
            return None
        try:
            if all(isinstance(item, (int, float, complex)) for item in history) or all(
                isinstance(item, np.ndarray) for item in history
            ):
                return np.mean(history, axis=0)  # type: ignore
            else:
                logger.warning(
                    f"Default averaging method in {self.component_id} might not be "
                    f"suitable for data type {type(history[0])}. Returning last item."
                )
                return history[-1]  # Fallback for non-numeric types
        except Exception as e:
            logger.error(
                f"Error during default averaging in {self.component_id}: {e}. "
                "Returning last item."
            )
            return history[-1]

    def _acquisition_loop(self) -> None:
        logger.info(f"Acquisition loop started for {self.component_id}.")
        self._acquisition_status = "running"

        while not self._stop_event.is_set():
            loop_start_time = time.perf_counter()
            try:
                raw_snapshot = self.perform_actual_acquisition()

                with self._data_lock:
                    self._data_history_raw.append(raw_snapshot)
                    if len(self._data_history_raw) > self.num_software_averages:
                        self._data_history_raw.pop(0)

                    if self._data_history_raw:
                        self._latest_processed_data = self._perform_averaging(
                            self._data_history_raw
                        )

                try:  # Clear any previous error from the queue if successful this cycle
                    while not self._error_queue.empty():
                        self._error_queue.get_nowait()
                except queue.Empty:
                    pass

            except Exception as e:
                logger.error(
                    f"Error in acquisition cycle for {self.component_id}: {e}",
                    exc_info=True,
                )
                self._error_queue.put(e)
                self._acquisition_status = "error"
                time.sleep(max(self.acquisition_interval_s, 1.0))
                continue

            loop_duration = time.perf_counter() - loop_start_time
            wait_time = self.acquisition_interval_s - loop_duration
            if wait_time > 0:
                self._stop_event.wait(timeout=wait_time)

        self._acquisition_status = "stopped"
        logger.info(f"Acquisition loop stopped for {self.component_id}.")

    def start_acquisition(self) -> None:
        if self._acquisition_thread is not None and self._acquisition_thread.is_alive():
            logger.warning(f"Acquisition for {self.component_id} already running.")
            return

        self._stop_event.clear()
        with self._data_lock:  # Clear history before starting a new acquisition run
            self._data_history_raw.clear()
            self._latest_processed_data = None
        while not self._error_queue.empty():
            self._error_queue.get_nowait()

        self._acquisition_thread = threading.Thread(
            target=self._acquisition_loop,
            daemon=True,
            name=f"{self.component_id}_acq_thread",
        )
        self._acquisition_thread.start()
        logger.info(f"Data acquisition thread initiated for {self.component_id}.")

    def stop_acquisition(self) -> None:
        if self._acquisition_thread is None or not self._acquisition_thread.is_alive():
            logger.info(
                f"Acquisition for {self.component_id} not running or already stopped."
            )
            self._acquisition_status = "stopped"
            return

        logger.info(f"Stopping acquisition for {self.component_id}...")
        self._stop_event.set()
        self._acquisition_thread.join(timeout=max(self.acquisition_interval_s * 2, 2.0))

        if self._acquisition_thread.is_alive():
            logger.error(
                f"Acquisition thread for {self.component_id} did not stop gracefully."
            )
        else:
            logger.info(f"Acquisition thread for {self.component_id} stopped.")

        self._acquisition_thread = None
        self._acquisition_status = "stopped"

    def get_latest_data(self) -> Dict[str, Any]:
        """
        Retrieves the latest processed (e.g., averaged) data and status.
        Dimensional acquirers (like Base2DDataAcquirer) may override this to
        further transform the data (e.g., into an xr.DataArray).
        """
        error_from_queue: Optional[Exception] = None
        try:
            error_from_queue = self._error_queue.get_nowait()
            if error_from_queue:
                self._acquisition_status = "error"
        except queue.Empty:
            pass

        with self._data_lock:
            data_to_return = self._latest_processed_data

        return {
            "data": data_to_return,
            "error": error_from_queue,
            "status": self._acquisition_status,
        }

    def update_parameters(self, parameters: Dict[str, Dict[str, Any]]):

        if self.component_id in parameters:
            params = parameters[self.component_id]
            if (
                "acquisition_interval_s" in params
                and self.acquisition_interval_s != params["acquisition_interval_s"]
            ):
                new_interval = float(params["acquisition_interval_s"])
                if new_interval > 0:
                    self.acquisition_interval_s = new_interval
                else:
                    logger.warning(
                        f"Invalid acquisition_interval_s: {new_interval}. Not updated."
                    )

            if (
                "num_software_averages" in params
                and self.num_software_averages != params["num_software_averages"]
            ):
                new_averages = int(params["num_software_averages"])
                if new_averages >= 1:
                    self.num_software_averages = new_averages
                    with self._data_lock:  # Clear history as averaging window changed
                        self._data_history_raw.clear()
                        self._latest_processed_data = None
                else:
                    logger.warning(
                        f"Invalid num_software_averages: {new_averages}. Not updated."
                    )

        for component in self.get_components():
            if component is self:
                continue

            if component.component_id in parameters:
                component.update_parameters(parameters)



    def get_dash_components(self, include_subcomponents: bool = True) -> List[Any]:
        components = super().get_dash_components(include_subcomponents)
        # UI for num_software_averages
        components.append(
            create_input_field(
                id=self._get_id("num_software_averages"),
                label="Software Averages",
                value=self.num_software_averages,
                min=1,
                step=1,
                debounce=True,
            )
        )
        return components

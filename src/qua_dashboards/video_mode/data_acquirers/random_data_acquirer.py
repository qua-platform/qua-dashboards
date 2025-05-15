# qua_dashboards/video_mode/data_acquirers/random_data_acquirer.py
import logging
from time import sleep
from typing import Any, Dict, List

import numpy as np
from dash import html

from qua_dashboards.core import ModifiedFlags
from qua_dashboards.utils.dash_utils import create_input_field
from qua_dashboards.video_mode.data_acquirers.base_2d_data_acquirer import (
    Base2DDataAcquirer,
)
from qua_dashboards.video_mode.sweep_axis import SweepAxis

logger = logging.getLogger(__name__)

__all__ = ["RandomDataAcquirer"]


class RandomDataAcquirer(Base2DDataAcquirer):
    """Data acquirer that generates random 2D data for simulation purposes.

    Inherits from Base2DDataAcquirer and simulates a delay for data acquisition.
    The actual data generation is a simple random number matrix.
    """

    def __init__(
        self,
        *,
        x_axis: SweepAxis,
        y_axis: SweepAxis,
        component_id: str = "random-data-acquirer",
        acquire_time: float = 0.05,  # Simulate 50ms acquisition time per frame
        # Other parameters like num_software_averages, acquisition_interval_s
        # are passed via **kwargs to Base2DDataAcquirer and then to BaseDataAcquirer.
        **kwargs: Any,
    ) -> None:
        """Initializes the RandomDataAcquirer.

        Args:
            component_id: Unique ID for Dash elements.
            x_axis: The X sweep axis.
            y_axis: The Y sweep axis.
            acquire_time: Simulated time in seconds to 'acquire' one raw data frame.
            **kwargs: Additional arguments for Base2DDataAcquirer, including
                num_software_averages and acquisition_interval_s for
                BaseDataAcquirer.
        """
        self.acquire_time: float = acquire_time
        logger.debug(
            f"Initializing RandomDataAcquirer (ID: {component_id}) with "
            f"acquire_time: {self.acquire_time}s"
        )
        super().__init__(
            component_id=component_id, x_axis=x_axis, y_axis=y_axis, **kwargs
        )

    def perform_actual_acquisition(self) -> np.ndarray:
        """Simulates data acquisition by sleeping and returning random data.

        This method is called by the background thread in BaseDataAcquirer.

        Returns:
            A 2D numpy array of random float values between 0 and 1, with
            dimensions (y_axis.points, x_axis.points).
        """
        sleep(self.acquire_time)
        logger.debug(
            f"RandomDataAcquirer (ID: {self.component_id}): "
            f"Generating random data for {self.y_axis.points}x{self.x_axis.points}"
        )
        # Ensure y_axis.points and x_axis.points are positive integers
        if self.y_axis.points <= 0 or self.x_axis.points <= 0:
            logger.warning(
                f"RandomDataAcquirer (ID: {self.component_id}): Invalid points "
                f"({self.y_axis.points}x{self.x_axis.points}). Returning empty array."
            )
            return np.array([[]])  # Return a 2D empty array to avoid downstream errors

        results = np.random.rand(self.y_axis.points, self.x_axis.points)
        return results

    def get_dash_components(self, include_subcomponents: bool = True) -> List[html.Div]:
        """Returns Dash UI components for configuring RandomDataAcquirer.

        Extends the components from Base2DDataAcquirer (which includes
        BaseDataAcquirer's components) with a field for 'acquire_time'.

        Args:
            include_subcomponents: Whether to include UI components from
                parent classes. Defaults to True.

        Returns:
            A list of Dash html.Div components.
        """
        dash_components = super().get_dash_components(
            include_subcomponents=include_subcomponents
        )

        # UI for acquire_time
        dash_components.append(
            html.Div(
                create_input_field(
                    # Use self._get_id() for namespaced ID
                    id=self._get_id("acquire-time"),
                    label="Simulated Acquire Time",
                    value=self.acquire_time,
                    min=0.0,
                    max=10.0,  # Max 10 seconds simulation
                    step=0.01,
                    units="s",
                    debounce=True,  # Useful for number inputs
                    type="number",  # Explicitly set type
                )
            )
        )
        return dash_components

    def update_parameters(self, parameters: Dict[str, Dict[str, Any]]) -> ModifiedFlags:
        """Updates RandomDataAcquirer parameters based on UI input.

        Args:
            parameters: A dictionary where keys are component IDs and values are
                dictionaries of parameter names to their new values.

        Returns:
            ModifiedFlags indicating what aspects of the component were changed.
        """
        flags = super().update_parameters(parameters)

        # Check if parameters for this specific component_id are present
        if self.component_id in parameters:
            params = parameters[self.component_id]
            if "acquire-time" in params:
                new_acquire_time = float(params["acquire-time"])
                if self.acquire_time != new_acquire_time and new_acquire_time >= 0:
                    self.acquire_time = new_acquire_time
                    flags |= ModifiedFlags.PARAMETERS_MODIFIED
                    logger.debug(
                        f"RandomDataAcquirer (ID: {self.component_id}): "
                        f"Updated acquire_time to {self.acquire_time}s"
                    )
                elif new_acquire_time < 0:
                    logger.warning(
                        f"RandomDataAcquirer (ID: {self.component_id}): "
                        f"Invalid acquire_time ({new_acquire_time}s) received. Not updated."
                    )

        return flags

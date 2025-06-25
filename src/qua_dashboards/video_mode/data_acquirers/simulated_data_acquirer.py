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
from qdarts.experiment import Experiment

logger = logging.getLogger(__name__)

__all__ = ["SimulatedDataAcquirer"]


class SimulatedDataAcquirer(Base2DDataAcquirer):
    """Data acquirer that generates simulated 2D data using the simulation package QDarts.

    Inherits from Base2DDataAcquirer and simulates a delay for data acquisition.
    """

    def __init__(
        self,
        *,
        x_axis: SweepAxis,
        y_axis: SweepAxis,
        experiment: Experiment,
        args_rendering: dict[str, Any],
        conversion_factor_unit_to_volt: float,
        SNR: float = 20, # signal-to-noise ratio
        component_id: str = "simulated-data-acquirer",
        acquire_time: float = 0.05,  # Simulate 50ms acquisition time per frame
        # Other parameters like num_software_averages, acquisition_interval_s
        # are passed via **kwargs to Base2DDataAcquirer and then to BaseDataAcquirer.
        **kwargs: Any,
    ) -> None:
        """Initializes the SimulatedDataAcquirer.

        Args:
            component_id: Unique ID for Dash elements.
            x_axis: The X sweep axis.
            y_axis: The Y sweep axis.
            args_rendering: The arguments used for the rendering function of the QDarts simulator.
            SNR: Signal-to-noise ratio used for adding noise to the simulated image.
            acquire_time: Simulated time in seconds to 'acquire' one raw data frame.
            **kwargs: Additional arguments for Base2DDataAcquirer, including
                num_software_averages and acquisition_interval_s for
                BaseDataAcquirer.
        """
        self.acquire_time: float = acquire_time
        self.experiment: Experiment = experiment
        self.args_rendering: dict[str, Any] = args_rendering
        self.conversion_factor_unit_to_volt: float = conversion_factor_unit_to_volt
        self.SNR: float = SNR
        self.simulated_image: np.ndarray = None
        self._plot_parameters_changed: bool = True
        logger.debug(
            f"Initializing SimulatedDataAcquirer (ID: {component_id}) with "
            f"acquire_time: {self.acquire_time}s"
        )
        super().__init__(
            component_id=component_id, x_axis=x_axis, y_axis=y_axis, **kwargs
        )    


    def perform_actual_acquisition(self) -> np.ndarray:
        """Simulates data acquisition by sleeping and returning simulated data.

        This method is called by the background thread in BaseDataAcquirer.

        Returns:
            A 2D numpy array of simulated float values between 0 and 1, with
            dimensions (y_axis.points, x_axis.points).
        """
        # First acquisition and when parameters changed
        if self._plot_parameters_changed:

            logger.debug(
                f"SimulatedDataAcquirer (ID: {self.component_id}): "
                f"Generating simulated data for {self.y_axis.points}x{self.x_axis.points}"
            )

            # Ensure y_axis.points and x_axis.points are positive integers
            if self.y_axis.points <= 0 or self.x_axis.points <= 0:
                logger.warning(
                    f"SimulatedDataAcquirer (ID: {self.component_id}): Invalid points "
                    f"({self.y_axis.points}x{self.x_axis.points}). Returning empty array."
                )
                return np.array([[]])  # Return a 2D empty array to avoid downstream errors

            # Generate the simulated image          
            logger.info("Generating simulated data")
            _ , _ , _ , _ , sensor_signalexp , _ = self.experiment.generate_CSD(**self.args_rendering)
            # In case of several sensors, just show the image of the first one
            if sensor_signalexp.ndim == 3:
                self.simulated_image = sensor_signalexp[:,:,0]
            else:
                self.simulated_image = sensor_signalexp
            
            self._plot_parameters_changed = False

        # After first acquisition
        else:
            sleep(self.acquire_time)

        # Add uniformly distributed noise to the simulated image 
        logger.info(f"Add noise to the image (SNR = {self.SNR})")
        mean_signal = np.mean(self.simulated_image)
        factor = np.sqrt(12)*mean_signal/self.SNR
        logger.info(f"factor: {factor}")
        noise = np.random.rand(self.y_axis.points, self.x_axis.points)*factor

        return self.simulated_image + noise


    def get_dash_components(self, include_subcomponents: bool = True) -> List[html.Div]:
        """Returns Dash UI components for configuring SimulatedDataAcquirer.

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
        """Updates SimulatedDataAcquirer parameters based on UI input.

        Args:
            parameters: A dictionary where keys are component IDs and values are
                dictionaries of parameter names to their new values.

        Returns:
            ModifiedFlags indicating what aspects of the component were changed.
        """
        flags = super().update_parameters(parameters)
        logger.info(f"parameters: {parameters}")
        #parameters: {'simulated-data-acquirer': {'num_software_averages': 5, 'acquire-time': 1}, 'x-axis': {'span': 4.4, 'points': 50}, 'y-axis': {'span': 4.2, 'points': 50}}

        if flags & ModifiedFlags.PLOT_PARAMETERS_MODIFIED:
            logger.debug("PLOT parameters were modified!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            
            # Update the rendering arguments
            logger.info("Updating the rendering arguments")
            span_x = parameters["x-axis"]["span"]
            points_x = parameters["x-axis"]["points"]
            span_y = parameters["y-axis"]["span"]
            points_y = parameters["y-axis"]["points"]            
            self.args_rendering["x_voltages"] = np.linspace(-span_x/2., span_x/2., points_x)*self.conversion_factor_unit_to_volt
            self.args_rendering["y_voltages"] = np.linspace(-span_y/2., span_y/2., points_y)*self.conversion_factor_unit_to_volt
            
            # Clear history as plotting parameters change
            with self._data_lock:  
                self._data_history_raw.clear()
                self._latest_processed_data = None
            
            self._plot_parameters_changed = True

        # Check if parameters for this specific component_id are present
        if self.component_id in parameters:
            params = parameters[self.component_id]
            logger.debug(f"params: {params}")
            if "acquire-time" in params:
                new_acquire_time = float(params["acquire-time"])
                if self.acquire_time != new_acquire_time and new_acquire_time >= 0:
                    self.acquire_time = new_acquire_time
                    flags |= ModifiedFlags.PARAMETERS_MODIFIED
                    logger.debug(
                        f"SimulatedDataAcquirer (ID: {self.component_id}): "
                        f"Updated acquire_time to {self.acquire_time}s"
                    )
                elif new_acquire_time < 0:
                    logger.warning(
                        f"SimulatedDataAcquirer (ID: {self.component_id}): "
                        f"Invalid acquire_time ({new_acquire_time}s) received. Not updated."
                    )
            # TO DO: Include change of span and number of points --> simulate new image

        

        return flags

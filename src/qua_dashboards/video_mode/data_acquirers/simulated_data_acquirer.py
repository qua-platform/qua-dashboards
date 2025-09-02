import logging
import copy
from time import sleep
from typing import Any, Dict, List, Sequence

import numpy as np
from dash import html

from qua_dashboards.core import ModifiedFlags, ParameterProtocol
from qua_dashboards.utils.dash_utils import create_input_field
from qua_dashboards.video_mode.data_acquirers.base_2d_data_acquirer import (
    Base2DDataAcquirer,
)
from qua_dashboards.video_mode.sweep_axis import SweepAxis
from qdarts.experiment import Experiment

from qua_dashboards.voltage_control import VoltageControlComponent
from qua_dashboards.utils import BasicParameter

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
        sensor_number: int = 0,  # Default to first sensor, can only be different from 0 if there are multiple sensors
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
        self.voltage_parameters: Sequence[ParameterProtocol] = None
        self._last_voltage_parameters: Sequence[ParameterProtocol] = None  
        self.m = None
        self._first_acquisition: bool = True
        self._plot_parameters_changed: bool = False
        self._voltage_control_component: bool = False
        if self.experiment.sensor_config["sensor_dot_indices"] is not None and 0 <= sensor_number < len(self.experiment.sensor_config["sensor_dot_indices"]):
            self.sensor_number = sensor_number
        else:
            raise ValueError(f"Invalid sensor number {sensor_number}.")
        logger.debug(
            f"Initializing SimulatedDataAcquirer (ID: {component_id}) with "
            f"acquire_time: {self.acquire_time}s"
        )
        super().__init__(
            component_id=component_id, x_axis=x_axis, y_axis=y_axis, **kwargs
        )    


    def _initialize_m(self) -> None:
        """Initializes the m parameter and adds it to the rendering arguments."""   ### DISCUSS: transform m ??? Or is it already transformed ???
        self.m = copy.deepcopy(self.experiment.tunneling_sim.boundaries(self.args_rendering["state_hint_lower_left"]).point_inside)  # Deepcopy needed that self.m does not have the same reference as polytope.point_inside
        logger.debug(f"initial m: {self.m}")


    def get_voltage_control_component(self, labels = None) -> VoltageControlComponent:
        """ Creates and returns a voltage control component for the simulated data acquirer.
            The voltage values are set to the initial guess for m.

            Arguments:
                Labels: They are used to set the names of the voltage parameters. 
                        Default is None, which means that the default labels are used.
        """ 
        logger.debug(f"Creating VoltageControlComponent with labels: {labels}")
        
        self._voltage_control_component = True
        self._initialize_m()

        if labels is not None and len(labels) is not self.args_rendering["P"].shape[0]:
            raise ValueError(
                f"Number of labels ({len(labels)}) does not match number of voltage parameters "
                f"({self.args_rendering['P'].shape[0]})."
            )

        if labels is not None:
            voltage_parameters = [
                BasicParameter(f"vg{i+1}", label, "mV", initial_value=self.m[i] * 1./self.conversion_factor_unit_to_volt)
                for i, label in enumerate(labels)
            ]
        else:
            # If no labels are provided, use default labels
            logger.info(f"No labels are provided: default labels are used.")
            voltage_parameters = [
                BasicParameter(f"vg{i+1}", f"vg{i+1}", "mV", initial_value=self.m[i] * 1./self.conversion_factor_unit_to_volt)
                for i in range(self.args_rendering["P"].shape[0])
            ]

        self.voltage_parameters = voltage_parameters
        self._last_voltage_parameters = copy.deepcopy(voltage_parameters)  # Store the initial voltage parameters

        def callback(parameter, previous_value):
            logger.info(f"Parameter {parameter.name} was changed from value {previous_value} to value {parameter.get_latest()}")
            
            # Update m 
            self._last_voltage_parameters = copy.deepcopy(self.voltage_parameters)  # Update the last voltage parameters
            for i in range(len(self.voltage_parameters)):
                self.m[i] = self.voltage_parameters[i].get_latest() * self.conversion_factor_unit_to_volt
            logger.info(f"Voltage parameters changed, m updated: {self.m}")
            # Clear the data history
            with self._data_lock:  
                self._data_history_raw.clear()
                self._latest_processed_data = None  
            
            # Set the flag that the plot parameters were changed
            self._plot_parameters_changed = True  # This will trigger a new simulated image in perform_actual_acquisition()

        # Get the VoltageControlComponent
        voltage_controller = VoltageControlComponent(
            component_id="voltage_control",
            voltage_parameters=self.voltage_parameters,
            callback_on_param_change=callback,  # Callback function to handle parameter changes
        )

        return voltage_controller


    def perform_actual_acquisition(self) -> np.ndarray:
        """Simulates data acquisition by sleeping and returning simulated data.

        This method is called by the background thread in BaseDataAcquirer.

        Returns:
            A 2D numpy array of simulated float values between 0 and 1, with
            dimensions (y_axis.points, x_axis.points).
        """
        
        def generate_simulated_image() -> np.ndarray:
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

            # Generate the simulated image          ### DISCUSS: Coordinates transformation here ??? Especially also in sensor_signalexp ???
            logger.info("Generating simulated data")
            state = self.experiment.tunneling_sim.poly_sim.find_state_of_voltage(v = self.m, 
                                                                                 state_hint = self.args_rendering["state_hint_lower_left"])
            sliced_sim = self.experiment.tunneling_sim.slice(P = self.args_rendering["P"], m = self.m)
            sensor_signalexp = sliced_sim.sensor_scan_2D(P = np.eye(2),
                                                         m = np.zeros(2),
                                                         minV = self.args_rendering["minV"],
                                                         maxV = self.args_rendering["maxV"],
                                                         resolution = self.args_rendering["resolution"],
                                                         state_hint_lower_left = state)

            # In case of several sensors, pick the sensor with the given sensor_number (default is 0)
            if sensor_signalexp.ndim == 3:
                sensor_values = sensor_signalexp[:,:,self.sensor_number].T
            else:
                sensor_values = sensor_signalexp.T

            self.simulated_image = sensor_values

        # First acquisition  
        if self._first_acquisition:
            logger.info("First acquisition, generating simulated image")
            self._first_acquisition = False
            if not self._voltage_control_component:
                self._initialize_m()
            logger.info(f"First acquisition, initial rendering arguments: {self.args_rendering}")
            generate_simulated_image()

        # Plot parameters changed (points, span, voltage parameters, etc.)
        if self._plot_parameters_changed:
            self._plot_parameters_changed = False # Do this here to PREVENT race conditions! It takes long to simulate the new image, and every change of plotting parameters would lead to race conditions.
            generate_simulated_image()
        
        else:
            sleep(self.acquire_time)  # Simulate acquisition time

        # Add uniformly distributed noise to the simulated image 
        logger.info(f"Add noise to the image (SNR = {self.SNR})")
        mean_signal = np.mean(self.simulated_image)
        factor = np.sqrt(12)*mean_signal/self.SNR
        noise = np.random.rand(self.y_axis.points, self.x_axis.points)*factor

        # Check if shape of axes still match the shape of the image
        if self.simulated_image.shape[0] != self.y_axis.points or self.simulated_image.shape[1] != self.x_axis.points:  # race conditions
            logger.warning("Axes resolution changed during rendering !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logger.info(f"signal shape: {self.simulated_image.shape}")
            logger.info(f"(y_points x x_points): ({self.y_axis.points}x{self.x_axis.points}). Returning empty array.")
            # logger.info(f"y_points: {self.y_axis.points}")
            # logger.info(f"x_points: {self.x_axis.points}")
            return np.empty((self.y_axis.points, self.x_axis.points))  # Return a 2D empty array with the correct shape to avoid downstream errors
        else:
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
    
    def ramp(self, v_start, v_end, N):
        """
        Generates a linear ramp from v_start to v_end with the specified resolution.
        
        Parameters:
        v_start (np.array): Starting voltage vector.
        v_end (np.array): Ending voltage vector.
        state_hint (list): State hint for the tunneling simulation.
        N (int): Number of points in the ramp (resolution).
        
        Returns:
        1D numpy array of voltage vectors along the ramp.
        """
        state_hint = self.args_rendering["state_hint_lower_left"]
        sensor_signal = self.experiment.tunneling_sim.sensor_scan(
                                                            v_start = v_start,
                                                            v_end = v_end,
                                                            resolution = N,
                                                            v_start_state_hint = state_hint,
                                                            )
        logger.info(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!! Ramp from {v_start} to {v_end} with {N} points generated.")
        #logger.info(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!! Sensor signal: {sensor_signal[:, 0]}")
        return sensor_signal[:, 0]  # Return the sensor signal as a 1D numpy array    ### DISCUSS: Transform coordinates ???

    def update_parameters(self, parameters: Dict[str, Dict[str, Any]]) -> ModifiedFlags:
        """Updates SimulatedDataAcquirer parameters based on UI input.

        Args:
            parameters: A dictionary where keys are component IDs and values are
                dictionaries of parameter names to their new values.

        Returns:
            ModifiedFlags indicating what aspects of the component were changed.
        """
        flags = super().update_parameters(parameters)

        if flags & ModifiedFlags.PLOT_PARAMETERS_MODIFIED:
            # Update the rendering arguments
            logger.info("Updating the rendering arguments")
            span_x = parameters["x-axis"]["span"]
            points_x = parameters["x-axis"]["points"]
            span_y = parameters["y-axis"]["span"]
            points_y = parameters["y-axis"]["points"]
            self.args_rendering["minV"] = [-span_x/2.*self.conversion_factor_unit_to_volt,-span_y/2.*self.conversion_factor_unit_to_volt]
            self.args_rendering["maxV"] = [ span_x/2.*self.conversion_factor_unit_to_volt, span_y/2.*self.conversion_factor_unit_to_volt]
            self.args_rendering["resolution"] = [points_x,points_y]
                        
            self._plot_parameters_changed = True   # Does not really matter whether here or above... 
                                                   # Here: plot parameters are changed, perform_actual_acquisition() might be called before _plot_parameters_changed is set to True --> an empty array will be returned
                                                   # Above: _plot_parameters_changed is set to True, perform_actual_acquisition() might be called before the plot parameters were changed --> new simulated image base on the old plot parameters, _plot_parameters_changed set to False --> empty arrays returned afterwards until a plot parameter is changed

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
        return flags

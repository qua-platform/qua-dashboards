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
        self._initial_m = None  
        self.m = None
        self._first_acquisition: bool = True
        self._plot_parameters_changed: bool = False
        logger.debug(
            f"Initializing SimulatedDataAcquirer (ID: {component_id}) with "
            f"acquire_time: {self.acquire_time}s"
        )
        super().__init__(
            component_id=component_id, x_axis=x_axis, y_axis=y_axis, **kwargs
        )    

    # def get_voltage_control_component(self, voltage_parameters) -> VoltageControlComponent:
    #     self.voltage_parameters = voltage_parameters
    #     logger.debug(f"voltage_parameters: {self.voltage_parameters}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #     logger.debug(f"get voltage parameter x: {self.voltage_parameters[0].get()}")
    #     logger.debug(f"get voltage parameter y: {self.voltage_parameters[1].get()}")

    #     # Get the VoltageControlComponent
    #     voltage_controller = VoltageControlComponent(
    #         component_id="voltage_control",
    #         voltage_parameters=voltage_parameters,
    #     )

    #     return voltage_controller
    
    def get_voltage_control_component(self) -> VoltageControlComponent:
        self.m = self.experiment.tunneling_sim.boundaries(self.args_rendering["state_hint_lower_left"]).point_inside  # initial guess for m
        self._initial_m = copy.deepcopy(self.m)  # Store the initial m value
        self.args_rendering["m"] = self.m
        logger.debug(f"initial m: {self.m}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # ABSOLUT
        voltage_parameters = [
           BasicParameter("vg1", "Gate 1 (x)", "mV", initial_value=self.m[0] * 1./self.conversion_factor_unit_to_volt),
           BasicParameter("vg2", "Gate 2 (y)", "mV", initial_value=self.m[1] * 1./self.conversion_factor_unit_to_volt),
           BasicParameter("vg3", "Sensor Gate", "mV", initial_value=self.m[2] * 1./self.conversion_factor_unit_to_volt)
        ]
        # RELATIVE
        # voltage_parameters = [
        #     BasicParameter("vg1", "Gate 1 (x)", "mV", 0),
        #     BasicParameter("vg2", "Gate 2 (y)", "mV", 0),
        #     BasicParameter("vg3", "Sensor Gate", "mV", 0)
        # ]         
        self.voltage_parameters = voltage_parameters
        self._last_voltage_parameters = copy.deepcopy(voltage_parameters)  # Store the initial voltage parameters
        logger.debug(f"voltage_parameters: {self.voltage_parameters}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logger.debug(f"get voltage parameter x: {self.voltage_parameters[0].get()}")
        logger.debug(f"get voltage parameter y: {self.voltage_parameters[1].get()}")

        # Get the VoltageControlComponent
        voltage_controller = VoltageControlComponent(
            component_id="voltage_control",
            voltage_parameters=voltage_parameters,
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

            # Generate the simulated image          
            logger.info("Generating simulated data")
            #_ , _ , _ , _ , sensor_signalexp , _ = self.experiment.generate_CSD(**self.args_rendering)
            tsim = self.experiment.tunneling_sim
            sensor_signalexp = tsim.sensor_scan_2D(**self.args_rendering)

            # In case of several sensors, just pick the first one
            if sensor_signalexp.ndim == 3:
                sensor_values = sensor_signalexp[:,:,0].T
            else:
                sensor_values = sensor_signalexp.T

            self.simulated_image = sensor_values

        # First acquisition  
        if self._first_acquisition:
            logger.debug("First acquisition, generating simulated image")
            self._first_acquisition = False
            #self._last_voltage_parameters = copy.deepcopy(self.voltage_parameters)  # Store the initial voltage parameters
            #self.m = self.experiment.tunneling_sim.boundaries(self.args_rendering["state_hint_lower_left"]).point_inside  # initial guess for m
            #self._initial_m = copy.deepcopy(self.m)  # Store the initial m value
            #self.args_rendering["m"] = self.m
            #logger.debug(f"Initial m: {self.m} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logger.debug(f"args_rendering: {self.args_rendering} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            generate_simulated_image()

        # Check if voltage parameters changed
        voltage_parameters_changed = self.voltage_parameters[0].get() != self._last_voltage_parameters[0].get() or \
                                     self.voltage_parameters[1].get() != self._last_voltage_parameters[1].get() or \
                                     self.voltage_parameters[2].get() != self._last_voltage_parameters[2].get()  # Assuming a third voltage parameter for the sensor gate
        logger.debug(f"voltage_parameters_changed: {voltage_parameters_changed} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        # Plot and/or voltage parameters changed
        if voltage_parameters_changed or self._plot_parameters_changed:
            #logger.debug("Plot or voltage parameters changed, generating new simulated image")
            if self._plot_parameters_changed:
                #logger.debug("Plot parameters changed")
                self._plot_parameters_changed = False  # Do this here to PREVENT race conditions! It takes long to simulate the new image, and every change of plotting parameters would lead to race conditions.
            if voltage_parameters_changed:
                #logger.debug("Voltage parameters changed")
                self._last_voltage_parameters = copy.deepcopy(self.voltage_parameters)  # Update the last voltage parameters
                # RELATIVE
                # self.m = copy.deepcopy(self._initial_m)  # Reset m to the initial value
                # self.m[0] += self._last_voltage_parameters[0].get() * self.conversion_factor_unit_to_volt
                # self.m[1] += self._last_voltage_parameters[1].get() * self.conversion_factor_unit_to_volt
                # self.m[2] += self._last_voltage_parameters[2].get() * self.conversion_factor_unit_to_volt  # Assuming a third voltage parameter for the sensor gate
                # self.args_rendering["m"] = self.m  # Use the current m value
                # ABSOLUTE
                self.m = copy.deepcopy(self.m)   # IT DOESN'T WORK WITHOUT DEEP COPY - WHY???
                self.m[0] = self._last_voltage_parameters[0].get() * self.conversion_factor_unit_to_volt
                self.m[1] = self._last_voltage_parameters[1].get() * self.conversion_factor_unit_to_volt
                self.m[2] = self._last_voltage_parameters[2].get() * self.conversion_factor_unit_to_volt  # Assuming a third voltage parameter for the sensor gate
                self.args_rendering["m"] = self.m
                #logger.debug(f"Updated m: {self.m} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                logger.debug(f"args_rendering: {self.args_rendering} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                # Clear the data history
                with self._data_lock:  
                    self._data_history_raw.clear()
                    self._latest_processed_data = None                
            generate_simulated_image()
        
        else:
            sleep(self.acquire_time)  # Simulate acquisition time

        # # Generate simulated image in case of first acquisition, when plot parameters changed, or m changed
        # if self._plot_parameters_changed or self._first_acquisition:
        #     self._first_acquisition = False
        #     self._plot_parameters_changed = False  # Do this here to PREVENT race conditions! It takes long to simulate the new image, and every change of plotting parameters would lead to race conditions.
        #     logger.debug(
        #         f"SimulatedDataAcquirer (ID: {self.component_id}): "
        #         f"Generating simulated data for {self.y_axis.points}x{self.x_axis.points}"
        #     )

        #     # Ensure y_axis.points and x_axis.points are positive integers
        #     if self.y_axis.points <= 0 or self.x_axis.points <= 0:
        #         logger.warning(
        #             f"SimulatedDataAcquirer (ID: {self.component_id}): Invalid points "
        #             f"({self.y_axis.points}x{self.x_axis.points}). Returning empty array."
        #         )
        #         return np.array([[]])  # Return a 2D empty array to avoid downstream errors

        #     # Generate the simulated image          
        #     logger.info("Generating simulated data")
        #     #_ , _ , _ , _ , sensor_signalexp , _ = self.experiment.generate_CSD(**self.args_rendering)
        #     tsim = self.experiment.tunneling_sim
        #     self.args_rendering["m"] = tsim.boundaries(self.args_rendering["state_hint_lower_left"]).point_inside
        #     #logger.debug(f"self.args_rendering m: {self.args_rendering["m"]} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #     self.args_rendering["cache"]=False
        #     sensor_signalexp = tsim.sensor_scan_2D(**self.args_rendering)

        #     # In case of several sensors, just pick the first one
        #     if sensor_signalexp.ndim == 3:
        #         sensor_values = sensor_signalexp[:,:,0].T
        #     else:
        #         sensor_values = sensor_signalexp.T

        #     self.simulated_image = sensor_values
            
        # # After first acquisition or when parameters unchanged
        # else:
        #     sleep(self.acquire_time)

        # Add uniformly distributed noise to the simulated image 
        logger.info(f"Add noise to the image (SNR = {self.SNR})")
        mean_signal = np.mean(self.simulated_image)
        factor = np.sqrt(12)*mean_signal/self.SNR
        noise = np.random.rand(self.y_axis.points, self.x_axis.points)*factor

        #logger.debug(f"Shape simulated image: {self.simulated_image.shape} !!!!!!!!!!!!!!!!!!!!!!!!!!")
        #logger.debug(f"Shape noise: {noise.shape} !!!!!!!!!!!!!!!!!!!!!!!!!!")

        # Check if shape of axes still match the shape of the image
        if self.simulated_image.shape[0] != self.y_axis.points or self.simulated_image.shape[1] != self.x_axis.points:  # race conditions
            logger.warning("Axes resolution changed during rendering !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logger.info(f"signal shape: {self.simulated_image.shape}")
            logger.info(f"y_points: {self.y_axis.points}")
            logger.info(f"x_points: {self.x_axis.points}")
            #logger.debug(f"flag plot parameters changed: {self._plot_parameters_changed}")
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

    def update_parameters(self, parameters: Dict[str, Dict[str, Any]]) -> ModifiedFlags:
        """Updates SimulatedDataAcquirer parameters based on UI input.

        Args:
            parameters: A dictionary where keys are component IDs and values are
                dictionaries of parameter names to their new values.

        Returns:
            ModifiedFlags indicating what aspects of the component were changed.
        """
        flags = super().update_parameters(parameters)
        logger.info(f"parameters: {parameters}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #parameters: {'simulated-data-acquirer': {'num_software_averages': 5, 'acquire-time': 1}, 'x-axis': {'span': 4.4, 'points': 50}, 'y-axis': {'span': 4.2, 'points': 50}}

        if flags & ModifiedFlags.PLOT_PARAMETERS_MODIFIED:
            # Update the rendering arguments
            logger.info("Updating the rendering arguments")
            span_x = parameters["x-axis"]["span"]
            points_x = parameters["x-axis"]["points"]
            span_y = parameters["y-axis"]["span"]
            points_y = parameters["y-axis"]["points"]            
            #self.args_rendering["x_voltages"] = np.linspace(-span_x/2., span_x/2., points_x)*self.conversion_factor_unit_to_volt
            #self.args_rendering["y_voltages"] = np.linspace(-span_y/2., span_y/2., points_y)*self.conversion_factor_unit_to_volt
            #logger.debug(f"rendering parameters: {self.args_rendering}")
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

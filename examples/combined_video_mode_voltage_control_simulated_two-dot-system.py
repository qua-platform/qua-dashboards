"""
Example Script: 
Video Mode with SimulatedDataAcquirer for a two dot system with one sensor, and voltage control

"""

from qua_dashboards.core import build_dashboard
from qua_dashboards.utils import setup_logging
from qua_dashboards.video_mode import (
    SweepAxis,
    SimulatedDataAcquirer,
    VideoModeComponent,
)
from qdarts.experiment import Experiment
import numpy as np

from qua_dashboards.voltage_control import VoltageControlComponent
from qua_dashboards.utils import BasicParameter


def get_video_mode_component() -> VideoModeComponent:
    """
    Creates and returns a VideoModeComponent instance configured with a SimulatedDataAcquirer.

    This function encapsulates the setup of the sweep axes and the data acquirer,
    making it easy to reuse this configuration or import it into other scripts.

    Returns:
        VideoModeComponent: The configured video mode component.
    """
    # Define the system

    #All capacitances are given in aF
    N = 3 #number of dots   
    C_DD=20* np.eye((N))/ 2 #The self-capacitance of each dot, NOTE: factor of 2 due to symmetrization
    C_DD[1,1] = 10 / 2 #NOTE: factor of 2 due to symmetrization
    C_DD[0,1] = 5 #capacitance between dot 0 and dot 1 (Left double dot) 

    C_DD[0,2] = 1.6/2 #capacitance between sensor dot 2 and dot 0
    C_DD[1,2] = 1.4/2 #capacitance between sensor dot 2 and dot 1
    C_DD = C_DD + C_DD.T

    # C_DG=11*np.eye(N) #dot-to-gate capacitances 
    C_DG=11*np.eye(N,N+1) #dot-to-gate capacitances, there is one barrier gate (index 3)
    # cross-capacitances
    C_DG[0,1] = 1.5 #dot 0 from gate 1
    C_DG[1,0] = 1.2 #dot 1 from gate 0
    C_DG[0,3] = 0   #dot 0 from barrier gate
    C_DG[1,3] = 0   #dot 1 from barrier gate

    # Definition of the tunnel couplings in eV 
    # NOTE: we use the convention that tc is the energy gap at avoided crossing H = tc/2 sx
    tunnel_couplings = np.zeros((N,N))
    tunnel_couplings[0,1] = 50*1e-6
    tunnel_couplings[1,0] = 50*1e-6

    # Definition of barrier levels
    barrier_levers = np.zeros((N,N,N+1))  # barrier between dot i and dot j is affected by gate k
    barrier_levers[0,1,3] = 100*1e-6
    barrier_levers[1,0,3] = 100*1e-6
    barrier_levers = np.log(barrier_levers + 1.e-20)

    # Experiment configurations
    capacitance_config = {
        "C_DD" : C_DD,  #dot-dot capacitance matrix
        "C_Dg" : C_DG,  #dot-gate capacitance matrix
        "ks" : 4,       #distortion of Coulomb peaks. NOTE: If None -> constant size of Coublomb peak 
    }

    tunneling_config = {
            "tunnel_couplings": tunnel_couplings, #tunnel coupling matrix
            "temperature": 0.1,                   #temperature in Kelvin
            "energy_range_factor": 1,  #energy scale for the Hamiltonian generation. NOTE: Smaller -> faster but less accurate computation 
            "barrier_levers": barrier_levers,  #barrier levels matrix
    }

    sensor_config = {
            "sensor_dot_indices": [2],  #Indices of the sensor dots
            "sensor_detunings": [-0.0005],  #Detuning of the sensor dots
            "noise_amplitude": {"fast_noise": 0.5*1e-6, "slow_noise": 1e-8}, #Noise amplitude for the sensor dots in eV
            "peak_width_multiplier": 15,  #Width of the sensor peaks in the units of thermal broadening m *kB*T/0.61.
    }

    # Set up experiment
    experiment = Experiment(capacitance_config, tunneling_config, sensor_config)

    # Arguments for the function that renders the capacitance CSD
    unit = 'mV'
    factor_mV_to_V = 1e-3
    span_x = 20
    span_y = 20
    points_x = 50
    points_y = 50

    P=np.zeros((N+1,2))
    P[0,0]=1
    P[1,1]=1
    state_hint_lower_left = [1,1,5]

    W = np.eye(N+1)

    args_sensor_scan_2D = {
        "P": P,
        "virtualisation_matrix": W,
        "minV": [-span_x/2.*factor_mV_to_V,-span_y/2.*factor_mV_to_V],
        "maxV": [ span_x/2.*factor_mV_to_V, span_y/2.*factor_mV_to_V],
        "resolution": [points_x,points_y],
        "state_hint_lower_left": state_hint_lower_left,
        "cache": False,  # Do not cache the results
        "insitu_axis": None,
    }

    # Define the X-axis for the 2D scan.
    x_axis = SweepAxis(
        name="x",  # Internal identifier for the axis.
        label="X Coordinate",  # Display label in the UI.
        units=unit,       # Physical units of the axis.
        span=span_x,      # The total range of the X-axis sweep.
        points=points_x,  # The number of points (pixels) along the X-axis.
    )

    # Define the Y-axis for the 2D scan.
    y_axis = SweepAxis(
        name="y",  # Internal identifier for the axis.
        label="Y Coordinate",  # Display label in the UI.
        units=unit,       # Physical units of the axis.
        span=span_y,      # The total range of the Y-axis sweep.
        points=points_y,  # The number of points (pixels) along the Y-axis.
    )

    # Instantiate the SimulatedDataAcquirer.
    # This acquirer simulates data fetching by generating random 2D arrays.
    simulated_acquirer = SimulatedDataAcquirer(
        component_id="simulated-data-acquirer",  # Unique ID for Dash elements. 
        x_axis=x_axis,
        y_axis=y_axis,
        experiment = experiment,
        args_rendering = args_sensor_scan_2D,
        conversion_factor_unit_to_volt=factor_mV_to_V,
        SNR=20,  # Signal-to-noise ratio on simulated images
        acquire_time=0.1,  # Simulated delay (seconds) for acquiring one raw frame.
        num_software_averages=5,  # Number of raw frames to average for display.
        acquisition_interval_s=0.5,  # Target time (seconds) between acquiring raw frames.
    )

    # Instantiate the VideoModeComponent.
    # This is the main UI component that displays the live 2D plot and controls.
    video_mode_component = VideoModeComponent(
        component_id=VideoModeComponent.DEFAULT_COMPONENT_ID,  # Uses a default ID.
        data_acquirer=simulated_acquirer,  # The source of the data.
        # How often the frontend asks for new data (linked to acquirer's interval).
        data_polling_interval_s=simulated_acquirer.acquisition_interval_s,
    )
    return video_mode_component


def get_voltage_control_component(video_mode_component, labels) -> VoltageControlComponent:
    voltage_controller = video_mode_component.data_acquirer.get_voltage_control_component(labels=labels)
    return voltage_controller


def main() -> None:
    """
    Sets up logging, creates the VideoModeComponent, builds the dashboard,
    and runs the Dash application server.
    """
    # Configure logging for the application.
    logger = setup_logging(__name__)
    logger.info("Starting Video Mode application with SimulatedDataAcquirer (based on QDarts).")

    # Get the configured VideoModeComponent.
    video_mode_component = get_video_mode_component()
    logger.info(
        f"VideoModeComponent instance created: {video_mode_component.component_id}"
    )

    # Get the VoltageControlComponent
    labels = ["Gate 1 (x)", "Gate 2 (y)", "Sensor Gate", "Barrier Gate"]
    voltage_control_component = get_voltage_control_component(video_mode_component, labels)
    logger.info(
        f"VoltageControlComponent instance created: {voltage_control_component.component_id}"
    )  

    logger.info("Building the dashboard...")

    # Use build_dashboard to create the Dash app layout.
    app = build_dashboard(
        components=[video_mode_component,voltage_control_component],  # List of dashboard components to include.
        title="Video Mode Simulation (QDarts)",  # Browser window title.
    )

    logger.info("Dashboard built. Starting Dash server on http://localhost:8050")
    # Run the Dash server.
    app.run(
        debug=True,  # Enables helpful Dash debugging features.
        host="0.0.0.0",  # Makes the server accessible on your local network.
        port=8050,  # Sets the server port.
        use_reloader=False,  # Often recommended for stability with background threads.
    )


if __name__ == "__main__":
    # This ensures that main() is called only when the script is executed directly.
    main()
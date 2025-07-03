"""
Example Script: Video Mode with SimulatedDataAcquirer for a two dot system with one sensor

This script demonstrates how to use the VideoModeComponent with a SimulatedDataAcquirer.
The data is simulated using the QDarts package.
This setup is ideal for simulating and testing video mode dashboards without needing
a live connection to an OPX or other hardware. It allows you to understand the
dashboard's functionality, test UI interactions, and develop custom components
in a controlled environment.

Core Components Used:
- SweepAxis: Defines the parameters for each axis in the 2D scan (name, label,
             units, span, number of points).
- SimulatedDataAcquirer: A data acquirer that generates simulated data based on QDarts for the 2D scan,
                         simulating a real data acquisition process.
- VideoModeComponent: The main Dash component that orchestrates the video mode
                      display, taking a data acquirer as input and rendering
                      the live plot and controls.
- build_dashboard: A utility function from qua_dashboards.core to construct
                   a Dash application layout with the provided components.

How to Run:
1. Ensure you have `qua-dashboards` and its dependencies installed.
   (e.g., `pip install qua-dashboards`)
2. Save this script as a Python file (e.g., `run_random_video_mode.py`).
3. Run the script from your terminal: `python run_random_video_mode.py`
4. Open your web browser and navigate to `http://127.0.0.1:8050/` (or the
   address shown in your terminal).

You should see a dashboard titled "Video Mode Simulation (Simulated Data)"
displaying a 2D plot that updates with new simulated data periodically. You will
also have controls to adjust the parameters of the X and Y axes (Span and Points)
and the SimulatedDataAcquirer (Software Averages, Simulated Acquire Time).
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
    C_DD=20* np.eye((N))/2 #The self-capacitance of each dot, NOTE: factor of 2 due to symmetrization
    C_DD[0,1] = 10 #capacitance between dot 0 and dot 1 (Left double dot) 

    C_DD[0,2] = 1.6 #capacitance between sensor dot 4 and dot 0
    C_DD[1,2] = 1.4 #capacitance between sensor dot 4 and dot 1
    C_DD = C_DD + C_DD.T

    C_DG=11*np.eye(N) #dot-to-gate capacitances 
    #cross-capacitances
    C_DG[0,1] = 1.5 #dot 0 from dot 1
    C_DG[1,0] = 1.2 #dot 1 from dot 0

    # Definition of the tunnel couplings in eV 
    # NOTE: we use the convention that tc is the energy gap at avoided crossing H = tc/2 sx
    tunnel_couplings = np.zeros((N,N))
    tunnel_couplings[0,1] = 50*1e-6
    tunnel_couplings[1,0] = 50*1e-6

    # Experiment configurations
    capacitance_config = {
        "C_DD" : C_DD,  #dot-dot capacitance matrix
        "C_Dg" : C_DG,  #dot-gate capacitance matrix
        "ks" : 4,       #distortion of Coulomb peaks. NOTE: If None -> constant size of Coublomb peak 
    }

    tunneling_config = {
            "tunnel_couplings": tunnel_couplings, #tunnel coupling matrix
            "temperature": 0.1,                   #temperature in Kelvin
            "energy_range_factor": 5,  #energy scale for the Hamiltonian generation. NOTE: Smaller -> faster but less accurate computation 
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
    span_x = 4.4*3
    span_y = 4.2*3
    #span_x = 14
    #span_y = 20
    points_x = 50
    points_y = 50

    P=np.zeros((3,2))
    P[0,0]=1
    P[1,1]=1
    state_hint_lower_left = [1,1,3]

    args_sensor_scan_2D = {
        "P": P,
        "minV": [-span_x/2.*factor_mV_to_V,-span_y/2.*factor_mV_to_V],
        "maxV": [ span_x/2.*factor_mV_to_V, span_y/2.*factor_mV_to_V],
        "resolution": [points_x,points_y],
        "state_hint_lower_left": state_hint_lower_left,
        "cache": True,
        "insitu_axis": None,
    }

    args_generate_CSD = {
        "plane_axes": np.array([[1,0,0],[0,1,0]]), # vectors spanning the cut in voltage space
        "target_state": [1,1,5],  # target state for transition
        "target_transition": [-1,1,0], #target transition from target state, here transition to [2,3,2,3,5,5]
        "x_voltages": np.linspace(-span_x/2., span_x/2., points_x)*factor_mV_to_V, #voltage range for x-axis, originally: np.linspace(-0.0022, 0.0018, 100)
        "y_voltages": np.linspace(-span_y/2., span_y/2., points_y)*factor_mV_to_V, #voltage range for y-axis, originally: np.linspace(-0.0021, 0.0019, 100)
        "compute_polytopes": False, #compute the corners of constant occupation
        "compensate_sensors": False, #compensate the sensor signals
        "use_virtual_gates": False, #use the virtual gates
        "use_sensor_signal": True, #use the sensor signals     
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
        #args_rendering = args_generate_CSD,
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

    logger.info("Building the dashboard...")
    # Use build_dashboard to create the Dash app layout.
    app = build_dashboard(
        components=[video_mode_component],  # List of dashboard components to include.
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
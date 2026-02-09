# Examples

This document summarizes the main examples included in this folder, and how they differ. 

## Main examples

The main examples combine the various components together to create a dashboard which gives you full control over your device. There are three examples: 

### example_video_mode_full.py

This example is a basic, full implementation of video mode, integrating the VideoModeComponent with the VirtualGatesComponent and the VoltageControlComponent. It assumes the use of a QDACII controller, although this can be replaced by any external source to be used with quam-builder's VirtualDCSet. 

### hybrid_opx_qdac_example.py

This hybrid example sets up a version of VideoMode, where the Y axis is stepped via the QDAC, while the X axis is swept via the OPX. This is particularly useful for experiments which require a long time-scale, and use a bias tee. The example requires the setup of a QCoDeS QDACII, and a trigger to be sent from the OPX to a single external trigger port of the QDACII. The basic functionality is otherwise identical to the fully OPX video mode. 

### qd_quam_example_video_mode.py

Quantum Machines provides a detailed Quam structure specific for use with quantum dot expeirments. This example covers the instantiation of such a Quam state, and its use with Video Mode. While the script currently creates a Quam from scratch, if an existing Quam state exists, the user can simply load this Quam state instead of creating one from scratch. 

## examples/video_mode

The examples/video_mode folder contains examples unique to VideoMode, which do not use the VoltageControlComponent. Here, you will find [example_video_mode_opx.py](video_mode/example_video_mode_opx.py) and [example_video_mode_2_opx.py](video_mode/example_video_mode_2_opx.py). These are two ways to instantiate VideoMode, providing a thorough preview of the requirements. 

## examples/video_mode/simulated_examples

In this directory, we integrate video mode with various simulations. In [example_video_mode_random.py](video_mode/simulated_examples/example_video_mode_random.py), a RandomSimulator is instantiated to collect random data to display in Video Mode. One also has the options to instantiate a simulator, as in the example shown in [example_video_mode_simulation.py](video_mode/simulated_examples/example_video_mode_simulation.py), which integrates a Qarray simulation. To implement your own simulation, simply subclass the BaseSimulator in [../src/qua_dashboards/video_mode/inner_loop_actions/simulators/base_simulator.py](src/qua_dashboards/video_mode/inner_loop_actions/simulators/base_simulator.py), ensuring that the measure_data function returns the data in the correct format. 

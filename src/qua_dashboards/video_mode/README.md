# Video Mode

This module performs a continuous rapid 2D scan of two sweep axes, and measures a signal at each point. The results are shown through an interactive web frontend with a live plot and controls for the sweep parameters.

The video mode has been designed as a modular tool that is composed of four parts:

1. The `OPXDataAcquirer` class, which is responsible for the data acquisition.
2. The `ScanMode` class, which is responsible for how the 2D grid is traversed.
3. The `InnerLoopAction` class, which is responsible for what QUA code is performed during each step of the scan.
4. The `VideoModeApp` class, which handles interaction with the frontend.

The `ScanMode` and `InnerLoopAction` classes are highly flexible and can be selected/modified to suit the specific needs of the user. For example, three different scan modes (`RasterScan`, `SpiralScan`, and `SwitchRasterScan`) are provided, which can be used to acquire data in different ways. Similarly, the `InnerLoopAction` class can be modified to perform additional actions, such as adding specific pulses prior to each measurement.


## Basic Usage
To use the video mode, it is necessary to initialize the relevant classes and pass them to the `VideoMode` class. 
We will go through a simple example to demonstrate the video mode. Most of the classes and functions described here have additional options which can be found in the docstrings and in the source code.

If you don't have access to an OPX but still want to try the video mode, see the `Simulated Video Mode`section in `Advanced Usage`

First, we assume that a `QuantumMachinesManager` is already connected with variable `qmm`.


### Scan mode

First we can define the dictionary of scan modes to pass to Video Mode, so that one can easily switch scan modes in the UI. For this, we can import the scan modes as follows. 

```python
from qua_dashboards.video_mode import scan_modes
scan_mode_dict = {
    "Switch_Raster_Scan": scan_modes.SwitchRasterScan(), 
    "Raster_Scan": scan_modes.RasterScan(), 
    "Spiral_Scan": scan_modes.SpiralScan(),
}
```

This scan can be visualized by calling
```python
scan_mode.plot_scan(x_points, y_points)
```
where `x_points` and `y_points` are the number of sweep points along each axis.

### Inner loop action

The user has full freedom in the definition of the most inner loop sequence performed by the OPX which is defined under the `__call__()` method of an `InnerLoopAction` subclass.

For example, the `BasicInnerLoopAction` performs a reflectometry measurement after applying the relevant SweepAxis changes; in the default case, changing the x and y voltage values:

```python
def __call__(
        self, x: QuaVariableFloat, y: QuaVariableFloat
    ) -> List[QuaVariableFloat]:

        # Apply functions
        # For FrequencySweepAxis, applies update_frequency and returns empty dict.
        # For VoltageSweepAxis, updates the axis.last_val (will be changed once VoltageSequence is validated) and returns empty dict.
        # For AmplitudeSweepAxis, calculates amplitude scale and returns the scale as a dict component {element: scale}.
        # Add functionalities to existing BaseSweepAxis objects in their respective apply commands
        # For new SweepAxis objects (e.g. QubitSweepAxis), simply have an apply command that returns an empty dict.

        x_apply = self.x_axis.apply(x)
        y_apply = self.y_axis.apply(y) if (self.y_axis and y is not None) else None
        amplitude_scales = {
            **x_apply.get("amplitude_scales", {}),
            **(y_apply.get("amplitude_scales", {}) or {}),
        } if y_apply is not None else {**x_apply.get("amplitude_scales", {})}

        qua.align()
        duration = max(
            self._pulse_for(op).length for op in self.selected_readout_channels
        )
        if self.pre_measurement_delay > 0:
            duration += self.pre_measurement_delay
            qua.wait(duration // 4)
        qua.align()
        result = []
        for channel in self.selected_readout_channels:
            elem = channel.name
            scale = 1
            if elem in amplitude_scales:
                scale = amplitude_scales.get(elem, 1)
            I, Q = channel.measure(self._pulse_for(channel).id, amplitude_scale=scale)
            result.extend([I, Q])
        qua.align()

        for channel in self.selected_readout_channels:
            qua.ramp_to_zero(channel.name, duration=self.ramp_duration)
        qua.align()
        qua.wait(2000)

        return result
```

The `BasicInnerLoopAction` is instantiated automatically in the OPXDataAcquirer. If another inner loop action is preferred, instantiate it separately and hand it to the OPXDataAcquirer as an argument. 
Note that this `BasicInnerLoopAction` assumes that the `readout_pulse` has two integration weights called `cos` and `sin`


## GateSet

 Next, we must instantiate the GateSet. The GateSet serves as an abstraction to the handling of OPX outputs, as a combined means of operating quantum dot devices. This is explored in great detail in the relevant README. Virtual gates are available through the use of VirtualGateSet, which will be instantiated in this example. 

```python
from quam_builder.architecture.quantum_dots.components import VirtualGateSet  # Requires quam-builder
channels = {
    "ch1": machine.channels["ch1"].get_reference(), # .get_reference() necessary to avoid reparenting the Quam component
    "ch2": machine.channels["ch2"].get_reference(),
    "ch1_readout": machine.channels["ch1_readout"].get_reference()
}
gate_set = VirtualGateSet(id = "Plungers", channels = channels)
gate_set.add_layer(
    source_gates = ["V1", "V2"], # Pick the virtual gate names here 
    target_gates = ["ch1", "ch2"], # Must be a subset of gates in the gate_set
    matrix = [[1, 0.2], [0.2, 1]] # Any example matrix
)
machine.gate_set = gate_set
```

## OPXDataAcquirer

To use Video Mode, instantiate the OPXDataAcquirer. The OPXDataAcquirer takes the instantiated GateSet, as well as a series of readout_pulse objects to use for readout. Note that the frequency and pulse amplitudes of the channels associated with these readout pulses will be sweep-able. 

```python
from qua_dashboards.video_mode.data_acquirer import OPXDataAcquirer
data_acquirer = OPXDataAcquirer(
    qmm=qmm,
    machine=machine,
    gate_set=virtual_gate_set,  # Replace with your GateSet instance
    x_axis_name="ch1",  # Must appear in gate_set.valid_channel_names; Virtual gate names also valid
    y_axis_name="ch2",  # Must appear in gate_set.valid_channel_names; Virtual gate names also valid
    scan_modes=scan_mode_dict,
    result_type="I",  # "I", "Q", "amplitude", or "phase"
    available_readout_pulses=[readout_pulse_ch1, readout_pulse_ch2], # Input a list of pulses. The default only reads out from the first pulse, unless the second one is chosen in the UI. 
    acquisition_interval_s=0.05
)
```

You can now test the data acquirer before using it in video mode.
```python
data_acquirer.run_program()
results = data_acquirer.acquire_data()
```

Finally, we can start the video mode.
```python
from qua_dashboards.video_mode import VideoModeComponent
video_mode_component = VideoModeComponent(
    data_acquirer=data_acquirer,
    data_polling_interval_s=0.5,  # How often the dashboard polls for new data
    save_path = save_path
)
app = build_dashboard(
    components=[video_mode_component],
    title="OPX Video Mode Dashboard",  # Title for the web page
)
logger.info("Dashboard built. Starting Dash server on http://localhost:8050")
app.run(debug=True, host="0.0.0.0", port=8050, use_reloader=False)
```


You can now access video mode from your browser at `http://localhost:8050/` (the port may be different, see the output logs for details).

Note that one can add the VoltageControlTabController to VideoModeComponent for combined control of OPX voltages with external voltage sources. 


## Advanced Usage
### Voltage offsets

The `SweepAxis` class has an `offset_parameter` attribute, which is an optional parameter that defines the sweep offset. This can be a QCoDeS DC voltage source parameter or a `VoltageParameter` object.

As an example, let us assume that we have a QCoDeS parameter `x_gate` for the DC voltage of a gate:

```python
x_offset()  # Returns the DC voltage, e.g. 0.62
```

In this case, we can pass this parameter to the `SweepAxis` class to define the sweep offset.
```python
x_axis = VoltageSweepAxis(name="gate", span=0.03, points=51, offset_parameter=x_offset)
```
The video mode plot should now correctly show the sweep axes with the correct offset.

Note that if the offset voltage is changed, it will need to be changed in the same kernel where the video mode is running. One solution for this is using the `VoltageControl` module in py-qua-tools.


### Simulated Video Mode
Below is an example of how to run the video mode without an actual OPX.
In this case, we will use the `RandomDataAcquirer` class, which simply displays uniformly-sampled random data.
```python
from qua_dashboards.video_mode import *

x_axis = SweepAxis(name="X", span=0.1, points=101)
y_axis = SweepAxis(name="Y", span=0.1, points=101)

data_acquirer = RandomDataAcquirer(
    x_axis=x_axis,
    y_axis=y_axis,
    num_averages=5,
)

live_plotter = VideoModeApp(data_acquirer=data_acquirer)
live_plotter.run()
```

# Debugging

To see the logs which include useful debug information, you can update the logging configuration.

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("hpack.hpack").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
```
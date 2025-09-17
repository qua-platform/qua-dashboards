# Virtual Gating Dashboard

The `VirtualLayerEditor` within `qua-dashboards` allows one to add and edit VirtualGateSet layers with user-input matrices and layer names for clarity. This component is essential for quantum dot experiments, in which one must account for both cross-capacitive gate voltage effects and arbitrary axes on the charge stability diagram.  

## Core Functionality

- **View Layers**: Displays the virtualisation layer saved in the VirtualGateSet; it's matrix, source and target gates, and the assigned name in the UI. 

- **Update Layer Matrices**: Allows users to input new matrix values in existing virtualisation layers in the VirtualGateSet, as well as resetting easily back to identity. 

- **Virtualisation Layer Addition**: Allows users to easily add a new virtualisation layer to the VirtualGateSet to a list of target gates which are unassigned. Allows users to assign a name to each layer for clarity. 

## Integration with Other Dashboards

`VirtualLayerEditor` can be used with other `qua-dashboards` components, such as the `VoltageControlComponent` and the `VideoModeComponent`. This allow for experimental workflows in which: 

- **Manage existing virtual layers interactively**: Edit the existing virtualisation layers, and see immediately the effect on the video mode output, to live-tune the matrix elements. 

- **Add new virtualisation layers**: If an arbitrary axis is identified, add a new virtual layer to easily rotate the plane to scan along this new axis

  For instance, you can use the `VirtualLayerEditor` to rotate the stability diagram of a 2D scan performed by the Video Mode.

This combination enables efficient device tuning and characterization by providing direct control and immediate visual feedback.

## Examples

The `examples/virtual_gating` folder contains scripts to help you get started:

### 1. Simulated VirtualGateSet Control ([example_virtual_gating.py](example_virtual_gating.py))

- **Purpose**: This example uses a simulated Quam machine, comprising of an arbitary number of SingleChannel elements. 
  It's designed for **testing the functionality and familiarizing yourself with the `VirtualLayerEditor` interface** without needing any actual hardware.

- **Use Case**:

  - Understand how to edit and add virtualisation layers.

  - Test the UI for editing and adding virtualisation layers, and observing currently existing layers.

  - Develop custom dashboard layouts incorporating Virtual Gating in a simulated environment.

- **How it Works**: It creates a basic machine (machine = BasicQuam()), in which it adds a user-defined number of simulated SingleChannel elements (Plunger1, Plunger2, etc). These represent the quantum device gate connections (and therefore the set of physical target gates), upon which the virtual layers can be applied. 
  These are then passed to the `VirtualLayerEditor` to create the dashboard.
"""
Helper functions for annotation tasks.

Includes functions for drawing annotations on figures, calculating distances,
loading/saving annotation files, and performing basic analysis like slope calculation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import plotly.graph_objects as go

from dataclasses import dataclass # dataclass decorator: automatically generates special methods (e.g. __init__, __repr__, __eq__, etc.) for classes primarily used to store data
from typing import Callable
import autograd.numpy as anp
import autograd.scipy.stats as stat
from autograd import value_and_grad
import scipy.optimize
from scipy.interpolate import interpn
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import cv2 as cv
import autograd.numpy as anp
from autograd import value_and_grad
import matplotlib.pyplot as plt
from scipy.interpolate import interpn 


logger = logging.getLogger(__name__)

__all__ = [
    "generate_annotation_traces",
    "get_point_coords_by_id",
    "find_closest_point_id",
    "find_closest_line_id",
    "calculate_slopes",
    "compute_gate_compensation_ml",
    "compute_gate_compensation_al",
    "compute_transformation_matrix_from_image_gradients",
    "compute_transformation_matrix",
    "warp_image_with_normals"
]


def generate_annotation_traces(
    annotations_data: Dict[str, List[Dict[str, Any]]],
    viewer_ui_state_input: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Generates Plotly trace dictionaries for annotation points and lines.

    Args:
        annotations_data: Dictionary containing lists of point and line objects.
            Expected structure:
            {
                "points": [{"id": str, "x": float, "y": float, ...}, ...],
                "lines": [{"id": str, "start_point_id": str, "end_point_id": str, ...}, ...]
            }
        selected_point_to_move_id: The ID of the point currently selected for moving.
        selected_indices_for_line: A list of point IDs currently selected for forming a line.

    Returns:
        A list of Plotly trace dictionaries.
    """
    traces = []
    points = annotations_data.get("points", [])
    lines = annotations_data.get("lines", [])

    if not points:
        return []

    # Highlighting and labels
    selected_point_to_move_id = viewer_ui_state_input.get("selected_point_to_move",None)
    selected_indices_for_line = viewer_ui_state_input.get("selected_point_for_line",[])
    labels_list = viewer_ui_state_input.get("show_labels",[])
    
    point_x_coords = [p["x"] for p in points]
    point_y_coords = [p["y"] for p in points]
    point_ids = [p["id"] for p in points]

    # Prepare line coordinates
    line_x_coords: List[Optional[float]] = []
    line_y_coords: List[Optional[float]] = []

    for line in lines:
        start_coords = get_point_coords_by_id(annotations_data, line["start_point_id"])
        end_coords = get_point_coords_by_id(annotations_data, line["end_point_id"])

        if start_coords and end_coords:
            line_x_coords.extend([start_coords[0], end_coords[0], None])
            line_y_coords.extend([start_coords[1], end_coords[1], None])
        else:
            logger.warning(
                f"Could not find coordinates for points in line {line['id']}: "
                f"start={line['start_point_id']}, end={line['end_point_id']}"
            )

    # Generate marker sizes and colors (highlight selected points)
    sizes = []
    marker_colors = []  # Example: can be used for different point types or states

    for p_id in point_ids:
        is_selected_for_move = p_id == selected_point_to_move_id
        is_selected_for_line = p_id in selected_indices_for_line

        if is_selected_for_move or is_selected_for_line:
            sizes.append(13)
        else:
            sizes.append(10)
        # Example color logic (can be expanded)
        marker_colors.append(
            "rgba(255, 255, 255, 1)"
            if not (is_selected_for_move or is_selected_for_line)
            else "rgba(255, 0, 0, 1)"
        )

    # Point labels (e.g., "P1", "P2" based on order or a specific label property)
    texts = [p_id for p_id in point_ids] if 'points' in labels_list else None

    points_trace = go.Scatter(
        x=point_x_coords,
        y=point_y_coords,
        mode="markers+text",
        marker=dict(
            color=marker_colors,  # Use dynamic colors
            size=sizes,
            line=dict(color="black", width=1),
            opacity=1.0,  # Opacity can also be dynamic
        ),
        text=texts,
        textposition="top center",
        textfont=dict(color="white", size=10),
        hoverinfo="text",  # Show text on hover
        customdata=point_ids,  # Store unique point ID
        name="annotations_points",  # Consistent name for viewer identification
        zorder=2,
    ).to_plotly_json()
    traces.append(points_trace)

    if line_x_coords:
        lines_trace = go.Scatter(
            x=line_x_coords,
            y=line_y_coords,
            mode="lines",
            line=dict(
                color="rgba(255, 255, 255, 1)", width=2
            ),  # Slightly transparent white
            hoverinfo="none",
            name="annotations_lines",  # Consistent name
            zorder=1,
        ).to_plotly_json()
        traces.append(lines_trace)

    return traces


def get_point_coords_by_id(
    annotations_data: Dict[str, List[Dict[str, Any]]], point_id: str
) -> Optional[Tuple[float, float]]:
    """
    Gets coordinates of a point given its unique string ID.

    Args:
        annotations_data: The main annotations data structure.
        point_id: The string ID of the point.

    Returns:
        A tuple (x, y) of coordinates, or None if not found.
    """
    for point in annotations_data.get("points", []):
        if point["id"] == point_id:
            return point["x"], point["y"]
    logger.warning(f"Point with ID '{point_id}' not found in annotations_data.")
    return None


def find_closest_point_id(
    x_click: float,
    y_click: float,
    annotations_data: Dict[str, List[Dict[str, Any]]],
    tolerance: float,
) -> Optional[str]:
    """
    Finds the unique ID of the closest annotation point within a tolerance.

    Args:
        x_click: X-coordinate of the click event.
        y_click: Y-coordinate of the click event.
        annotations_data: The main annotations data structure.
        tolerance: The maximum distance to consider a point "close".

    Returns:
        The string ID of the closest point, or None.
    """
    min_dist_sq = tolerance**2
    closest_p_id: Optional[str] = None

    for point in annotations_data.get("points", []):
        try:
            px, py = float(point["x"]), float(point["y"])
            dist_sq = (x_click - px) ** 2 + (y_click - py) ** 2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_p_id = point["id"]
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(
                f"Invalid data for point ID {point.get('id', 'N/A')} "
                f"during distance calculation: {e}"
            )
            continue
    return closest_p_id


def find_closest_line_id(
    x_click: float,
    y_click: float,
    annotations_data: Dict[str, List[Dict[str, Any]]],
    tolerance: float,
) -> Optional[str]:
    """
    Finds the ID of the closest annotation line within a tolerance.

    Args:
        x_click: X-coordinate of the click event.
        y_click: Y-coordinate of the click event.
        annotations_data: The main annotations data structure.
        tolerance: The maximum distance to consider a line "close".

    Returns:
        The string ID of the closest line, or None.
    """
    max_dist = tolerance
    closest_l_id: Optional[str] = None

    logger.debug(
        f"Starting find_closest_line_id ||||||||||||||| with x_click={x_click}, y_click={y_click}, tolerance={tolerance}"
    )
    logger.debug(
        f"||||||||||||||| Annotations data lines: {annotations_data.get('lines', [])}"
    )

    for line in annotations_data.get("lines", []):
        logger.debug(f"||||||||||||||| Checking line: {line}")
        coords1 = get_point_coords_by_id(annotations_data, line["start_point_id"])
        coords2 = get_point_coords_by_id(annotations_data, line["end_point_id"])

        logger.debug(
            " ||||||||||||||| Start coords: %s, End coords: %s",
            coords1,
            coords2,
        )

        if coords1 is None or coords2 is None:
            logger.warning(
                f"||||||||||||||| Skipping line {line['id']} due to missing point coordinates."
            )
            continue

        x1, y1 = coords1
        x2, y2 = coords2

        dx, dy = x2 - x1, y2 - y1
        d_sq = dx**2 + dy**2

        logger.debug(f"||||||||||||||| dx: {dx}, dy: {dy}, d_sq: {d_sq}")

        if np.isclose(d_sq, 0):  # Line is a point
            dist = np.sqrt((x_click - x1) ** 2 + (y_click - y1) ** 2)
            logger.debug(
                f"||||||||||||||| Line is a point. Distance from click: {dist}"
            )
        else:
            # Project click point onto the line segment
            t = ((x_click - x1) * dx + (y_click - y1) * dy) / d_sq
            t = max(0.0, min(1.0, t))  # Clamp t to the segment
            proj_x, proj_y = x1 + t * dx, y1 + t * dy
            dist = np.sqrt((x_click - proj_x) ** 2 + (y_click - proj_y) ** 2)
            logger.debug(
                f"||||||||||||||| Projection t: {t}, proj_x: {proj_x}, proj_y: {proj_y}, "
                f"Distance: {dist}"
            )

        if dist < max_dist:
            logger.debug(
                f"||||||||||||||| New closest line found: {line['id']} with distance {dist}"
            )
            max_dist = dist
            closest_l_id = line["id"]

    logger.debug(
        f"||||||||||||||| Closest line id: {closest_l_id} with distance {max_dist}"
    )
    return closest_l_id


def calculate_slopes(
    annotations_data: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, float]:
    """
    Calculates slopes for annotated lines.

    Args:
        annotations_data: The main annotations data structure.
            Expected: {"points": [...], "lines": [...]}

    Returns:
        A dictionary mapping line ID (string) to its slope.
    """
    slopes: Dict[str, float] = {}
    lines = annotations_data.get("lines", [])

    for line in lines:
        line_id = line["id"]
        coords1 = get_point_coords_by_id(annotations_data, line["start_point_id"])
        coords2 = get_point_coords_by_id(annotations_data, line["end_point_id"])

        if coords1 and coords2:
            x1, y1 = coords1
            x2, y2 = coords2
            delta_x = x2 - x1
            delta_y = y2 - y1
            if np.isclose(delta_x, 0):
                slope_val = (
                    float("inf")
                    if delta_y > 0
                    else float("-inf")
                    if delta_y < 0
                    else float(
                        "nan"
                    )  # Undefined (points coincide or vertical line of zero length)
                )
                slopes[line_id] = slope_val
            else:
                slopes[line_id] = delta_y / delta_x
        else:
            slopes[line_id] = float("nan")  # Cannot calculate if points are missing

    logger.info(f"Calculated slopes for {len(slopes)} lines.")
    return slopes


# Define the class Lines (groups three NumPy arrays together)
@dataclass
class Lines:
    sensorv: np.ndarray        # repeated input values, shape: (num_gate_values, num_samples)   corresponds to (num_measurements, N), i.e. (gate values tested,points per sensor ramp)
    other_gates: np.ndarray    # gate values, shape: (num_gate_values,)
    obs: np.ndarray            # actual observations, shape: (num_gate_values, num_samples)


@dataclass
class Model:
    solution: anp.ndarray                         # optimized parameters of the model, shape (N+2,). solution[0] is w0 (slope for compensation). solution[1:] parameters for the background model, solution[1] is bias b, solution[2:] are Gaussian weights.
    model: Callable                               # function that takes inputs Xs and parameters
    woffset: float                                # constant offset value, the value of the gate where we centered the linear compensation (== mid_val_gate)!!!  It is the linear shift of the compensation model.
    res: scipy.optimize._optimize.OptimizeResult  # result of BFGS optimization (e.g. final loss, number of iterations, convergence status, etc.)

    # Function to make the class callable like a function
    # Once the model is trained, you can evaluate how well it captures sensor behaviour under gate influence by calling: model(Xs)  -->  predict outputs, plot residuals, ...
    def __call__(self, Xs):
        return self.model(Xs, self.solution)  # returns predicted sensor observations, shape (M,)


def fit_compensation_parameters(lines, N, mid_val_gate, min_w0  = 0.0, max_w0  = 0.4, num_trials=2, max_iterations=1000000, epsilon=1.e-5):
    """
    Model fitting for sensor compensation based on gate values and sensor voltages.
    Fits a model of the form sensor response = f(gate value, sensor value). This is to compensate how a control gate affects a sensor.
    Model is of the form obs(x_sensor,x_gate) = sum_i alpha_i Gaussian_pdf_i(x_sensor + w0(x_gate - midpoint)) + b
    Gaussians model smooth sensor response
    w0 is the learned linear slope that tells how the gate, we are compensating for, perturbs the sensor

    Input parameters:
        Lines dataclass: contains sensor voltage sweep values, the values of the gate we are compensating for, actual observations
        N: number of Gaussian basis functions
        mid_val_gate: central value of gate we are compensating for
        min_w0, max_w0: range of initial slope guesses for the compensation
        num_trials: how many different initial slopes to try

    Returns the best model (parameters, model function, linear shift of the compensation model, optimization result object)
    """
    # background model components: one parameter per Gaussian basis function plus one bias term
    num_params_bg = N + 1  

    # background feature construction
    # sensor is swept between these values --> total range
    x_min = lines.sensorv.min()  # minimum sensor sweep value across all gate values
    x_max = lines.sensorv.max()  # maximum

    # N Gaussian basis functions placed along the sensor axis to model the sensor's response curve. Gaussians are fixed in location and variance.
    # This background model captures the behaviour of the system in the absence of something interfering, i.e. the sensor's signal depending on it's own ramp voltage without interference from other gates
    # Background model: what the sensor should look like in ideal, isolated conditions
    def background_model(Zs, mean, var, alpha, b):  # definition, basis function regression model
        """
        Input parameters:
            Zs: (compensated) sensor voltage values at M points (input points to evaluate the model at). 1D array of shape (M,)
            mean: centers of the Gaussian bumps. 1D array of shape (N,)
            var: scalar controlling the width of each Gaussian
            alpha: weights for each Gaussian basis function. 1D array of shape (N,)
            b: bias
        
        Computes a design matrix of features, with a row per input point and a column per Gaussian basis function. 

        Returns the model's prediction at each Zs value (shape (M,)), which is a linear combination of features
        """
        N = mean.shape[0]  # number of Gaussion basis functions
        # Repeat the array Zs N times along the 1st axis, transpose it.
        Zs = anp.tile(Zs, (N, 1)).T  # Broadcast 1D array (shape (M,)) of sensor voltage values into 2D array (shape (M,N)), so that all Gaussion basis functions can be evaluated simultaneously evaluated at every point in Zs. 
        features = stat.norm.pdf(Zs, mean, var)  # Design matrix of features, 2D array of shape (M,N). Compute Gaussian basis functions: each column of features is a different Gaussian basis function, one row per input point.
        return features @ alpha + b  # (M,N) @ (N,) + scalar --> 1D array of shape (M,)

    # Background model
    def model_par_bg(Zs, parameters):
        """
        Input parameters:
            Zs: (compensated) sensor voltage values at M points (input points to evaluate the model at). 1D array of shape (M,)
            Parameter vector [b, alpha] of shape (N+1,)

        Returns the models's prediction at each Zs value
        """
        b = parameters[0]
        alpha = parameters[1:]

        # Q: why are we using abs() here?
        mean = anp.linspace(x_min - (x_max-x_min), x_max + (x_max-x_min), N)  # 1D array, shape (N,). The means are spread around [x_min, x_max] range (use extended range!) --> good coverage over the sensor sweep, ensures the model does NOT fail near the edges.
        var = (mean.max()-mean.min()) / N
        return background_model(Zs, mean, var, alpha, b)

    # Full compensation model
    def model_par(Xs, parameters):
        """
        Model that accounts for the effect of a gate on the sensor signal.

        Input parameters:
            Xs: 2D array of shape (num_samples, 2), where Xs[:,0] are the gate values and Xs[:,1] are the sensor values. I.e. each sample is [gate_value, sensor_value]
            parameters: 1D array of shape (N+2,)

        Returns the predicted sensor signal. Model obs(x_sensor,x_gate) = background_model(x_sensor + w0(x_gate-mid))
        Assumption: For the current gate, each gate value produces a shifted sensor curve. The model tries to find the slope that unshifts those curves and aligns them. The gate affects the sensor linearly with slope w0.
        """
        w0 = parameters[0]  # linear compensation slope: how does the current gate value affect the sensor! Learned from data. Entries in the matrix P!
        wrest = parameters[1:]  # N+1 parameters used by the background model (Gaussian weights and bias): define the shape of the sensor response when compensated
        # core compensation step
        Zs = (
            Xs[:, 1]
            + w0 * (Xs[:, 0] - mid_val_gate)  # subtract central value of gate so that the correction is zero at the central point! --> Makes the compensation gate-centered, ensuring the learned slope w0 is meaningful relative to each gate's typical operating range.
        )   
        return model_par_bg(Zs, wrest) # Plug the compensated sensor values in the background model

    # Fit wrapper
    def fit(model_func, x0, Xs, Zs, max_iterations, epsilon) -> scipy.optimize._optimize.OptimizeResult:   
        """
        Fit model parameters to minimize mean squared error (MSE) between predicted and observed data. Use BFGS gradient-based optimization (good for smooth functions) - it uses gradients and curvature info (approximated Hessian).

        Input parameters:
            model_func: a function that takes inputs Xs and a parameter vector and returns predictions
            x0: initial guess for the model parameters. 1D jax.Array
            Xs: model input data, 2D array typically of shape (num_samples, num_features)   num_samples == M == number of training data points
            Zs: target output of shape (num_samples,), true values from data
        
        Returns a scipy.optimize._optimize.OptimizeResult object containing .x (optimized parameter vector), .fun (final function value), .success (did it converge?), .jac, .hess, .nit (number of iterations), .message (description of cause of termination)
        """
        # define loss function
        def f(parameters):
            predictions = model_func(Xs, parameters)
            return anp.mean((predictions - Zs) ** 2)  # loss (scalar)  -  mean of a vector of shape (num_samples,) gives a scalar
        
        problem = value_and_grad(f)

        return scipy.optimize.minimize(problem,x0,method='BFGS',jac=True,tol=0.0,options={'maxiter':max_iterations, 'gtol':epsilon})

    ## Data preparation
    #n, m = lines.sensorv.shape
    num_gate_vals, num_sensor_samples = lines.sensorv.shape  # Note that M = num_gate_vals*num_sensor_samples!
    #lines_other_gatesv = anp.tile(lines.other_gates.reshape(n, 1, 1), (1, m, 1))
    # expand gate values to match shape: 1D array of shape (num_gate_vals,) --> 3D array of shape (num_gate_vals, num_sensor_samples, 1), the third dimension is reserved for stacking with the lines.sensorv data!
    lines_other_gatesv = anp.tile(
        lines.other_gates.reshape(num_gate_vals, 1, 1), # Reshapes 1D array into 3D array to prepare it for tiling across sensor samples, shape (num_gate_vals, 1, 1)
        (1, num_sensor_samples, 1)                      # Repeat each gate value across all sensor samples, shape (num_gate_val, num_sensor_samples, 1)
    )
    # Stack gate and sensor values. Note that lines.sensorv (2D) is automatically broadcasted to lines.sensorv[:, :, None] (3D) when stacking
    # First dim: gate values. Indexes over different values of the "other" gate (i.e. the gate being compensated for). "We probe the system at different settings of this other gate".
    # Second dim: indexes the sensor voltage ramp. For each gate setting, you're sweeping the sensor's voltage across N values. "We sample the response along the ramp of the sensor".
    # Third dim: features [gate,sensor]. 2D input vector per point Xs[i,j] = [gate_value_at_i, sensor_value_at_i_j]
    Xs = anp.dstack([lines_other_gatesv, lines.sensorv])  # Xs[:,:,0] are gate values (same along rows), Xs[:,:,1] are sensor values. 

    ## Fitting
    solution = anp.zeros(num_params_bg + 1) # best parameters
    f_best_sol = 1.0e10                     # best objective value (loss)
    res_best = None                         # best optimization result object
    # Fit background model first to initialize Gaussian weights and bias
    middle_obs = lines.obs.shape[0]//2  # index of the middle observation ("middle gate value")
    sol_bg = fit(model_par_bg, anp.zeros(num_params_bg), Xs[middle_obs, :, 1], lines.obs[middle_obs], max_iterations, epsilon)  # fit(model_func, x0, Xs, Zs, max_iterations, epsilon)   
    
    for slope_init in np.linspace(min_w0,max_w0,num_trials):  # tries multiple initial values for the slope w0 --> avoid local minima in the optimization
        x0 = anp.zeros(num_params_bg + 1) # initial parameter vector: 1 for slope w0, 1 for the bias, and N for Gaussian weights (num_params_bg = N+1)
        x0[0] = slope_init     # x0 = [w0, ... ]
        x0[1:] = sol_bg.x      # x0 = [w0, b, alpha_1, ..., alpha_N]  -  use background fit: sol_bg.x[0] = b, sol_bg.x[1:] = [alpha_1, ..., alpha_N]
        # Fit the full compensation model
        res = fit(
            model_par,
            x0, 
            Xs.reshape(num_gate_vals * num_sensor_samples, -1),  # collapses the first two dimensions into one, shape (num_gate_vals, num_sensor_samples, 2) turns into shape (num_gate_vals * num_sensor_samples, 2) --> Array of shape (samples, features) - input for optimizer!
            lines.obs.reshape(-1),  # flattens the 2D array into a 1D vector, shape (num_gate_vals, num_sensor_samples) turns into shape (num_gate_vals * num_sensor_samples,)
            max_iterations,
            epsilon        
        )
        # Keep the best result (res.fun is equal to the loss)
        if (f_val := res.fun) < f_best_sol:   # Walrus operator := allows assignment within an expression
            res_best = res
            solution = res.x
            f_best_sol = f_val

    return Model(solution, model_par, mid_val_gate, res_best)   # return the best model


def compute_gate_compensation_ml(ramp, central_point, sensor_id, ranges, num_measurements = 6, N=200, max_w0=0.7, num_trials=2, max_iterations=1000000, epsilon=1.e-5):   ### DISCUSS: coordinate transformation here ???
    '''
    Compensation routine for a sensor based on the influence of other gates.
    This function computes a linear model that compensates for the effect of other gates on a particular sensor.

    Input parameters
        ramp: function that takes two endpoints and returns sensor data? -- 
              A ramp refers to a simulation/measurement fct that captures how the sensor respond when the system moves along a straight line ("ramp") in input space (specifically between two points)
              Used to measure/simulate how a sensor's reading changes as one or more gate parameters are varied from a starting point to and ending point.
        central_point: vector representing a central configuration point - chosen reference config that should remain invariant under compensation. Dimension: dim
        sensor_id: index for which sensor to compensate
        ranges: dictionary mapping gate/sensor IDs to their min-max range of values
        num_measurements, N, max_w0, num_trials: optional parameters for tuning

    Returns
        P: transformation matrix (dim,dim) that models how other gate values influence the sensor input
        m: offset vector (dim,) that ensures alignment at the central point
    that neutralize the effects of other gates. Linear model adjusted_input = P*original_input + m to correct the sensor's input before performing readout.
    '''

    dim = len(central_point)  # dimension (number of gates/sensors)

    #as reference, create the ramp coordinates for the sensor ramp around the
    #central point
    sensor_range = ranges[sensor_id]  # value range (min, max) for the sensor being compensated
    central_start = central_point.copy() # make two copies of the central point to form the endpoints of a ramp - modify only the sensor_id dim of the start and end points to span the sensor's full range
    central_end = central_point.copy()
    central_start[sensor_id] = sensor_range[0]
    central_end[sensor_id] = sensor_range[1]
    
    #final learned coordinate transform
    P = np.eye(dim)    # initialize matrix for linear compensation
    m = np.zeros(dim)  # initialize offset vector for compensation
    # loop over all gates
    for gate_id in ranges.keys():
        #skip the current sensor for compensation!!!
        if gate_id == sensor_id:
            continue
        #print('meas:', gate_id, num_measurements, N) 
        logger.info(f"Compensating sensor {sensor_id} for gate {gate_id} with {num_measurements} gate values/measurements and {N} samples per ramp (resolution).")
        #create values to probe for this gate
        minv,maxv = ranges[gate_id]  # min/max range of the current gate
         
        gate_values = np.linspace(minv,maxv,num_measurements)  # evenly spaced values across the range for probing  (shape: (num_measurements,), i.e. 1D NumPy Array)
        
        observations = []  # Collection of sensor readings
        # Sweep the gate across its values, i.e. a possible setting for this gate. We are probing this gate!
        # Straight-line ramp along the sensor dimension, for a specific gate value
        # Create a new ramp_start/ramp_end, where only gate "gate_id" is changed to val (both ramp_start and ramp_end) - all other gate values are the same as in central_start and central_end. I.e. only ONE component is modified!
        for val in gate_values:
            # E.g. np.eye(1,4) is [[1. 0. 0. 0.]]. np.eye(1,4,2) is [[0. 0. 1. 0.]], i.e. shifted 2 to the right. np.eye(1,4,2)[0] is [0. 0. 1. 0.], i.e. a 1D vector
            # Keep all coordinates of central_start the same, except gate "gate_id", which is set to val
            # delta_vector = np.eye(1,dim, gate_id)[0]*(val-central_start[gate_id])
            # ramp_start = central_start + delta_vector
            # ramp_end = central_end + delta_vector
            ramp_start = central_start + np.eye(1,dim, gate_id)[0]*(val-central_start[gate_id])  # equal to val for gate "gate_id", equal to central_start otherwise, sensor low
            # Keep all coordinates of central_end the same, except gate "gate_id", which is set to val
            ramp_end = central_end + np.eye(1,dim, gate_id)[0]*(val-central_start[gate_id])      # equal to val for gate "gate_id", equal to central_end otherwise, sensor high. At this point, central_start[gate_id] == central_end[gate_id] (they only differ in the sensor_id dimension).
            obs = ramp(ramp_start, ramp_end, N)   # Call the ramp function to compute the response between start and end
            obs = (obs - obs.mean()) / obs.std()  # Standardize (zero mean, unit var)
            observations.append(obs)
        # Evenly spaced points on the sensor range, i.e. sensor input values for the data collected via the ramp function - these span the sensor's range.
        inputs = np.linspace(sensor_range[0],sensor_range[1],len(observations[0]))  # shape: (N, ) 1D NumPy array of length N, inputs[None] adds a new axis -> shape: (1,N) 2D row vector
        
        #transform to jax object
        lines = Lines(
            np.repeat(inputs[None], len(gate_values), 0),  # 2D array, where each row is a copy of the sensor input values (inputs), one for each gate value. Shape: (num_measurements, N)
            gate_values,                                   # 1D array, values at which the current gate was probed. Shape: (num_measurements,)
            np.array(observations)                         # 2D array, actual sensor output for each gate value. Shape: (num_measurements, N)
        )
    
        #fit
        midv = (minv+maxv)/2  # midpoint of a gate's range
        model = fit_compensation_parameters(lines, N, midv, min_w0=0.0, max_w0=max_w0, num_trials=num_trials, max_iterations=max_iterations, epsilon=epsilon)  # fitting function to learn how the current gate influences the sensor
        #save compensation parameters
        logger.debug(f"model solution: {model.solution[0]}")
        P[sensor_id,gate_id] = -model.solution[0]  # learn slope: how does gate gate_id affect sensor sensor_id
        #compute offset such that the compensated model produces the gate values of central point
        #as we only change the coordinates of the sensor dot, only a change of the offset for the
        #sensor is needed
        #m[sensor_id] -= P[sensor_id,gate_id] * central_point[gate_id]  # offset needed so that the central point remains invariant after compensation -> compensation does NOT distort the sensor reading at the central point
    
    # Note that only one row of P is modified: P[sensor_id,:]. All other rows stay as identity.
    # Note that only one entry of m is modified: m[sensor_id]. All other entries stay zero.
    return P #, m


def align_linear(values, sensor_gate_vs, ys, N_init=20,w_min=-1.2,w_max=1.2,use_mse=False):
    """ 
    Find a linear compensation factor (a slope comp_param) that explains how much the sensor's effective axis shifts with gate changes.
    This is done by interpolating the measured data, testing different compensation factors, and optimizing with scipy.optimize.minimize.
    """
    
    def interpolate(points,img):
        results = []
        for i in range(points.shape[0]):
            p = interpn([sensor_gate_vs], img[i], points[i], method='linear', bounds_error=False, fill_value=1e100)
            results.append(p)
        return np.array(results)

    def transform_data(comp_param, lag):
        ys_diff = (ys[lag:]-ys[:-lag])
        comp_sens_vs = sensor_gate_vs[None,:]+comp_param*ys_diff[:,None]
        interp_data = interpolate(comp_sens_vs,values[lag:])
        return interp_data
    
    def objective(comp_param,lag=2):
        interp_data = transform_data(comp_param, lag)
        diff = (values[:-lag]-interp_data)
        mask = interp_data<100
        sample_error = np.sum(np.abs(mask*diff),axis=1)/np.sum(mask,axis=1)

        if use_mse:
            return np.mean(sample_error)
        else:
            return np.mean(sample_error**0.5)

    #optimize
    ws_init = np.linspace(w_min,w_max,N_init)
    dw = ws_init[1]-ws_init[0]
    w_init = ws_init[np.argmin([objective(w) for w in ws_init])]
    res = minimize(objective, w_init, method='Powell',bounds=((w_init-2*dw,w_init+2*dw),))
    comp_param = res.x[0]

    ys_diff = (ys-ys[0])
    comp_sens_vs = sensor_gate_vs[None,:]+comp_param*ys_diff[:,None]
    interp_data = interpolate(comp_sens_vs,values)
    interp_data[interp_data>100]=np.mean(values)

    return comp_param, interp_data


def amplitude_scan_2D(ramp, ids, central_point, ranges, gate_values, N):
    """
    Perform a 2D amplitude scan: sweep one gate and, for each gate value,
    perform a ramp along the sensor axis.

    Parameters:
    ramp: Function to perform the ramp measurement.
    ids (list[int]): [gate_id, sensor_id], the indices of the gate to probe
                     and the sensor to sweep.
    central_point (np.array): Central voltage vector for the scan.
    ranges: dictionary mapping gate/sensor IDs to their min/max amplitude value
    N (int): Number of points in the sensor ramp.

    Returns:
    np.ndarray: 2D array of shape (num_measurements, resolution)
                with sensor signal values.
                Axis 0 = gate sweep, Axis 1 = sensor sweep.
    """
    
    gate_id = ids[0]
    sensor_id = ids[1]
    ramp_start = central_point.copy() # make two copies of the central point to form the endpoints of a ramp - modify only the sensor_id dim of the start and end points to span the sensor's full range
    ramp_end = central_point.copy()
    ramp_start[sensor_id] = ranges[sensor_id][0]
    ramp_end[sensor_id] = ranges[sensor_id][1]

    observations = []
    for val in gate_values:
            # Update gate value in both ramp start and end
            ramp_start_ = ramp_start.copy()
            ramp_end_ = ramp_end.copy()
            ramp_start_[gate_id] = val
            ramp_end_[gate_id] = val

            # Call the ramp function to compute the response between start and end
            obs = ramp(ramp_start_, ramp_end_, N)
            #print(f'raw observations: {obs}')
            observations.append(obs)

    return np.array(observations)  # shape (num_measurements, resolution)


def compute_gate_compensation_al(ramp, central_point, sensor_id, ranges, num_measurements = 6, N=200, w_min=-2.0, w_max=0.2, sigma_gaussian=1.0, normalize_rows=True, plot=False):
    '''
    This function computes a linear model that compensates for the effect of other gates on a particular sensor.

    Input parameters
        ramp: function that takes two endpoints and returns sensor data? -- 
              A ramp refers to a simulation/measurement fct that captures how the sensor respond when the system moves along a straight line ("ramp") in input space (specifically between two points)
              Used to measure/simulate how a sensor's reading changes as one or more gate parameters are varied from a starting point to and ending point.
        central_point: vector representing a central configuration point - chosen reference config that should remain invariant under compensation. Dimension: dim
        sensor_id: index for which sensor to compensate
        ranges: dictionary mapping gate/sensor IDs to their min/max amplitude value
        num_measurements, N, max_w0, num_trials: optional parameters for tuning

    Returns
        P: transformation matrix (dim,dim) that models how other gate values influence the sensor input
        m: offset vector (dim,) that ensures alignment at the central point
    that neutralize the effects of other gates. Linear model adjusted_input = P*original_input + m to correct the sensor's input before performing readout.
    '''
    dim = len(central_point)  # dimension (number of gates/sensors)

    P = np.eye(dim)    # initialize matrix for linear compensation
    m = np.zeros(dim)  # initialize offset vector for compensation

    # Create sensor values for fitting
    sensor_values = np.linspace(ranges[sensor_id][0], ranges[sensor_id][1], N)

    # loop over all gates
    for gate_id in ranges.keys():

        #skip the current sensor for compensation!!!
        if gate_id == sensor_id:
            continue
        
        print('meas:', gate_id, num_measurements, N)
        
        # Compute relative gate values for fitting
        gate_values = np.linspace(ranges[gate_id][0], ranges[gate_id][1], num_measurements)

        # Perform 2D amplitude scan: sweep gate, for each gate value perform a ramp along the sensor axis
        observations = amplitude_scan_2D(ramp, [gate_id, sensor_id], central_point, ranges, gate_values, N)  # Shape: (num_measurements, N)

        # Apply Gaussian filter to smooth out noise
        observations = gaussian_filter1d(observations, sigma_gaussian, order=0,axis=1)  

        # Normalize observations to zero mean, unit variance
        if normalize_rows:  # row-wise normalization
            observations -= np.mean(observations, axis=1)[:,None]  # np.mean(..., axis=1) gives shape (num_measurements,); [:,None] makes it (num_measurements,1)
            observations /= np.std(observations, axis=1)[:,None]
        else:  # global normalization
            observations = (observations - np.mean(observations)) / np.std(observations)

        # Fit compensation model
        comp_factor, comp_obs = align_linear(observations, sensor_values, gate_values, w_min=w_min, w_max=w_max)
        print(f"Fitted compensation factor for sensor {sensor_id} wrt gate {gate_id}: {comp_factor:.4f}")
        P[sensor_id,gate_id] = -comp_factor
        #m[sensor_id] -= P[sensor_id,gate_id] * central_point[gate_id]
    
    return P #, m


def subtract_low_norm_mean(img, frac=0.7):
    """
    Subtract the mean of the pixels with smallest norm from the input image

    Input:
    img     Input image
    frac    Fraction of pixels to use for mean computation (i.e. with smallest norm)

    Return:
    Centered image      Input image minus mean of pixels with smallest norm
    """
    pixels = img.reshape(-1,1)
    norm = np.linalg.norm(pixels,axis=1)
    threshold = np.quantile(norm, frac)
    mask = norm <= threshold    # mask: the fraction of pixels used for the mean computation
    mean_value = pixels[mask].mean()
    return img - mean_value


def image_gradients(img, cfg):
    """
    Compute the gradients in x- and y-direction of an image using Sobel filters. 
    Apply Gaussian blurring before computing the gradients. Subtract the mean of pixels with small norm from the gradient images.

    Input:
    img                 Input image I of shape (H,W)
    cfg                 Config parameters
        sigmaX_blur         Standard deviation of the Gaussian in x direction used for blurring the image
        sigmaY_blur         Standard deviation of the Gaussian in y direction used for blurring the image
        ksize_sobelX        Kernel size for computing the gradients in x-direction. Note: Has to be ODD number.
        ksize_sobelY        Kernel size for computing the gradients in x-direction. Note: Has to be ODD number.
        frac                Fraction of pixels to use for mean computation

    Returns:
    sobel_x         Image Ix of gradients in x-direction of shape (H,W)
    sobel_y         Image Iy of gradients in y-direction of shape (H,W)
    """

    # Apply Gaussian blur to the image
    blurred_img = cv.GaussianBlur(img, (0, 0), cfg.sigmaX_blur, cfg.sigmaY_blur)   # (0,0) means that the kernel size is automatically computed based on sigmaX and sigmaY; sigmaX=sigmaY

    # Apply Sobel filter to compute gradients
    sobel_x = cv.Sobel(blurred_img, cv.CV_64F, 1, 0, ksize=cfg.ksize_sobelX)  # Output image type is np.float64
    sobel_y = cv.Sobel(blurred_img, cv.CV_64F, 0, 1, ksize=cfg.ksize_sobelY)

    # Subtract mean of pixels with small norm
    sobel_x = subtract_low_norm_mean(sobel_x, frac=cfg.frac)  
    sobel_y = subtract_low_norm_mean(sobel_y, frac=cfg.frac)

    return sobel_x, sobel_y


def generate_2D_gradient_vector(img, cfg):
    """
    Generates a dataset of 2D gradient vectors from the input image using Sobel filters. Only one channel is considered.

    Input:
    img             Input image I of shape (H,W)
    cfg             Config parameters for computing the image gradients

    Returns:
    points          A 2D array where each row is a gradient vector [Gx, Gy] for each pixel of the input image, i.e. rows are observations (pixels), columns are variables (sobel_x, sobel_y).
    """
    print("Generating 2D gradient vectors using Sobel filters")
    
    # Convert the image to uint8, i.e. range [0,255] (standard OpenCV format)
    if img.dtype != np.uint8:
        img = (255 * (img - img.min()) / (img.max() - img.min())).astype(np.uint8)

    # Compute gradients
    sobel_x, sobel_y = image_gradients(img, cfg)

    # Generate 2D gradient vector
    points = np.stack([sobel_x,sobel_y],axis=-1).reshape(-1,2)  # stack along the last axis -> shape (H,W,2), then reshape to (H*W,2), i.e. rows are observations (pixels), columns are variables (sobel_x, sobel_y)

    return points


def scale_data(points, scale):
    print(f"Scaling data using method: {scale}")
    if scale == "overall":
        data = points/anp.std(points)
        data_std = {'x': 1, 'y': 1}
    elif scale == "per-dimension":
        data = points.copy()
        std_x = anp.std(data[:,0])
        std_y = anp.std(data[:,1])
        data[:,0] = data[:,0]/std_x
        data[:,1] = data[:,1]/std_y
        data_std = {'x': std_x, 'y': std_y}

    return data, data_std


def make_covariance_matrices(params):
    # Given parameters: p1x, p1y, p2x, p2y, tau, return covariance matrices for the Gaussian components
    p1 = params[:2]  
    p2 = params[2:4] 
    sigma = anp.exp(params[4]) + 1.e-6*anp.linalg.norm(p1) + 1.e-6*anp.linalg.norm(p2)  # positive!
    I = anp.eye(2)   

    Sigma0 = sigma**2 * I
    Sigma1 = anp.outer(p1, p1) + sigma**2 * I  # anp.outer computes the outer product: 2D matrix here
    Sigma2 = anp.outer(p2, p2) + sigma**2 * I

    return Sigma0, Sigma1, Sigma2
    

def normal_distribution_2D_vectorized(X, cov):
    # Multivariate normal distribution PDF for 2D data with zero mean.
    # X: array of shape (N,2) where each row is a 2D data point
    # cov: 2x2 covariance matrix
    det = anp.linalg.det(cov)
    inv_cov = anp.linalg.inv(cov)
    quad_form = anp.sum((X @ inv_cov) * X, axis=1)
    pdf = 1 / (2 * anp.pi * anp.sqrt(det)) * anp.exp(-0.5 * quad_form)

    return pdf.reshape(-1,1) # Otherwise, shape (N,). Alternatively, use [:,None] to convert to column vector with shape (N,1)


# f: function to minimize, i.e. negative log-likelihood of the Gaussian Mixture Model
def gmm_log_likelihood(params,data,w):
    # Compute the negative log-likelihood of the Gaussian Mixture Model
    Sigma0, Sigma1, Sigma2 = make_covariance_matrices(params)
    likelihood = (w[0] * normal_distribution_2D_vectorized(data, Sigma0) +
                  w[1] * normal_distribution_2D_vectorized(data, Sigma1) +
                  w[2] * normal_distribution_2D_vectorized(data, Sigma2))
    log_likelihood = anp.sum(anp.log(likelihood + 1e-10))  # Add small constant to avoid log(0)

    return -log_likelihood


def gmm_log_likelihood_reg(params,data,w,reg_param):
    # Compute the negative log-likelihood of the Gaussian Mixture Model
    p1 = params[0:2]  
    p2 = params[2:4]
    Sigma0, Sigma1, Sigma2 = make_covariance_matrices(params)
    likelihood = (w[0] * normal_distribution_2D_vectorized(data, Sigma0) +
                  w[1] * normal_distribution_2D_vectorized(data, Sigma1) +
                  w[2] * normal_distribution_2D_vectorized(data, Sigma2))
    log_likelihood = anp.sum(anp.log(likelihood + 1e-10))  # Add small constant to avoid log(0)
    reg = reg_param * (p2[0]**2/(anp.linalg.norm(p2))**2 + p1[1]**2/(anp.linalg.norm(p1))**2)  # ensure that the normal p1 is aligned with y-axis, and the normal p2 with x-axis

    return -log_likelihood + reg


def compute_transformation_matrix(n1,n2):
    # Build transformation
    B = np.column_stack([n1, n2])  # columns are normals
    B /=np.diag(B)[None,:] # divide each column by its diagonal element; np.diag(B)[None,:] is the row vector [n11,n22] of shape (1,2)
    A = B.T                      # transformation: original coordinates -> new coordinates
    A_inv = np.linalg.inv(A)     # transformation: new -> original
    A_inv = A_inv @ np.diag(np.diag(A_inv)**(-1))
    A = np.linalg.inv(A_inv)
    return A_inv, A


def warp_image_with_normals(img, n1, n2, fill_value=1e-6):
    """
    Warp an image defined on a rectangular grid (xs, ys) so that lines with normals n1, n2
    become orthogonal in the transformed coordinates. Plot the warped image.

    Input:    
    img             Input image (2D xarray) of shape (H,W)
    n1, n2          Normalized line normals: (2,) arrays
    fill_value      Value for points outside the input domain
    """
    H, W = img.shape    # height, width of image
    xs = np.arange(W)   # original x-grid (cols)
    ys = np.arange(H)   # original y-grid (rows)

    A_inv, A = compute_transformation_matrix(n1,n2)

    # Output grid
    nx = W  # output image size as the original image
    ny = H

    XX, YY = np.meshgrid(xs, ys, indexing="xy")      # original grid, 2D arrays of all x and y coordinates (Cartesian)
    pts = np.column_stack([XX.ravel(), YY.ravel()])  # list of pixel coordinates (points), shape (H*W,2), i.e. one row per pixel
    new_pts = (A @ pts.T).T                          # transformed coordinates of all original pixels, shape (H*W,2), i.e. one row per pixel

    x_min, x_max = new_pts[:,0].min(), new_pts[:,0].max()   # determine bounding box in transformed space (both in x- and y-direction) --> how big the new image grid needs to be!
    y_min, y_max = new_pts[:,1].min(), new_pts[:,1].max()
    new_xs = np.linspace(x_min, x_max, nx)   # define a rectangular grid in transformed space
    new_ys = np.linspace(y_min, y_max, ny)
    grid_x, grid_y = np.meshgrid(new_xs, new_ys, indexing="xy")  # grid for the output image (covers the whole transformed image)

    # Map new coords back to original coords
    Yq = np.column_stack([grid_x.ravel(), grid_y.ravel()]) # list of output pixel coordinates (points)
    Xq = (A_inv @ Yq.T).T                                  # output pixel coordinates mapped back to original image coordinates --> need to be able to interpolate the image values at the new positions

    # Interpolate at these mapped coordinates Xq
    warped = interpn(
        points=(xs, ys), # original coordinates (xs, ys), pixels
        values=img.T,    # image values at these coordinates (first index is x, second index is y)
        xi=Xq,           # mapped coordinates, where we want to know the values
        method="linear",
        bounds_error=False,
        fill_value=fill_value,
    ).reshape(ny, nx)  # converts it back to a 2D image

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(img, origin="lower")
    axs[0].set_title("Original")
    axs[1].imshow(warped, origin="lower", extent=(new_xs.min(), new_xs.max(), new_ys.min(), new_ys.max()))
    axs[1].set_title("Orthogonalized")
    fig.suptitle(f"Normals: n1 = {np.array2string(n1, precision=4)}, n2 = {np.array2string(n2, precision=4)}")
    plt.show()


def compute_transformation_matrix_from_image_gradients(img, cfg):

    # Turn tuples into arrays (they do not change values during optimization)
    w = np.array(cfg.model.w).reshape(-1, 1)  # values have shape (3,1) for broadcasting
    init_params = np.array(cfg.model.init_params, dtype=float)

    # Opencv assumes that the origin is at upper left. Apply vertical flip to the image to convert origin at lower left to origin at upper left.
    img = np.flipud(img)

    # Image gradients
    data = generate_2D_gradient_vector(img, cfg.gradient)

    # Scale the image gradients
    data, data_std = scale_data(data, cfg.model.scale)

    # Data fitting (GMM on gradients)
    logger.info(f"Optimizing GMM")            
    if cfg.model.likelihood == "without-reg":
        logger.info(f"Initial log-likelihood: {gmm_log_likelihood(init_params,data,w)}")
        problem = value_and_grad(lambda params: gmm_log_likelihood(params,data,w))
    if cfg.model.likelihood == "with-reg":
        logger.info(f"Initial log-likelihood: {gmm_log_likelihood_reg(init_params,data,w,cfg.model.reg_param)}")
        problem = value_and_grad(lambda params: gmm_log_likelihood_reg(params,data,w,cfg.model.reg_param))
    result = minimize(problem, init_params, method="L-BFGS-B", jac=True, tol=0.0, options={'maxiter':cfg.optimization.max_iterations, 'gtol': cfg.optimization.epsilon})

    p1 = result.x[:2]  # normal aligned with x-axis
    p2 = result.x[2:4] # normal aligned with y-axis
    # Ensure that the normals point in positive x- and y-direction (in the origin at lower left coordinate system)       NOTE: THIS ONLY WORKS IF P1 ALIGNED WITH X-AXIS, AND P2 WITH Y-AXIS!!!
    if np.abs(p1[0]) > np.abs(p1[1]) and p1[0] < 0: # check that p1 is aligned with x-axis, and ensure that it points in the positive x-direction     
        p2 = -p2
    if np.abs(p2[1]) > np.abs(p2[0]) and p2[1] > 0: # check that p2 is aligned with y-axis, and ensure that it points in the negative y-direction (flipped vertically afterwards!)
        p2 = -p2
    p1_rescaled = np.array([p1[0]*data_std['x'],p1[1]*data_std['y']])
    p2_rescaled = np.array([p2[0]*data_std['x'],p2[1]*data_std['y']])
    p1_rescaled = p1_rescaled / np.linalg.norm(p1_rescaled)
    p2_rescaled = p2_rescaled / np.linalg.norm(p2_rescaled)
    m1 = -p1[0]*data_std['x']/(p1[1]*data_std['y'])
    m2 = -p2[0]*data_std['x']/(p2[1]*data_std['y'])

    # Convert results back to origin at lower left, i.e. apply vertical flip.
    p1[1] = -p1[1]
    p2[1] = -p2[1]
    p1_rescaled[1] = -p1_rescaled[1]
    p2_rescaled[1] = -p2_rescaled[1]
    m1 = -m1
    m2 = -m2    

    return p1_rescaled, p2_rescaled, m1, m2


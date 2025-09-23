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
    "calculate_normals",
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


def calculate_normals(annotations_data: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, np.ndarray]:
    """
    Calculate lines of annotated lines.

    Args:
    annotations_data    The main annotations data structure.
            Expected: {"points": [...], "lines": [...]}

    Returns:
    normals             A dictionary mapping line ID (string) to its normal (np.array).
    """
    normals: Dict[str, np.ndarray] = {}
    lines = annotations_data.get("lines", [])

    for line in lines:
        line_id = line["id"]
        coords1 = get_point_coords_by_id(annotations_data, line["start_point_id"])
        coords2 = get_point_coords_by_id(annotations_data, line["end_point_id"])

        x1, y1 = coords1
        x2, y2 = coords2
        delta_x = x2 - x1
        delta_y = y2 - y1
        n = np.array([-delta_y,delta_x], dtype=float)
        n = n/np.linalg.norm(n)
        normals[line_id] = n

    logger.info(f"Calculated normals for {len(normals)} lines.")
    return normals


def subtract_low_norm_mean(img, frac=0.7):
    """
    Subtract the mean of the pixels with smallest norm from the input image

    Args:
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

    Args:
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

    Args:
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
    """
    Scale the 2D data points. Either with the overall standard deviation (scale=="overall") or separately for each dimension (scale=="per-dimension").

    Input:
    points      2D data points

    Args:
    data        Scaled data points
    data_std    Scaling factor for each dimension
    """
    print(f"Scaling data using method: {scale}")
    if scale == "overall":
        std_x = std_y = anp.std(points)
    elif scale == "per-dimension":
        std_x = anp.std(points[:,0])
        std_y = anp.std(points[:,1])
    data = points.copy()    
    data[:,0] = data[:,0]/std_x
    data[:,1] = data[:,1]/std_y
    data_std = {'x': std_x, 'y': std_y}

    return data, data_std


def make_covariance_matrices(params):
    """
    Compute covariance matrices

    Input:
    params                      Given parameters: p1x, p1y, p2x, p2y, tau

    Args:
    Sigma0, Sigma1, Sigma2      Covariance matrices for the Gaussian components
    """
    p1 = params[:2]  
    p2 = params[2:4] 
    sigma = anp.exp(params[4]) + 1.e-6*anp.linalg.norm(p1) + 1.e-6*anp.linalg.norm(p2)  # positive!
    I = anp.eye(2)   

    Sigma0 = sigma**2 * I
    Sigma1 = anp.outer(p1, p1) + sigma**2 * I  # anp.outer computes the outer product: 2D matrix here
    Sigma2 = anp.outer(p2, p2) + sigma**2 * I

    return Sigma0, Sigma1, Sigma2
    

def normal_distribution_2D_vectorized(X, cov):
    """
    Multivariate normal distribution PDF for 2D data with zero mean.

    Input:
    X        array of shape (N,2) where each row is a 2D data point
    cov      2x2 covariance matrix

    Args:
    pdf     array of shape (N,1)
    """
    det = anp.linalg.det(cov)
    inv_cov = anp.linalg.inv(cov)
    quad_form = anp.sum((X @ inv_cov) * X, axis=1)
    pdf = 1 / (2 * anp.pi * anp.sqrt(det)) * anp.exp(-0.5 * quad_form)

    return pdf.reshape(-1,1) # Otherwise, shape (N,). Alternatively, use [:,None] to convert to column vector with shape (N,1)


# f: function to minimize, i.e. negative log-likelihood of the Gaussian Mixture Model
def gmm_log_likelihood(params,data,w):
    """
    Compute the negative log-likelihood of the Gaussian Mixture Model.
    This function is used to compute the normals of two lines by fitting a Gaussian Mixture Model on image gradients in the function compute_transformation_matrix_from_image_gradients().
    """
    Sigma0, Sigma1, Sigma2 = make_covariance_matrices(params)
    likelihood = (w[0] * normal_distribution_2D_vectorized(data, Sigma0) +
                  w[1] * normal_distribution_2D_vectorized(data, Sigma1) +
                  w[2] * normal_distribution_2D_vectorized(data, Sigma2))
    log_likelihood = anp.sum(anp.log(likelihood + 1e-10))  # Add small constant to avoid log(0)

    return -log_likelihood


def gmm_log_likelihood_reg(params,data,w,reg_param):
    """
    Compute the negative log-likelihood of the Gaussian Mixture Model and adds a regularization term to ensure that the normals are aligned with the correct axes.
    This function is used to compute the normals of two lines by fitting a Gaussian Mixture Model on image gradients in the function compute_transformation_matrix_from_image_gradients().
    """
    p1 = params[0:2]  
    p2 = params[2:4]
    Sigma0, Sigma1, Sigma2 = make_covariance_matrices(params)
    likelihood = (w[0] * normal_distribution_2D_vectorized(data, Sigma0) +
                  w[1] * normal_distribution_2D_vectorized(data, Sigma1) +
                  w[2] * normal_distribution_2D_vectorized(data, Sigma2))
    log_likelihood = anp.sum(anp.log(likelihood + 1e-10))  # Add small constant to avoid log(0)
    reg = reg_param * (p2[0]**2/(anp.linalg.norm(p2))**2 + p1[1]**2/(anp.linalg.norm(p1))**2)  # ensure that the normal p1 is aligned with x-axis, and the normal p2 with y-axis

    return -log_likelihood + reg


def compute_transformation_matrix(n1,n2):
    """
    Compute the transformation matrix between two coordinate systems. 
    In the original coordinate system, there are two non-orthogonal lines with the normals n1 and n2.
    In the new coordinate system, these lines are orthogonal.

    Args:
    n1, n2      The normals of the (non-orthogonal) lines.

    Returns:
    A_inv       Transformation matrix: new --> original coordinate system, diagonal elements are 1
    A           Transformation matrix: original --> new coordinate system, diagonal elements are 1
    """
    B = np.column_stack([n1, n2])   # columns are normals
    B /=np.diag(B)[None,:]          # divide each column by its diagonal element; np.diag(B)[None,:] is the row vector [n11,n22] of shape (1,2)
    A = B.T                         # transformation: original coordinates -> new coordinates (orthogonal lines)
    A_inv = np.linalg.inv(A)        # transformation: new -> original
    A_inv = A_inv @ np.diag(np.diag(A_inv)**(-1))
    A = np.linalg.inv(A_inv)
    return A_inv, A


def warp_image_with_normals(img, n1, n2, fill_value=1e-6):
    """
    Warp an image defined on a rectangular grid (xs, ys) so that lines with normals n1, n2
    become orthogonal in the transformed coordinates. Plot the warped image.

    Args:    
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
    """
    This function computes the normals p1, p2 and slopes m1, m2 of lines with 2 distinct directions in the input image.
    This is done by fitting a Gaussian Mixture Model on the image gradients in x- and y-direction.
    The unknown parameters are given by θ = [p1_x, p1_y, p2_x, p2_y, τ] with σ_ε = exp(τ) > 0.
    The log-likelihood function is given by L(θ) = Σ_(i=1)^N log(Σ_(k=0)^2 w_k N(v_i | 0,Σ_k(θ))) with Σ_0=σ_ε^2*I, Σ_k=p_k*p_k^T + σ_ε^2*I for k=1,2, w is fixed.

    Args:
    img     Input image (xarray)
    cfg     Config parameters (for computing the image gradients, for the model, and for the optimization)

    Returns:
    p1, p2  Normalized normals (p1 aligned with x-axis, p2 aligned with y-axis)
    m1, m2  Slopes of the lines with 2 distinct directions (abs(m1) > abs(m2))
    """
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
    p1_rescaled = np.array([p1[0]*data_std['x'],p1[1]*data_std['y']])   # Scale back
    p2_rescaled = np.array([p2[0]*data_std['x'],p2[1]*data_std['y']])
    p1_rescaled = p1_rescaled / np.linalg.norm(p1_rescaled)             # Normalize
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


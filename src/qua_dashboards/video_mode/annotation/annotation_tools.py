"""
Helper functions for the AnnotationComponent.

Includes functions for drawing annotations on figures, calculating distances,
loading/saving annotation files, and performing basic analysis like slope calculation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

__all__ = [
    "generate_figure_with_annotations",
    "get_point_coords_by_index",
    "find_closest_point_id",
    "find_closest_line_index",
    "list_annotation_files",
    "load_annotation_file",
    "save_annotation_file",
]

# --- Figure Generation ---


def generate_figure_with_annotations(
    base_figure_dict: Dict, points_state: Dict, lines_state: Dict
) -> go.Figure:
    """
    Generates a Plotly figure by overlaying annotations on a base figure.

    Args:
        base_figure_dict: The dictionary representation of the base Plotly figure
                          (e.g., the snapshot heatmap). Can be empty.
        points_state: Dictionary containing annotation points data. Expected keys:
                      'added_points': {'x': [], 'y': [], 'index': []},
                      'selected_point': {'move': bool, 'index': int|None}
        lines_state: Dictionary containing annotation lines data. Expected keys:
                     'selected_indices': [],
                     'added_lines': {'start_index': [], 'end_index': []}

    Returns:
        A Plotly Figure object with annotations overlaid.
    """
    # Create figure from dict, handle empty dict case
    if base_figure_dict:
        fig = go.Figure(base_figure_dict)
    else:
        fig = go.Figure()  # Start with empty if no base provided

    points = points_state.get("added_points", {"x": [], "y": [], "index": []})
    lines = lines_state.get("added_lines", {"start_index": [], "end_index": []})
    selected_point_to_move = points_state.get(
        "selected_point", {"move": False, "index": None}
    )
    selected_points_for_line = set(lines_state.get("selected_indices", []))

    # Prepare line coordinates
    line_x, line_y = [], []
    point_x_coords = points.get("x", [])
    point_y_coords = points.get("y", [])
    if point_x_coords:  # Check if there are points to draw lines between
        start_indices = lines.get("start_index", [])
        end_indices = lines.get("end_index", [])
        max_point_list_idx = len(point_x_coords) - 1
        for start_idx, end_idx in zip(start_indices, end_indices):
            # Convert point index (from points['index']) to list index
            try:
                start_list_idx = points["index"].index(start_idx)
                end_list_idx = points["index"].index(end_idx)

                if (
                    0 <= start_list_idx <= max_point_list_idx
                    and 0 <= end_list_idx <= max_point_list_idx
                ):
                    line_x.extend(
                        [
                            point_x_coords[start_list_idx],
                            point_x_coords[end_list_idx],
                            None,
                        ]
                    )
                    line_y.extend(
                        [
                            point_y_coords[start_list_idx],
                            point_y_coords[end_list_idx],
                            None,
                        ]
                    )
                else:
                    logger.warning(
                        f"Invalid line list index derived: start={start_list_idx}, end={end_list_idx}"
                    )
            except (ValueError, IndexError):
                logger.warning(
                    f"Could not find list index for point indices {start_idx} or {end_idx}"
                )

    # Generate marker sizes (highlight selected points)
    marked_index = selected_point_to_move.get("index")  # This is the point's unique ID
    point_indices = points.get("index", [])  # List of unique point IDs
    sizes = [
        13 if i == marked_index or i in selected_points_for_line else 10
        for i in point_indices
    ]
    # Point labels are index + 1
    texts = [str(int(i) + 1) for i in point_indices]

    # Remove any existing annotation traces before adding new ones
    # This prevents traces from piling up if figure dict is reused incorrectly
    traces_to_keep = []
    if fig.data:
        for trace in fig.data:
            if trace.name not in ["annotations_points", "annotations_lines"]:
                traces_to_keep.append(trace)
    fig.data = traces_to_keep

    # Add Points Trace (only if points exist)
    if point_x_coords:
        fig.add_trace(
            go.Scatter(
                x=point_x_coords,
                y=point_y_coords,
                mode="markers+text",
                marker=dict(
                    color="white",
                    size=sizes,
                    line=dict(color="black", width=1),
                    opacity=1.0,
                ),
                text=texts,
                textposition="top center",
                textfont=dict(color="white"),
                showlegend=False,
                name="annotations_points",  # Use a consistent name
                customdata=point_indices,  # Store unique point index in customdata
                meta={"layer": "above"},  # Attempt to draw above heatmap
            )
        )

    # Add Lines Trace (only if lines exist)
    if line_x:
        fig.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                line=dict(color="white", width=2),
                showlegend=False,
                name="annotations_lines",  # Use a consistent name
                meta={"layer": "above"},  # Attempt to draw above heatmap
            )
        )

    # Ensure clickmode is set for interactions
    fig.update_layout(
        clickmode="event+select", uirevision=np.random.rand()
    )  # uirevision helps preserve zoom

    return fig


# --- Interaction Helpers ---


def get_point_coords_by_index(
    points_data: Dict, index_id: int
) -> Optional[Tuple[float, float]]:
    """Gets coordinates of a point given its unique ID."""
    try:
        # Find the position in the list corresponding to the unique ID
        list_index = points_data.get("index", []).index(index_id)
        return points_data["x"][list_index], points_data["y"][list_index]
    except (ValueError, IndexError):
        logger.warning(f"Point with index ID {index_id} not found in points_data.")
        return None


def find_closest_point_id(
    x_click: float, y_click: float, points_data: Dict, tolerance: float
) -> Optional[int]:
    """
    Finds the unique ID of the closest annotation point within a tolerance.

    Args:
        x_click: X-coordinate of the click event.
        y_click: Y-coordinate of the click event.
        points_data: Dictionary with 'x', 'y', and 'index' lists.
        tolerance: The maximum distance to consider a point "close".

    Returns:
        The unique ID (from points_data['index']) of the closest point, or None.
    """
    min_dist_sq = tolerance**2
    closest_point_id = None
    point_indices = points_data.get("index", [])
    point_x = points_data.get("x", [])
    point_y = points_data.get("y", [])

    # Iterate using the unique IDs
    for point_id, px, py in zip(point_indices, point_x, point_y):
        dist_sq = (x_click - px) ** 2 + (y_click - py) ** 2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest_point_id = point_id
    return closest_point_id


def find_closest_line_index(
    x_click: float,
    y_click: float,
    points_data: Dict,
    lines_data: Dict,
    tolerance: float,
) -> Optional[int]:
    """
    Finds the index (position) of the closest annotation line within a tolerance.

    Args:
        x_click: X-coordinate of the click event.
        y_click: Y-coordinate of the click event.
        points_data: Dictionary with point data (needed for coordinates).
        lines_data: Dictionary with 'start_index', 'end_index' lists (using point IDs).
        tolerance: The maximum distance to consider a line "close".

    Returns:
        The positional index in the lines_data lists of the closest line, or None.
    """
    min_dist = tolerance
    closest_line_pos_idx = None

    start_indices = lines_data.get("start_index", [])
    end_indices = lines_data.get("end_index", [])

    for i, (start_id, end_id) in enumerate(zip(start_indices, end_indices)):
        # Get coordinates using the point IDs
        coords1 = get_point_coords_by_index(points_data, start_id)
        coords2 = get_point_coords_by_index(points_data, end_id)

        if coords1 is None or coords2 is None:
            logger.warning(
                f"Skipping line {i} due to invalid point IDs {start_id} or {end_id}"
            )
            continue  # Skip if point IDs are invalid

        x1, y1 = coords1
        x2, y2 = coords2

        # Calculate distance from point (x_click, y_click) to line segment (x1,y1)-(x2,y2)
        dx, dy = x2 - x1, y2 - y1
        d_sq = dx**2 + dy**2

        if d_sq == 0:  # Line segment is just a point
            dist = np.sqrt((x_click - x1) ** 2 + (y_click - y1) ** 2)
        else:
            # Project click point onto the infinite line containing the segment
            t = ((x_click - x1) * dx + (y_click - y1) * dy) / d_sq
            # Clamp t to [0, 1] to find the closest point on the segment
            t = max(0, min(1, t))
            proj_x, proj_y = x1 + t * dx, y1 + t * dy
            dist = np.sqrt((x_click - proj_x) ** 2 + (y_click - proj_y) ** 2)

        if dist < min_dist:
            min_dist = dist
            closest_line_pos_idx = i  # Store the positional index

    return closest_line_pos_idx


# --- File I/O ---


def list_annotation_files(directory: Path) -> List[Dict[str, str]]:
    """Lists JSON files in a directory for use in a Dropdown."""
    options = []
    try:
        if directory.is_dir():
            # Sort by name for consistency
            for f in sorted(directory.glob("*.json")):
                options.append({"label": f.name, "value": str(f.resolve())})
    except Exception as e:
        logger.error(f"Error listing annotation files in {directory}: {e}")
    return options


def load_annotation_file(file_path_str: str) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Loads and validates annotation data from a JSON file path string."""
    if not file_path_str:
        return None, None
    file_path = Path(file_path_str)
    if not file_path.exists() or not file_path.is_file():
        logger.error(f"Annotation file not found: {file_path}")
        return None, None

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in file: {file_path}")
        return None, None
    except Exception as e:
        logger.error(f"Failed to read annotation file {file_path}: {e}")
        return None, None

    # Basic validation
    if not isinstance(data, dict) or "points" not in data or "lines" not in data:
        logger.error(f"Invalid structure in annotation file: {file_path}")
        return None, None
    if (
        not isinstance(data["points"], dict)
        or "x" not in data["points"]
        or "y" not in data["points"]
    ):
        logger.error(f"Invalid 'points' structure in annotation file: {file_path}")
        return None, None
    if (
        not isinstance(data["lines"], dict)
        or "start_index" not in data["lines"]
        or "end_index" not in data["lines"]
    ):
        logger.error(f"Invalid 'lines' structure in annotation file: {file_path}")
        return None, None

    points_data = data["points"]
    lines_data = data["lines"]

    # --- Data Integrity Checks ---
    num_points = len(points_data.get("x", []))
    if len(points_data.get("y", [])) != num_points:
        logger.error(f"X/Y point count mismatch in {file_path}")
        return None, None

    start_indices = lines_data.get("start_index", [])
    end_indices = lines_data.get("end_index", [])
    if len(start_indices) != len(end_indices):
        logger.error(f"Start/End index count mismatch in {file_path}")
        return None, None

    # Add unique point IDs (indices 0 to N-1) if they don't exist
    if "index" not in points_data or len(points_data["index"]) != num_points:
        logger.warning(f"Generating missing 'index' field for points in {file_path}")
        points_data["index"] = list(range(num_points))

    point_ids = set(points_data["index"])
    # Check if line indices refer to valid point IDs
    invalid_indices = [
        idx for idx in start_indices + end_indices if idx not in point_ids
    ]
    if invalid_indices:
        logger.error(
            f"Line indices refer to non-existent point IDs in {file_path}: {invalid_indices}"
        )
        return None, None

    logger.info(f"Successfully loaded annotations from: {file_path}")
    return points_data, lines_data


def save_annotation_file(
    directory: Path, points_data: Dict, lines_data: Dict, prefix: str = "annotation"
) -> Optional[Path]:
    """Saves annotation data to a JSON file with an incrementing index."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        idx = 1
        # Find the next available index
        while idx <= 9999:
            filename = f"{prefix}_{idx:04d}.json"
            filepath = directory / filename
            if not filepath.exists():
                break
            idx += 1
        else:
            logger.error("Maximum number of annotation files reached (9999).")
            return None

        # Prepare data for saving (only save necessary fields: x, y for points; start/end for lines)
        points_to_save = {
            "x": points_data.get("x", []),
            "y": points_data.get("y", []),
            # Do not save 'index' field by default, can be regenerated on load
        }
        lines_to_save = {
            "start_index": lines_data.get("start_index", []),  # These are the point IDs
            "end_index": lines_data.get("end_index", []),  # These are the point IDs
        }
        data_to_save = {"points": points_to_save, "lines": lines_to_save}

        with open(filepath, "w") as f:
            json.dump(data_to_save, f, indent=4)
        logger.info(f"Annotation data saved successfully: {filepath}")
        return filepath
    except Exception as e:
        logger.exception(f"Failed to save annotation data to {directory}: {e}")
        return None


# --- Analysis ---


def calculate_slopes(points_data: Dict, lines_data: Dict) -> Dict[int, float]:
    """
    Calculates slopes for annotated lines.

    Args:
        points_data: Dictionary with point coordinates and unique IDs ('index').
        lines_data: Dictionary with line start/end point IDs ('start_index', 'end_index').

    Returns:
        A dictionary mapping line index (positional index, 0 to M-1) to its slope.
        Returns float('inf') for vertical lines and float('nan') if points are invalid.
    """
    slopes: Dict[int, float] = {}  # Explicit type hint
    start_indices = lines_data.get("start_index", [])
    end_indices = lines_data.get("end_index", [])

    for i, (start_id, end_id) in enumerate(zip(start_indices, end_indices)):
        coords1 = get_point_coords_by_index(points_data, start_id)
        coords2 = get_point_coords_by_index(points_data, end_id)

        if coords1 and coords2:
            x1, y1 = coords1
            x2, y2 = coords2
            delta_x = x2 - x1
            delta_y = y2 - y1
            if np.isclose(delta_x, 0):  # Use numpy.isclose for float comparison
                # Vertical line: slope is infinity (or -infinity depending on direction)
                slopes[i] = (
                    float("inf")
                    if delta_y > 0
                    else float("-inf")
                    if delta_y < 0
                    else float("nan")
                )  # Undefined if also dy=0
            else:
                slopes[i] = delta_y / delta_x
        else:
            slopes[i] = float("nan")  # Invalid point IDs for this line

    logger.info(f"Calculated slopes for {len(slopes)} lines.")
    return slopes

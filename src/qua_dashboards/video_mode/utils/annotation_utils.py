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

logger = logging.getLogger(__name__)

__all__ = [
    "generate_annotation_traces",
    "get_point_coords_by_id",
    "find_closest_point_id",
    "find_closest_line_id",
    "calculate_slopes",
]


def generate_annotation_traces(
    annotations_data: Dict[str, List[Dict[str, Any]]],
    # The following are for highlighting, passed from AnnotationTabController's transient state
    selected_point_to_move_id: Optional[str] = None,
    selected_indices_for_line: Optional[List[str]] = None,
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

    # Ensure selected_indices_for_line is a list for consistent checking
    _selected_indices_for_line = selected_indices_for_line or []

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
        is_selected_for_line = p_id in _selected_indices_for_line

        if is_selected_for_move or is_selected_for_line:
            sizes.append(13)
        else:
            sizes.append(10)
        # Example color logic (can be expanded)
        marker_colors.append(
            "rgba(255, 255, 255, 0.8)"
            if not (is_selected_for_move or is_selected_for_line)
            else "rgba(255, 0, 0, 0.9)"
        )

    # Point labels (e.g., "P1", "P2" based on order or a specific label property)
    texts = [p_id for p_id in point_ids]

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
        meta={"layer": "above"},  # Ensure drawn on top
    ).to_plotly_json()
    traces.append(points_trace)

    if line_x_coords:
        lines_trace = go.Scatter(
            x=line_x_coords,
            y=line_y_coords,
            mode="lines",
            line=dict(
                color="rgba(255, 255, 255, 0.7)", width=2
            ),  # Slightly transparent white
            hoverinfo="none",
            name="annotations_lines",  # Consistent name
            meta={"layer": "above"},
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
    min_dist = tolerance
    closest_l_id: Optional[str] = None

    for line in annotations_data.get("lines", []):
        coords1 = get_point_coords_by_id(annotations_data, line["start_point_id"])
        coords2 = get_point_coords_by_id(annotations_data, line["end_point_id"])

        if coords1 is None or coords2 is None:
            logger.warning(
                f"Skipping line {line['id']} due to missing point coordinates."
            )
            continue

        x1, y1 = coords1
        x2, y2 = coords2

        dx, dy = x2 - x1, y2 - y1
        d_sq = dx**2 + dy**2

        if np.isclose(d_sq, 0):  # Line is a point
            dist = np.sqrt((x_click - x1) ** 2 + (y_click - y1) ** 2)
        else:
            # Project click point onto the line segment
            t = ((x_click - x1) * dx + (y_click - y1) * dy) / d_sq
            t = max(0.0, min(1.0, t))  # Clamp t to the segment
            proj_x, proj_y = x1 + t * dx, y1 + t * dy
            dist = np.sqrt((x_click - proj_x) ** 2 + (y_click - proj_y) ** 2)

        if dist < min_dist:
            min_dist = dist
            closest_l_id = line["id"]

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

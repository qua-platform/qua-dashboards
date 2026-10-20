import xarray as xr
import plotly.graph_objects as go
import numpy as np
from qua_dashboards.video_mode.utils.dash_utils import xarray_to_plotly
from plotly.subplots import make_subplots
from math import ceil

import math


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _make_subplot_boxes_square(
    fig,
    rows: int,
    cols: int,
    left: float = 0.06,
    right: float = 0.08,
    top: float = 0.08,
    bottom: float = 0.10,
    hspace: float = 0.04,
    vspace: float = 0.08,
):
    """
    Make every subplot box square in paper coordinates.

    This makes each subplot panel square visually.
    It does NOT force the data aspect ratio to 1:1.
    """

    usable_w = 1.0 - left - right - (cols - 1) * hspace
    usable_h = 1.0 - top - bottom - (rows - 1) * vspace

    cell_w = usable_w / cols
    cell_h = usable_h / rows
    side = min(cell_w, cell_h)

    total_grid_w = cols * side + (cols - 1) * hspace
    total_grid_h = rows * side + (rows - 1) * vspace

    x0 = left + (usable_w - total_grid_w) / 2
    y0 = bottom + (usable_h - total_grid_h) / 2

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c + 1

            x_start = x0 + c * (side + hspace)
            x_end = x_start + side

            # plotly row 1 is the top row
            y_end = 1.0 - (y0 + r * (side + vspace))
            y_start = y_end - side

            xaxis_name = "xaxis" if idx == 1 else f"xaxis{idx}"
            yaxis_name = "yaxis" if idx == 1 else f"yaxis{idx}"

            if hasattr(fig.layout, xaxis_name):
                getattr(fig.layout, xaxis_name).domain = [_clamp01(x_start),_clamp01(x_end)]

            if hasattr(fig.layout, yaxis_name):
                getattr(fig.layout, yaxis_name).domain = [_clamp01(y_start),_clamp01(y_end)]

    return fig



def _axis_ref(axis: str, axis_num: int) -> str:
    """Return Plotly axis reference name: x, x2, x3, ... / y, y2, y3, ..."""
    return axis if axis_num == 1 else f"{axis}{axis_num}"


def _make_subplot_pixel_square(fig, row: int, col: int, axis_num: int, x_vals, y_vals) -> None:
    """Keep one subplot plotting box square in screen pixels.

    The data ranges are fixed explicitly, and Plotly is told to satisfy the
    linked x/y scale by shrinking the axis *domain* rather than expanding a
    data range. This avoids the extra blank axis area that appeared around
    3-sensor plots and also keeps a single-sensor plot square.
    """
    try:
        x = np.asarray(x_vals, dtype=float)
        y = np.asarray(y_vals, dtype=float)
        x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
        y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
        x_span = x_max - x_min
        y_span = y_max - y_min
    except Exception:
        x_min, x_max = 0.0, 1.0
        y_min, y_max = 0.0, 1.0
        x_span = y_span = 1.0

    if not np.isfinite(x_span) or x_span <= 0:
        x_min, x_max, x_span = 0.0, 1.0, 1.0
    if not np.isfinite(y_span) or y_span <= 0:
        y_min, y_max, y_span = 0.0, 1.0, 1.0

    # Keep the physical scan ranges unchanged.  ``constrain=domain`` makes
    # Plotly resize the plotting box instead of widening one of the ranges.
    fig.update_xaxes(
        range=[x_min, x_max],
        autorange=False,
        constrain="domain",
        row=row,
        col=col,
    )
    fig.update_yaxes(
        range=[y_min, y_max],
        autorange=False,
        scaleanchor=_axis_ref("x", axis_num),
        scaleratio=x_span / y_span,
        constrain="domain",
        row=row,
        col=col,
    )


def _add_subplot_title(fig, title: str, axis_num: int) -> None:
    """Add a title tied to the subplot's axis domains so it follows resizing."""
    xref = f'{_axis_ref("x", axis_num)} domain'
    yref = f'{_axis_ref("y", axis_num)} domain'
    fig.add_annotation(
        text=str(title),
        x=0.5,
        y=1.04,
        xref=xref,
        yref=yref,
        showarrow=False,
        xanchor="center",
        yanchor="bottom",
        font=dict(size=13),
    )

def rc(i: int) -> tuple[int, int]:
    """
    Returns the row and column (RC) for subplot with index i, based on 2 columns
    """
    return (i // 2) + 1, (i % 2) + 1

def make_readout_subplots_with_profile(
    da: xr.DataArray, profile: dict
) -> go.Figure:
    """
    Build the same readout grid used by the normal viewer, replacing one
    sensor panel with its line profile. Heatmap panels stay square and the
    profile view intentionally hides colorbars to keep the mixed layout clean.
    """
    assert isinstance(da, xr.DataArray) and da.ndim == 3 and "readout" in da.dims

    y_name, x_name = [d for d in da.dims if d != "readout"]
    labels = [str(v) for v in da.coords["readout"].values]
    n = len(labels)

    if profile.get("subplot_col") is not None:
        target_idx = max(0, min(int(profile["subplot_col"]) - 1, n - 1))
    else:
        target_label = str(profile.get("readout", labels[0]))
        try:
            target_idx = labels.index(target_label)
        except ValueError:
            target_idx = 0

    cols = 2 if n > 2 else n
    rows = 1 if n <= 2 else ceil(n / 2)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        horizontal_spacing=0.04 if n > 2 else safe_hspace(n),
        vertical_spacing=0.12 if rows > 1 else 0.08,
        shared_xaxes=False,
        shared_yaxes=False,
    )

    yc, xc = da.coords[y_name], da.coords[x_name]
    y_units = yc.attrs.get("units")
    x_units = xc.attrs.get("units")
    y_title = (
        f"{yc.attrs.get('long_name', y_name)} ({y_units})"
        if y_units
        else yc.attrs.get("long_name", y_name)
    )
    x_title = (
        f"{xc.attrs.get('long_name', x_name)} ({x_units})"
        if x_units
        else xc.attrs.get("long_name", x_name)
    )
    x_vals = np.asarray(xc.values)
    y_vals = np.asarray(yc.values)

    for i, label in enumerate(labels):
        r, c = rc(i)
        axis_num = i + 1

        if i == target_idx:
            s_vals = np.asarray(profile.get("s", []), dtype=float)
            signal_vals = np.asarray(profile.get("vals", []), dtype=float)
            fig.add_trace(
                go.Scatter(
                    x=s_vals,
                    y=signal_vals,
                    mode="lines",
                    name=f"profile_{label}",
                    showlegend=False,
                ),
                row=r,
                col=c,
            )
            fig.update_xaxes(
                title_text=profile.get("x_label", "Distance"),
                autorange=True,
                row=r,
                col=c,
            )
            fig.update_yaxes(
                title_text=profile.get("y_label", "Signal"),
                autorange=True,
                row=r,
                col=c,
            )
        else:
            sub = da.isel(readout=i).reset_coords("readout", drop=True)
            small = xarray_to_plotly(sub)
            if small.data:
                tr = small.data[0]
                # A shared colorbar is useful for normal 2D viewing, but in a
                # mixed heatmap/profile figure it consumes too much layout space.
                tr.update(showscale=False)
                try:
                    z = np.asarray(tr["z"])
                    z_rows, z_cols = z.shape
                    tr["customdata"] = np.full(
                        (z_rows, z_cols), i + 1, dtype=np.int16
                    )
                except Exception:
                    pass
                fig.add_trace(tr, row=r, col=c)
                fig.update_xaxes(title_text=x_title, row=r, col=c)
                if c == 1:
                    fig.update_yaxes(title_text=y_title, row=r, col=c)
                _make_subplot_pixel_square(
                    fig, r, c, axis_num, x_vals, y_vals
                )

        _add_subplot_title(fig, label, axis_num)

    # Line-profile views need more breathing room than the normal heatmap
    # layout so subplot titles, tick labels, and axis titles are not clipped.
    if n <= 2:
        left_margin = 70
        right_margin = 35
        top_margin = 70
        bottom_margin = 80
    else:
        left_margin = 55
        right_margin = 30
        top_margin = 55
        bottom_margin = 90

    fig.update_xaxes(title_standoff=12)
    fig.update_yaxes(title_standoff=12)

    fig.update_layout(
        template="plotly_dark",
        showlegend=False,
        margin=dict(
            l=left_margin,
            r=right_margin,
            t=top_margin,
            b=bottom_margin,
        ),
    )
    return fig


def _square_subplot_domains(
    fig: go.Figure,
    rows: int,
    cols: int,
    *,
    x_pad: float = 0.06,
    y_pad: float = 0.10,
) -> go.Figure:
    """Keep each cartesian subplot square in screen pixels.

    The outer viewer uses an aspect ratio of ``cols / rows``. Plotly axis
    domains are normalized independently in x and y, so equal normalized
    widths/heights are not square when the full figure is rectangular. This
    helper compensates for the figure aspect ratio while preserving the
    physical x/y data ranges.
    """
    if rows < 1 or cols < 1:
        return fig

    # The responsive outer viewer is cols:rows (1:1, 2:1, 2:2, ...).
    figure_aspect = cols / rows  # figure width / figure height

    # Maximum normalized space available to one cell after subplot gaps.
    max_cell_w = (1.0 - x_pad * max(cols - 1, 0)) / cols
    max_cell_h = (1.0 - y_pad * max(rows - 1, 0)) / rows

    # For a square in pixels:
    #   x_domain_width * figure_width == y_domain_height * figure_height
    # therefore y_domain_height = x_domain_width * figure_aspect.
    cell_w = min(max_cell_w, max_cell_h / figure_aspect)
    cell_h = cell_w * figure_aspect

    total_w = cols * cell_w + x_pad * max(cols - 1, 0)
    total_h = rows * cell_h + y_pad * max(rows - 1, 0)
    x0 = (1.0 - total_w) / 2.0
    y0 = (1.0 - total_h) / 2.0

    for idx in range(rows * cols):
        r = idx // cols
        c = idx % cols
        axis_num = idx + 1
        x_name = "xaxis" if axis_num == 1 else f"xaxis{axis_num}"
        y_name = "yaxis" if axis_num == 1 else f"yaxis{axis_num}"

        if not hasattr(fig.layout, x_name) or not hasattr(fig.layout, y_name):
            continue

        x_left = x0 + c * (cell_w + x_pad)
        x_right = x_left + cell_w
        y_top = 1.0 - y0 - r * (cell_h + y_pad)
        y_bottom = y_top - cell_h

        # Clamp tiny floating-point overshoots (e.g. -5e-17 or 1+5e-17)
        # because Plotly requires axis domains to lie strictly within [0, 1].
        x_left = max(0.0, min(1.0, x_left))
        x_right = max(0.0, min(1.0, x_right))
        y_bottom = max(0.0, min(1.0, y_bottom))
        y_top = max(0.0, min(1.0, y_top))

        getattr(fig.layout, x_name).domain = [x_left, x_right]
        getattr(fig.layout, y_name).domain = [y_bottom, y_top]

    return fig

def safe_hspace(n: int, default: float = 0.06) -> float:
    """Calculate a safe horizontal spacing to avoid overlap"""
    return min(default, 1.0 / (n - 1) - 1e-6) if n > 1 else default

def pick_readout_dim(da: xr.DataArray) -> tuple[str, str]:
    """
    Picks which Xarray DataArray dimension should be treated as 'readout' or 2D
    """

    dims = list(da.dims)
    if "readout" in dims:
        rd = "readout"
        return rd, [d for d in dims if d != rd][0]

    def _non_numeric(d: str) -> bool:
        c = da.coords.get(d)
        if c is None:
            return False
        return np.asarray(c.values).dtype.kind not in ("i", "u", "f")

    rd = min(dims, key=lambda d: (da.sizes[d], 0 if _non_numeric(d) else 1))
    return rd, [d for d in dims if d != rd][0]

def axis_vals_and_label(
    da: xr.DataArray, dim: str
) -> tuple[np.ndarray, str]:
    """Pulls the coordinate alues for dims, otherwise np.arange. Also builds an axis label"""
    c = da.coords.get(dim)
    x = np.asarray(c.values) if c is not None else np.arange(da.sizes[dim])
    label = c.attrs.get("label", dim) if c is not None else dim
    u = c.attrs.get("units") if c is not None else None
    if u:
        label = f"{label} [{u}]"
    return x, label

def is_line(da: xr.DataArray) -> bool:
    """
    Returns True if input DataArray is effectively a 1D line
    """
    return (
        isinstance(da, xr.DataArray)
        and "readout" not in da.dims
        and (da.ndim == 1 or (da.ndim == 2 and 1 in da.shape))
    )

def make_readout_line_subplots(da: xr.DataArray) -> go.Figure:
    """
    Renders a 2D DataArray as multiple line plots; one per 'readout'

    - If there are <= 8 lines, it builds a small multi-subplot grid with two columns
    - If there are > 8 lines (unlikely, as >4 often crashes the programme) renders them in a single figure with a legend
    """
    assert da.ndim == 2
    rd_dim, x_dim = pick_readout_dim(da)
    labels = [
        str(v)
        for v in (
            np.asarray(da.coords[rd_dim].values)
            if rd_dim in da.coords
            else range(da.sizes[rd_dim])
        )
    ]
    n = len(labels)
    x_vals, x_label = axis_vals_and_label(da, x_dim)

    if n > 8:
        fig = go.Figure()
        for i, lab in enumerate(labels):
            y = np.asarray(da.isel({rd_dim: i}).values).ravel()
            fig.add_trace(go.Scatter(x=x_vals, y=y, mode="lines", name=lab))
        fig.update_layout(
            template="plotly_dark",
            xaxis_title=x_label,
            yaxis_title="Value",
            showlegend=True,
        )
        return fig

    if n <= 1:
        y = np.asarray(da.isel({rd_dim: 0}).values).ravel()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=y, mode="lines", name=labels[0]))
        fig.update_layout(
            template="plotly_dark",
            xaxis_title=x_label,
            yaxis_title="Value",
            showlegend=False,
        )
        return fig

    cols = 2 if n > 2 else n
    rows = 1 if n <= 2 else ceil(n / 2)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=labels,
        horizontal_spacing=0.08,
        vertical_spacing=0.16,
        shared_xaxes=False,
        shared_yaxes=False,
    )

    for i, lab in enumerate(labels):
        r, c = rc(i)
        y = np.asarray(da.isel({rd_dim: i}).values).ravel()
        fig.add_trace(
            go.Scatter(x=x_vals, y=y, mode="lines", name=lab, showlegend=False),
            row=r,
            col=c,
        )
        fig.update_xaxes(title_text=x_label, row=r, col=c)
        if c == 1:
            fig.update_yaxes(title_text="Value", row=r, col=c)

    fig.update_layout(
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=60, r=30, t=40, b=40),
    )
    return _square_subplot_domains(fig, rows, cols)

def make_readout_subplots(da: xr.DataArray) -> go.Figure:
    """
    For true 3D data (one dimension is 'readout'), build a 1xN grid of 2D plots.
    Each plotting box is kept square in screen pixels.
    """
    if not (isinstance(da, xr.DataArray) and da.ndim == 3 and "readout" in da.dims):
        return xarray_to_plotly(da)

    y_name, x_name = [d for d in da.dims if d != "readout"]
    labels = [str(v) for v in da.coords["readout"].values]
    n = len(labels)

    rows = 1
    cols = n
    fig = make_subplots(
        rows=rows,
        cols=cols,
        horizontal_spacing=safe_hspace(n),
    )

    yc, xc = da.coords[y_name], da.coords[x_name]
    y_units = yc.attrs.get("units")
    y_title = (
        f"{yc.attrs.get('long_name', y_name)} ({y_units})"
        if y_units
        else yc.attrs.get("long_name", y_name)
    )
    x_units = xc.attrs.get("units")
    x_title = (
        f"{xc.attrs.get('long_name', x_name)} ({x_units})"
        if x_units
        else xc.attrs.get("long_name", x_name)
    )

    x_vals = np.asarray(xc.values)
    y_vals = np.asarray(yc.values)

    for i in range(n):
        sub = da.isel(readout=i).reset_coords("readout", drop=True)
        small = xarray_to_plotly(sub)
        if not small.data:
            continue
        tr = small.data[0]
        tr.update(showscale=(i == n - 1))
        try:
            z = np.asarray(tr["z"])
            z_rows, z_cols = z.shape
            tr["customdata"] = np.full((z_rows, z_cols), i + 1, dtype=np.int16)
        except Exception:
            pass

        col = i + 1
        axis_num = i + 1
        fig.add_trace(tr, row=1, col=col)
        fig.update_xaxes(title_text=x_title, row=1, col=col)
        if col == 1:
            fig.update_yaxes(title_text=y_title, row=1, col=col)

        _make_subplot_pixel_square(fig, 1, col, axis_num, x_vals, y_vals)
        _add_subplot_title(fig, labels[i], axis_num)

    fig.update_layout(
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=50, r=25, t=48, b=52),
    )
    return fig

def build_live_figure(data_object: dict) -> go.Figure:
    """Return a figure for a registry 'live' object (dict that may contain base_image_data or data)."""
    base = None
    if isinstance(data_object, dict):
        base = data_object.get("base_image_data")
        if base is None:
            base = data_object.get("data")
    return figure_from_data(base).update_layout(clickmode="event+select")

def build_static_figure(static_obj: dict, ui_state: dict | None) -> go.Figure:
    """
    Return a figure for a compound static object: may include
    base_image_data (xr.DataArray), annotations (dict), profile_plot (dict).
    """
    base = None
    profile = None
    ann = None
    if isinstance(static_obj, dict):
        base = static_obj.get("base_image_data")
        profile = static_obj.get("profile_plot")
        ann = static_obj.get("annotations")

    if isinstance(profile, dict):
        fig = figure_with_profile(base, profile)
        return apply_annotations(fig, base, ann, ui_state, profile)
    else:
        fig = figure_from_data(base)
        return apply_annotations(fig, base, ann, ui_state, profile)

def empty_dark() -> go.Figure:
    """Return a blank dark-theme figure as placeholder"""
    return go.Figure().update_layout(template="plotly_dark")

def figure_from_data(da: xr.DataArray | None) -> go.Figure:
    """
    Reference the appropriate plot builder based on shape and the presence of a 'readout' dimension
    """
    if not isinstance(da, xr.DataArray):
        return empty_dark()

    if is_line(da):
        d = da.squeeze(drop=True)
        if d.ndim == 2 and 1 in d.shape:
            d = xr.DataArray(
                d.values.ravel(), dims=("idx",), coords={"idx": np.arange(d.size)}
            )
        x_dim = d.dims[0]
        c = d.coords.get(x_dim)
        x = np.asarray(c.values) if c is not None else np.arange(d.size)
        xlabel = c.attrs.get("label") if c is not None else str(x_dim)
        u = c.attrs.get("units") if c is not None else None
        if u:
            xlabel = f"{xlabel} [{u}]"

        # Invisible 2D grid under the 1D plot, so that click to jump works. 
        y = np.asarray(d.values).ravel()
        ymin = float(np.nanmin(y)) if np.size(y) else 0.0
        ymax = float(np.nanmax(y)) if np.size(y) else 1.0
        span = (ymax - ymin) if (ymax > ymin) else 1.0
        pad = 0.05 * span
        cap_y = [ymin - pad, ymax + pad]
        cap_z = np.zeros((2, len(x)), dtype=float)


        fig = go.Figure()
        # Visible 1D line.
        fig.add_trace(go.Scatter(x=x, y=np.asarray(d.values).ravel(), mode="lines"))

        fig.add_trace(
            go.Heatmap(
                x=x,
                y=cap_y,
                z=cap_z,
                showscale=False,
                name="_click_capture_1d",
                colorscale=[[0, "rgba(128,128,128,0.001)"], [1, "rgba(128,128,128,0.001)"]],
                zmin=0, zmax=1,
                hoverongaps=False
            )
        )
        
        return fig.update_layout(
            template="plotly_dark", xaxis_title=xlabel, yaxis_title="Value"
        )

    if da.ndim == 2 and "readout" in da.dims:
        return make_readout_line_subplots(da)

    if da.ndim == 3 and "readout" in da.dims:
        try:
            n_readouts = int(da.sizes["readout"])
        except Exception:
            n_readouts = len(list(da.coords["readout"].values))
        if n_readouts > 2:
            return build_two_col_readout_grid(da)
        return make_readout_subplots(da)

    return xarray_to_plotly(da).update_layout(template="plotly_dark")

def figure_with_profile(da: xr.DataArray | None, profile: dict) -> go.Figure:
    """
    Builds a figure that overlays a profile plot onto one of the subplots
    """
    if da.ndim == 3 and "readout" in da.dims:
        try:
            return make_readout_subplots_with_profile(da, profile)
        except Exception:
            pass
    s = profile.get("s", [])
    vals = profile.get("vals", [])
    y_label = profile.get("y_label", "Value")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s, y=vals, mode="lines"))

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=70, r=35, t=60, b=80),
        xaxis_title=profile.get("x_label", "Distance"),
        yaxis_title=y_label,
        showlegend=False,
    )
    fig.update_xaxes(autorange=True, title_standoff=12)
    fig.update_yaxes(autorange=True, title_standoff=12)
    return fig

def apply_annotations(
    fig: go.Figure,
    da: xr.DataArray | None,
    annotations: dict | None,
    ui_state: dict | None,
    profile: dict | None = None,
) -> go.Figure:
    """
    Adds point/line annotations to a figure. Plots the relevant annotations on the appropriate figure, respecting the subplots layout
    """
    if not isinstance(annotations, dict):
        return fig
    
    if da is None:
        return fig

    is_multipane = da.ndim == 3 and "readout" in da.dims
    n_readouts = int(da.sizes["readout"]) if is_multipane else 1
    two_col = is_multipane and n_readouts > 2

    def rc_for(col: int | None) -> tuple[int, int]:
        """Return (row, col) for a given 1-based subplot column marker."""
        if not is_multipane or col is None:
            return 1, 1
        c = max(1, int(col))
        if not two_col:
            # 1×N layout
            return 1, c
        # 2-column packing: 1->(1,1), 2->(1,2), 3->(2,1), 4->(2,2), ...
        r = (c - 1) // 2 + 1
        cc = 2 if c % 2 == 0 else 1
        return r, cc

    points = annotations.get("points") or []
    lines = annotations.get("lines") or []

    def add(tr, col: int | None = None):
        if not is_multipane:
            fig.add_trace(tr)
        else:
            r, c = rc_for(col)
            fig.add_trace(tr, row=r, col=c)

    target_col: int | None = None
    try:
        target_col = resolve_profile_col(da, profile)
    except Exception:
        target_col = None

    if lines and points:
        by_id = {str(p.get("id")): p for p in points}

        for ln in lines:
            sid, eid = str(ln.get("start_point_id")), str(ln.get("end_point_id"))
            p1, p2 = by_id.get(sid), by_id.get(eid)
            if not p1 or not p2:
                continue
            try:
                col = int(ln.get("subplot_col", p1.get("subplot_col", 1)) or 1)
            except Exception:
                col = 1
            if target_col is not None and col == target_col:
                continue

            add(
                go.Scatter(
                    x=[float(p1["x"]), float(p2["x"])],
                    y=[float(p1["y"]), float(p2["y"])],
                    mode="lines",
                    name="annotations_lines",
                    showlegend=False,
                    hoverinfo="skip",
                    line=dict(width=2, color="white"),
                    customdata=[[str(ln.get("id", "")), col], [str(ln.get("id", "")), col]],
                ),
                col,
            )

    if points:
        by_col: dict[int, list[dict]] = {}
        for p in points:
            try:
                c = int(p.get("subplot_col", 1) or 1)
            except Exception:
                c = 1
            by_col.setdefault(c, []).append(p)

        show_labels = bool(
            ui_state
            and "show_labels" in ui_state
            and "points" in (ui_state.get("show_labels") or [])
        )

        for col, pts in by_col.items():
            skip = target_col is not None and col == target_col
            if skip:
                continue

            xs = [float(p["x"]) for p in pts]
            ys = [float(p["y"]) for p in pts]
            ids = [str(p["id"]) for p in pts]

            add(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    name="annotations_points",
                    showlegend=False,
                    hoverinfo="skip",
                    marker=dict(
                        size=10, symbol="circle", color="white", line=dict(width=2)
                    ),
                    customdata=[[pid, col] for pid in ids],
                ),
                col,
            )

            if show_labels:
                add(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="text",
                        text=ids,
                        textposition="top center",
                        name="annotations_point_labels",
                        showlegend=False,
                        hoverinfo="skip",
                        customdata=[[pid, col] for pid in ids],
                    ),
                    col,
                )

    return fig

def resolve_profile_col(
    da: xr.DataArray | None, profile: dict | None
) -> int | None:
    """Return 1-based subplot column for the profile, or None if unknown."""
    if not isinstance(profile, dict):
        return None

    try:
        if profile.get("subplot_col") is not None:
            col = int(profile["subplot_col"])
            return col
    except Exception:
        pass

    if "readout" in da.dims and "readout" in profile:
        labels = [str(v) for v in da.coords["readout"].values]
        target = str(profile["readout"])
        try:
            idx = labels.index(target) + 1
            return idx
        except ValueError:
            return None

def build_two_col_readout_grid(da: xr.DataArray) -> go.Figure:
    """Build a 2-column grid for 3+ readouts with consistent square panels."""
    assert isinstance(da, xr.DataArray) and da.ndim == 3 and "readout" in da.dims

    y_name, x_name = [d for d in da.dims if d != "readout"]
    labels = [str(v) for v in da.coords["readout"].values]
    n = len(labels)
    cols = 2
    rows = ceil(n / 2)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        horizontal_spacing=0.04,
        vertical_spacing=0.12,
        shared_xaxes=False,
        shared_yaxes=False,
    )

    yc, xc = da.coords[y_name], da.coords[x_name]
    y_units = yc.attrs.get("units")
    y_title = (
        f"{yc.attrs.get('long_name', y_name)} ({y_units})"
        if y_units
        else yc.attrs.get("long_name", y_name)
    )
    x_units = xc.attrs.get("units")
    x_title = (
        f"{xc.attrs.get('long_name', x_name)} ({x_units})"
        if x_units
        else xc.attrs.get("long_name", x_name)
    )

    x_vals = np.asarray(xc.values)
    y_vals = np.asarray(yc.values)

    for i, label in enumerate(labels):
        row, col = rc(i)
        axis_num = i + 1

        sub = da.isel(readout=i).reset_coords("readout", drop=True)
        small = xarray_to_plotly(sub)
        if not small.data:
            continue

        tr = small.data[0]
        tr.update(showscale=(i == n - 1))
        try:
            z = np.asarray(tr["z"])
            z_rows, z_cols = z.shape
            tr["customdata"] = np.full((z_rows, z_cols), i + 1, dtype=np.int16)
        except Exception:
            pass

        fig.add_trace(tr, row=row, col=col)
        fig.update_xaxes(title_text=x_title, row=row, col=col)
        if col == 1:
            fig.update_yaxes(title_text=y_title, row=row, col=col)

        _make_subplot_pixel_square(fig, row, col, axis_num, x_vals, y_vals)
        _add_subplot_title(fig, label, axis_num)

    # Multi-row (3/4-sensor) layouts need a little more room below the
    # bottom row for x tick labels and the x-axis title. Keep the 1-row
    # layout unchanged so 1/2-sensor plots retain their current size.
    bottom_margin = 82 if rows > 1 else 52

    fig.update_layout(
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=50, r=25, t=48, b=bottom_margin),
    )
    return fig


import xarray as xr
import plotly.graph_objects as go
import numpy as np
from qua_dashboards.video_mode.utils.dash_utils import xarray_to_plotly
from plotly.subplots import make_subplots
import logging
__all__ = ["SharedDataHandler"]
logger = logging.getLogger(__name__)
class SharedDataHandler:
    def __init__(self, 
                 ):
        pass
    def _make_readout_subplots_with_profile(self, da: xr.DataArray, profile: dict) -> go.Figure:
        assert isinstance(da, xr.DataArray) and da.ndim == 3 and "readout" in da.dims
        y_name, x_name = [d for d in da.dims if d != "readout"]
        rd_vals = list(da.coords["readout"].values)
        labels = [str(v) for v in rd_vals]
        n = len(labels)
        if "subplot_col" in profile:
            target_idx = max(0, int(profile["subplot_col"]) - 1)
        else:
            target_label = str(profile.get("readout", labels[0]))
            try:
                target_idx = labels.index(target_label)
            except ValueError:
                target_idx = 0

        fig = make_subplots(rows=1, cols=n, subplot_titles=labels,
                            horizontal_spacing=min(0.06, 1.0/(max(n-1,1)) - 1e-6))
        yc, xc = da.coords[y_name], da.coords[x_name]
        y_lab = yc.attrs.get("long_name", y_name)
        y_unit = yc.attrs.get("units", "")
        x_lab = xc.attrs.get("long_name", x_name)
        x_unit = xc.attrs.get("units", "")
        y_title = f"{y_lab} ({y_unit})" if y_unit else y_lab
        x_title = f"{x_lab} ({x_unit})" if x_unit else x_lab
        for i in range(n):
            if i == target_idx:
                s = np.asarray(profile.get("s", []))
                vals = np.asarray(profile.get("vals", []))
                fig.add_trace(
                    go.Scatter(x=s, y=vals, mode="lines", name=f"profile_{labels[i]}", showlegend=False),
                    row=1, col=i+1
                )
                fig.update_xaxes(title_text=profile.get("x_label", "Distance"), row=1, col=i+1)
                fig.update_yaxes(title_text=profile.get("y_label", "Value"),    row=1, col=i+1)
            else:
                sub = da.isel(readout=i).reset_coords("readout", drop=True)
                small = xarray_to_plotly(sub)
                if small.data:
                    tr = small.data[0]
                    tr.update(showscale=(i == n - 1))
                    try:
                        z = np.asarray(tr["z"])
                        rows, cols = z.shape
                        tr["customdata"] = [[i + 1] * cols for _ in range(rows)]
                    except Exception:
                        pass
                    fig.add_trace(tr, row=1, col=i+1)
                    fig.update_xaxes(title_text=x_title, row=1, col=i+1)
                if i == 0:
                    fig.update_yaxes(title_text=y_title, row=1, col=1)

        fig.update_layout(template="plotly_dark", showlegend=False, margin=dict(l=60, r=30, t=40, b=40))
        return fig
    
    def _safe_hspace(self, n: int, default: float = 0.06) -> float:
        return min(default, 1.0 / (n - 1) - 1e-6) if n > 1 else default
    
    def _pick_readout_dim(self, da: xr.DataArray) -> tuple[str, str]:
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

    def _axis_vals_and_label(self, da: xr.DataArray, dim: str) -> tuple[np.ndarray, str]:
        c = da.coords.get(dim)
        x = np.asarray(c.values) if c is not None else np.arange(da.sizes[dim])
        label = (c.attrs.get("label", dim) if c is not None else dim)
        u = (c.attrs.get("units") if c is not None else None)
        if u:
            label = f"{label} [{u}]"
        return x, label
    
    def _is_line(self, da: xr.DataArray) -> bool:
        return (
            isinstance(da, xr.DataArray)
            and "readout" not in da.dims
            and (da.ndim == 1 or (da.ndim == 2 and 1 in da.shape))
        )


    def _make_readout_line_subplots(self, da: xr.DataArray) -> go.Figure:
        assert isinstance(da, xr.DataArray) and da.ndim == 2
        rd_dim, x_dim = self._pick_readout_dim(da)
        labels = [str(v) for v in (np.asarray(da.coords[rd_dim].values)
                                if rd_dim in da.coords else range(da.sizes[rd_dim]))]
        n = len(labels)
        x_vals, x_label = self._axis_vals_and_label(da, x_dim)

        if n > 8:
            fig = go.Figure()
            for i, lab in enumerate(labels):
                y = np.asarray(da.isel({rd_dim: i}).values).ravel()
                fig.add_trace(go.Scatter(x=x_vals, y=y, mode="lines", name=lab))
            fig.update_layout(template="plotly_dark", xaxis_title=x_label, yaxis_title="Value", showlegend=True)
            return fig

        if n <= 1:
            y = np.asarray(da.isel({rd_dim: 0}).values).ravel()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_vals, y=y, mode="lines", name=labels[0]))
            fig.update_layout(template="plotly_dark", xaxis_title=x_label, yaxis_title="Value", showlegend=False)
            return fig

        fig = make_subplots(rows=1, cols=n, subplot_titles=labels, horizontal_spacing=self._safe_hspace(n))
        for i, lab in enumerate(labels):
            y = np.asarray(da.isel({rd_dim: i}).values).ravel()
            fig.add_trace(go.Scatter(x=x_vals, y=y, mode="lines", name=lab, showlegend=False), row=1, col=i + 1)
            fig.update_xaxes(title_text=x_label, row=1, col=i + 1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_layout(template="plotly_dark", showlegend=False, margin=dict(l=60, r=30, t=40, b=40))
        return fig
    
    def _make_readout_subplots(self, da: xr.DataArray) -> go.Figure:
        if not (isinstance(da, xr.DataArray) and da.ndim == 3 and "readout" in da.dims):
            return xarray_to_plotly(da)

        y_name, x_name = [d for d in da.dims if d != "readout"]
        labels = [str(v) for v in da.coords["readout"].values]
        n = len(labels)

        fig = make_subplots(rows=1, cols=n, subplot_titles=labels, horizontal_spacing=self._safe_hspace(n))
        yc, xc = da.coords[y_name], da.coords[x_name]
        y_title = f"{yc.attrs.get('long_name', y_name)} ({yc.attrs.get('units','')})".strip(" ()")
        x_title = f"{xc.attrs.get('long_name', x_name)} ({xc.attrs.get('units','')})".strip(" ()")

        for i in range(n):
            sub = da.isel(readout=i).reset_coords("readout", drop=True)
            small = xarray_to_plotly(sub)
            if not small.data:
                continue
            tr = small.data[0]
            tr.update(showscale=(i == n - 1))
            try:
                z = np.asarray(tr["z"])
                rows, cols = z.shape
                tr["customdata"] = [[i + 1] * cols for _ in range(rows)]
            except Exception:
                pass
            fig.add_trace(tr, row=1, col=i + 1)
        for col in range(1, n + 1):
            fig.update_xaxes(title_text=x_title, row=1, col=col)
        fig.update_yaxes(title_text=y_title, row=1, col=1)
        fig.update_layout(template="plotly_dark", showlegend=False, margin=dict(l=60, r=30, t=40, b=40))
        return fig
    
    def build_live_figure(self, data_object: dict) -> go.Figure:
        """Return a figure for a registry 'live' object (dict that may contain base_image_data or data)."""
        base = None
        if isinstance(data_object, dict):
            base = data_object.get("base_image_data")
            if base is None:
                base = data_object.get("data")
        return self._figure_from_data(base)
    
    def build_static_figure(self, static_obj: dict, ui_state: dict | None) -> go.Figure:
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
            fig = self._figure_with_profile(base, profile)
            return self._apply_annotations(fig, base, ann, ui_state, profile)
        else:
            fig = self._figure_from_data(base)  
            return self._apply_annotations(fig, base, ann, ui_state, profile)
    
    def empty_dark(self) -> go.Figure:
        return go.Figure().update_layout(template="plotly_dark")
    
    def _figure_from_data(self, da: xr.DataArray | None) -> go.Figure:
        if not isinstance(da, xr.DataArray):
            return self.empty_dark()

        if self._is_line(da):
            d = da.squeeze(drop=True)
            if d.ndim == 2 and 1 in d.shape:
                d = xr.DataArray(d.values.ravel(), dims=("idx",), coords={"idx": np.arange(d.size)})
            x_dim = d.dims[0]
            c = d.coords.get(x_dim)
            x = np.asarray(c.values) if c is not None else np.arange(d.size)
            xlabel = (c.attrs.get("label") if c is not None else str(x_dim))
            u = (c.attrs.get("units") if c is not None else None)
            if u:
                xlabel = f"{xlabel} [{u}]"
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=np.asarray(d.values).ravel(), mode="lines"))
            return fig.update_layout(template="plotly_dark", xaxis_title=xlabel, yaxis_title="Value")

        if da.ndim == 2 and "readout" in da.dims:
            return self._make_readout_line_subplots(da)

        if da.ndim == 3 and "readout" in da.dims:
            return self._make_readout_subplots(da)

        return xarray_to_plotly(da).update_layout(template="plotly_dark")
    
    def _figure_with_profile(self, da: xr.DataArray | None, profile: dict) -> go.Figure:
        if isinstance(da, xr.DataArray) and da.ndim == 3 and "readout" in da.dims:
            try:
                return self._make_readout_subplots_with_profile(da, profile)
            except Exception:
                pass
        s = profile.get("s", [])
        vals = profile.get("vals", [])
        y_label = profile.get("y_label", "Value")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s, y=vals, mode="lines"))
        
        return fig.update_layout(
            template="plotly_dark",
            margin=dict(l=40, r=10, t=10, b=40),
            xaxis_title="Arbitrary Units",
            yaxis_title=y_label,
            showlegend=False,
        )
    
    def _apply_annotations(self, fig: go.Figure, da: xr.DataArray | None, annotations: dict | None, ui_state: dict | None, profile: dict | None = None) -> go.Figure:
        if not isinstance(annotations, dict):
            return fig

        points = annotations.get("points") or []
        lines  = annotations.get("lines") or []
        
        n_cols = 1
        if isinstance(da, xr.DataArray) and da.ndim == 3 and "readout" in da.dims:
            try:
                n_cols = int(len(da.coords["readout"]))
            except Exception:
                n_cols = 2

        def add(tr, col: int | None = None):
            if n_cols == 1 or col is None:
                fig.add_trace(tr)
            else:
                fig.add_trace(tr, row=1, col=col)

        target_col: int | None = None
        try:
            target_col = self._resolve_profile_col(da, profile)
        except Exception:
            target_col = None

        if lines and points:
            by_id = {str(p.get("id")): p for p in points}
            target_col = self._resolve_profile_col(da, profile)
            logger.debug("Apply annotations: target_col=%s", target_col)

            for ln in lines:
                sid, eid = str(ln.get("start_point_id")), str(ln.get("end_point_id"))
                p1, p2 = by_id.get(sid), by_id.get(eid)
                if not p1 or not p2:
                    continue
                try:
                    col = int(ln.get("subplot_col", p1.get("subplot_col", 1)) or 1)
                except Exception:
                    col = 1
                logger.debug("Line id=%s col=%s (skip=%s)", ln.get("id"), col, (target_col is not None and col == target_col))
                if target_col is not None and col == target_col:
                    continue 

                add(go.Scatter(
                    x=[float(p1["x"]), float(p2["x"])],
                    y=[float(p1["y"]), float(p2["y"])],
                    mode="lines",
                    name="annotations_lines",
                    showlegend=False,
                    hoverinfo="skip",
                    line=dict(width=2, color="white"),
                ), col)

        if points:
            by_col: dict[int, list[dict]] = {}
            for p in points:
                try:
                    c = int(p.get("subplot_col", 1) or 1)
                except Exception:
                    c = 1
                by_col.setdefault(c, []).append(p)

            show_labels = bool(ui_state and "show_labels" in ui_state and "points" in (ui_state.get("show_labels") or []))

            for col, pts in by_col.items():
                skip = (target_col is not None and col == target_col)
                logger.debug("Points col=%s count=%s (skip=%s)", col, len(pts), skip)
                if skip:
                    continue

                xs = [float(p["x"]) for p in pts]
                ys = [float(p["y"]) for p in pts]
                ids = [str(p["id"]) for p in pts]

                add(go.Scatter(
                    x=xs, y=ys, mode="markers",
                    name="annotations_points", showlegend=False, hoverinfo="skip",
                    marker=dict(size=10, symbol="circle", color="white", line=dict(width=2)),
                    customdata=[[pid] for pid in ids],
                ), col)

                if show_labels:
                    add(go.Scatter(
                        x=xs, y=ys, mode="text", text=ids, textposition="top center",
                        name="annotations_point_labels", showlegend=False, hoverinfo="skip",
                        customdata=[[pid] for pid in ids],
                    ), col)

        return fig
    

    def _resolve_profile_col(self, da: xr.DataArray | None, profile: dict | None) -> int | None:
        """Return 1-based subplot column for the profile, or None if unknown."""
        if not isinstance(profile, dict):
            return None

        try:
            if profile.get("subplot_col") is not None:
                col = int(profile["subplot_col"])
                logger.debug("Profile resolver: using explicit subplot_col=%s", col)
                return col
        except Exception:
            pass

        if isinstance(da, xr.DataArray) and "readout" in da.dims and "readout" in profile:
            labels = [str(v) for v in da.coords["readout"].values]
            target = str(profile["readout"])
            try:
                idx = labels.index(target) + 1
                logger.debug("Profile resolver: derived subplot_col=%s from readout=%s", idx, target)
                return idx
            except ValueError:
                logger.debug("Profile resolver: readout '%s' not in %s", target, labels)

        logger.debug("Profile resolver: no subplot could be resolved")
        return None


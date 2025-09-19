import xarray as xr
import plotly.graph_objects as go
import numpy as np
from qua_dashboards.video_mode.utils.dash_utils import xarray_to_plotly
from plotly.subplots import make_subplots

__all__ = ["SharedDataHandler"]

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
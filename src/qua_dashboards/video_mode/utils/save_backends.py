from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from qualang_tools.external_frameworks.qcodes.saving import save_xarray as qcodes_save_xarray
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

class SaveBackend(ABC):
    name = "Save backend"

    @abstractmethod
    def save(
        self,
        data: xr.DataArray,
        *,
        annotations: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: str = "video_mode",
        machine=None,
    ):
        pass

    def success_message(self, details=None) -> str:
        return f"Data saved successfully using {self.name}."

    def error_message(self, error: Exception) -> str:
        return f"Failed to save data using {self.name}: {error}"


class QualibrateBackend(SaveBackend):
    name = "Qualibrate"

    def __init__(self, save_quam_state=True):
        self.save_quam_state = save_quam_state

    def save(
        self,
        data: xr.DataArray,
        *,
        annotations=None,
        metadata=None,
        name="video_mode",
        machine=None,
    ) -> Dict[str, Any]:
        from qua_dashboards.video_mode.utils.data_utils import save_data

        results = {
            "image_data": {
                "base_image_data": data,
                "annotations": annotations or {},
            }
        }

        return save_data(
            results,
            machine=machine if self.save_quam_state else None,
        )

    def success_message(self, details=None) -> str:
        if not isinstance(details, dict):
            return "Data saved successfully using Qualibrate."

        idx = details.get("idx")
        path = details.get("path")

        if idx is None or path is None:
            return "Data saved successfully using Qualibrate."

        message = f"Data **#{idx}** saved to {path}."

        if ".qualibrate" in str(path):
            message += (
                " To change the data root folder, please run "
                "**qualibrate-config config --storage-location "
                "<your-desired-root-folder>**"
            )

        return message


def _plot_annotations(ax, annotations: Optional[Dict[str, Any]]) -> None:
    if not annotations:
        return

    points = annotations.get("points", []) or []
    lines = annotations.get("lines", []) or []
    point_map = {p.get("id"): p for p in points if isinstance(p, dict)}

    for point in points:
        if not isinstance(point, dict):
            continue
        x = point.get("x")
        y = point.get("y")
        if x is None or y is None:
            continue
        ax.plot([x], [y], marker="o", linestyle="None")
        point_id = point.get("id")
        if point_id:
            ax.annotate(str(point_id), (x, y), xytext=(4, 4), textcoords="offset points")

    for line in lines:
        if not isinstance(line, dict):
            continue
        start_point = point_map.get(line.get("start_point_id"))
        end_point = point_map.get(line.get("end_point_id"))
        if start_point is None or end_point is None:
            continue
        x0, y0 = start_point.get("x"), start_point.get("y")
        x1, y1 = end_point.get("x"), end_point.get("y")
        if None in (x0, y0, x1, y1):
            continue
        ax.plot([x0, x1], [y0, y1])


def _plot_single_dataarray(ax, data: xr.DataArray, title: str, annotations: Optional[Dict[str, Any]] = None) -> None:
    if data.ndim == 2:
        y_dim, x_dim = data.dims
        x = data.coords[x_dim].values if x_dim in data.coords else np.arange(data.shape[1])
        y = data.coords[y_dim].values if y_dim in data.coords else np.arange(data.shape[0])
        mesh = ax.pcolormesh(x, y, data.values, shading="auto")
        ax.set_xlabel(x_dim)
        ax.set_ylabel(y_dim)
        ax.set_title(title)
        cbar = ax.figure.colorbar(mesh, ax=ax)
        cbar.set_label(title)
        _plot_annotations(ax, annotations)
        return

    if data.ndim == 1:
        x_dim = data.dims[0]
        x = data.coords[x_dim].values if x_dim in data.coords else np.arange(data.shape[0])
        ax.plot(x, data.values)
        ax.set_xlabel(x_dim)
        ax.set_ylabel(title)
        ax.set_title(title)
        return

    ax.text(0.5, 0.5, f"Unsupported shape: {data.shape}", ha="center", va="center")
    ax.set_axis_off()
    ax.set_title(title)


def _build_datahandler_figure(data: xr.DataArray, annotations=None, metadata=None, name="video_mode"):
    plot_name = data.name or name
    plot_data = data.squeeze()

    if isinstance(plot_data, xr.DataArray) and "readout" in plot_data.dims:
        labels = list(plot_data.coords["readout"].values)
        result_names = list((metadata or {}).get("readout_result_names", []))
        n = len(labels)
        if n == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No readout channels", ha="center", va="center")
            ax.set_axis_off()
            return fig
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(6 * ncols, 5 * nrows))
        axes_flat = axes.flatten()
        for i, label in enumerate(labels):
            sensor_data = plot_data.sel(readout=label)
            title = result_names[i] if i < len(result_names) else str(label)
            if not result_names and data.attrs.get("result_type"):
                title = f"{label}_{data.attrs['result_type']}"
            _plot_single_dataarray(axes_flat[i], sensor_data, title, annotations=annotations)
        for j in range(n, len(axes_flat)):
            axes_flat[j].set_axis_off()
        fig.tight_layout()
        return fig

    fig, ax = plt.subplots(figsize=(6, 5))
    _plot_single_dataarray(ax, plot_data, plot_name, annotations=annotations)
    fig.tight_layout()
    return fig


class DataHandlerBackend(SaveBackend):
    name = "DataHandler"

    def __init__(
        self,
        handler,
        save_quam_state=True,
        instrument_snapshot=None,
    ):
        self.handler = handler
        self.save_quam_state = save_quam_state
        self.instrument_snapshot = instrument_snapshot or {}

    def save(
        self,
        data: xr.DataArray,
        *,
        annotations=None,
        metadata=None,
        name="video_mode",
        machine=None,
    ) -> Dict[str, Any]:

        figure = _build_datahandler_figure(data, annotations=annotations, metadata=metadata, name=name)

        save_data = {
            "data": data.values,
            "annotations": annotations or {},
            "metadata": dict(metadata or {}),
            "dims": list(data.dims),
            "video_mode_figure": figure,
        }

        # Save all axis coordinates explicitly
        for dim in data.dims:
            if dim in data.coords:
                save_data[f"coord_{dim}"] = data.coords[dim].values

        # Save DataArray attributes as well
        if data.attrs:
            save_data["data_attrs"] = dict(data.attrs)

        if data.name is not None:
            save_data["data_name"] = data.name

        # Optional instrument snapshots
        if self.instrument_snapshot:
            save_data["instrument_snapshot"] = {
                instrument_name: snapshot_provider()
                for instrument_name, snapshot_provider
                in self.instrument_snapshot.items()
            }

        path = self.handler.save_data(
            data=save_data,
            name=name,
        )

        plt.close(figure)

        if self.save_quam_state and machine is not None:
            machine.save(path)

        return {
            "path": path,
        }

class QCoDeSBackend(SaveBackend):
    name = "QCoDeS"

    def __init__(
        self,
        save_quam_state=True,
    ):
        self.save_quam_state = save_quam_state

    def save(
        self,
        data: xr.DataArray,
        *,
        annotations=None,
        metadata=None,
        name="video_mode",
        machine=None,
    ) -> Dict[str, Any]:

        save_metadata = dict(metadata or {})

        if annotations:
            save_metadata["annotations"] = annotations

        if self.save_quam_state and machine is not None:
            save_metadata["quam_state"] = machine.to_dict()

        qcodes_data = data.copy(deep=False)
        readout_result_names = save_metadata.get("readout_result_names", [])
        if "readout" in qcodes_data.dims and len(readout_result_names) == qcodes_data.sizes["readout"]:
            qcodes_data = qcodes_data.assign_coords(readout=readout_result_names)

        run_id = qcodes_save_xarray(
            qcodes_data,
            name=name,
            metadata=save_metadata,
        )

        return {
            "run_id": run_id,
        }

    def success_message(self, details=None):
        if isinstance(details, dict):
            run_id = details.get("run_id")
            if run_id is not None:
                return f"Data saved to QCoDeS database as run #{run_id}."

        return "Data saved successfully using QCoDeS."

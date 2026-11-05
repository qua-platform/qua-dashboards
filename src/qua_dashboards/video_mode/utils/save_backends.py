from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from qualang_tools.external_frameworks.qcodes.saving import save_xarray as qcodes_save_xarray
import xarray as xr

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

        save_data = {
            "data": data.values,
            "annotations": annotations or {},
            "metadata": dict(metadata or {}),
            "dims": list(data.dims),
        }

        # Save all axis coordinates explicitly
        for dim in data.dims:
            if dim in data.coords:
                save_data[f"coord_{dim}"] = data.coords[dim].values

        # Save DataArray attributes as well
        if data.attrs:
            save_data["data_attrs"] = dict(data.attrs)

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

        run_id = qcodes_save_xarray(
            data,
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
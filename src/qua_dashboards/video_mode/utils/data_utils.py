from typing import Dict, Any, Optional
import logging
from qua_dashboards.video_mode import data_registry
from qualibrate import QualibrationNode
import xarray as xr


logger = logging.getLogger(__name__)


def dataarray_to_dataset(da: xr.DataArray, var_name: str = "data") -> xr.Dataset:
    """
    Convert an xarray.DataArray to an xarray.Dataset with a single variable.
    """
    if not isinstance(da, xr.DataArray):
        raise TypeError("Input must be an xarray.DataArray")
    return da.to_dataset(name=var_name)


def dataset_to_dataarray(
    ds: xr.Dataset, var_name: Optional[str] = None
) -> xr.DataArray:
    """
    Convert an xarray.Dataset with a single variable to a DataArray.
    If var_name is not provided, use the first variable in the dataset.
    """
    if not isinstance(ds, xr.Dataset):
        raise TypeError("Input must be an xarray.Dataset")
    if var_name is None:
        var_name = list(ds.data_vars)[0]
    return ds[var_name]


def save_data(results: Dict[str, Any]) -> Dict[str, Any]:
    node = QualibrationNode("2D_video_mode")

    # Convert any DataArray in results to Dataset for saving
    # Recursively convert any DataArray values in the dict
    for k, v in results["image_data"].items():
        if isinstance(v, xr.DataArray):
            results["image_data"][k] = dataarray_to_dataset(v)

    node.results = results
    node.save()

    try:
        storage_manager = node._get_storage_manager()
        path = storage_manager.data_handler.path
    except Exception:
        path = None

    return {
        "idx": node.snapshot_idx,
        "path": path,
    }


def load_data(idx: str) -> Dict[str, Any]:
    node = QualibrationNode.load_from_id(idx)
    results = node.results["image_data"]
    for k, v in results.items():
        if isinstance(v, xr.Dataset):
            results[k] = dataset_to_dataarray(v)

    return results

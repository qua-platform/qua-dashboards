import xarray as xr
import warnings


def serialise_data(data):
    if isinstance(data, dict):
        return {key: serialise_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [serialise_data(item) for item in data]
    elif isinstance(data, xr.DataArray):
        dict_data = data.to_dict()
        dict_data["__class__"] = "xarray.DataArray"
        return dict_data
    else:
        return data


def deserialise_data(data):
    if isinstance(data, dict):
        if "__class__" not in data:
            return {key: deserialise_data(value) for key, value in data.items()}

        cls_str = data.pop("__class__")
        if cls_str == "xarray.DataArray":
            return xr.DataArray.from_dict(data)
        else:
            warnings.warn(f"Unknown class: {cls_str}")
            return data
    elif isinstance(data, list):
        return [deserialise_data(item) for item in data]
    else:
        return data

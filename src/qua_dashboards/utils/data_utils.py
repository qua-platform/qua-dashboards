import xarray as xr
import warnings
import matplotlib.pyplot as plt
import io
import base64


__all__ = ["serialise_data", "deserialise_data"]


def serialise_data(data):
    if isinstance(data, dict):
        return {key: serialise_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [serialise_data(item) for item in data]
    elif isinstance(data, xr.DataArray):
        dict_data = data.to_dict()
        dict_data["__class__"] = "xarray.DataArray"
        return dict_data
    elif isinstance(data, xr.Dataset):
        dict_data = data.to_dict()
        dict_data["__class__"] = "xarray.Dataset"
        return dict_data
    elif isinstance(data, plt.Figure):
        buf = io.BytesIO()
        data.savefig(buf, format="png")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        return {
            "__class__": "image_base64",
            "data": f"data:image/png;base64,{image_base64}",
        }
    else:
        return data


def deserialise_data(data):
    if isinstance(data, dict):
        if "__class__" not in data:
            return {key: deserialise_data(value) for key, value in data.items()}

        cls_str = data.pop("__class__")
        if cls_str == "xarray.DataArray":
            return xr.DataArray.from_dict(data)
        elif cls_str == "xarray.Dataset":
            return xr.Dataset.from_dict(data)
        elif cls_str == "image_base64":
            image_base64 = data["data"]
            return image_base64
        else:
            warnings.warn(f"Unknown class: {cls_str}")
            return data
    elif isinstance(data, list):
        return [deserialise_data(item) for item in data]
    else:
        return data

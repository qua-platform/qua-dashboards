import logging
import threading
import uuid
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Simple in-memory cache for single-process Dash applications.
# For multi-process (e.g., Gunicorn with multiple workers), an external
# cache like Redis or a file-system based cache would be necessary.
_DATA_CACHE: Dict[str, Any] = {}
_DATA_VERSIONS: Dict[str, str] = {}  # Stores current version UUID for a key
_CACHE_LOCK: threading.Lock = threading.Lock()

# Define common keys for data stored in the registry
LIVE_DATA_KEY = "live_data"
STATIC_DATA_KEY = "static_data"


def set_data(key: str, data_object: Any) -> str:
    """
    Stores a data object in the registry and generates a new version ID.

    Args:
        key: The key under which to store the data object.
        data_object: The Python object to store.

    Returns:
        The new version UUID string for the stored data.
    """
    new_version = str(uuid.uuid4())
    with _CACHE_LOCK:
        _DATA_CACHE[key] = data_object
        _DATA_VERSIONS[key] = new_version
        logger.debug(
            f"Data set in registry for key '{key}' with new version '{new_version}'"
        )
    return new_version


def get_data(key: str) -> Optional[Any]:
    """
    Retrieves a data object from the registry.

    Args:
        key: The key of the data object to retrieve.

    Returns:
        The stored Python object, or None if the key is not found.
    """
    with _CACHE_LOCK:
        data_object = _DATA_CACHE.get(key)
        if data_object is None:
            logger.warning(f"Data not found in registry for key '{key}'")
        else:
            logger.debug(f"Data retrieved from registry for key '{key}'")
        return data_object


def get_current_version(key: str) -> Optional[str]:
    """
    Retrieves the current version ID for a data key.

    Args:
        key: The key for which to get the version.

    Returns:
        The current version UUID string, or None if the key is not found.
    """
    with _CACHE_LOCK:
        return _DATA_VERSIONS.get(key)

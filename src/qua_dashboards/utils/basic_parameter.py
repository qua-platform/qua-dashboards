import logging
import time
from typing import Callable, Optional, TYPE_CHECKING
from qua_dashboards.core import ParameterProtocol

__all__ = ["BasicParameter"]


class BasicParameter:
    def __init__(self, name:str, label:Optional[str]=None, units:str="V", initial_value:float=0.0):
        self.name = name
        self.label = label if label is not None else name
        self.latest_value = initial_value
        self._value = initial_value
        self.unit = units
        logging.debug(
            f"{self.name} initialized with value {self.latest_value} {self.unit}"
        )

    def get(self):
        time.sleep(0.2)  # Simulate a 200ms delay
        self.latest_value = self._value
        logging.debug(f"Getting {self.name}: {self.latest_value} {self.unit}")
        return self.latest_value

    def set(self, new_value):
        self._value = new_value
        updated_value = self.get()  # Return the value after setting
        logging.debug(
            f"Setting {self.name} to {new_value}: Actual value is {updated_value} {self.unit}"
        )
        return updated_value

    def get_latest(self):
        return self.latest_value

if TYPE_CHECKING:
    def check(name: str):
        _:ParameterProtocol = BasicParameter(name)
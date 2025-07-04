
from qua_dashboards.core import ParameterProtocol
from typing import Callable, TYPE_CHECKING

__all__ = ["CallbackParameter"]


class CallbackParameter:
    """ Calls a callback function when the value of this parameter is changed. The callback takes the parameter changed as argument."""

    def __init__(self, parameter: ParameterProtocol, callback_on_set: Callable[[ParameterProtocol],None]):
        self.parameter = parameter
        self.callback = callback_on_set
        
    @property
    def name(self) -> str:
        return self.parameter.name

    @property
    def label(self) -> str:
        return self.parameter.label
        
    @property
    def unit(self) -> str:
        return self.parameter.unit

    def get_latest(self) -> float:
        return self.parameter.get_latest()
    
    def get(self) -> float:
        return self.parameter.get()

    def set(self, value: float) -> None:
        self.parameter.set(value)
        self.callback(self.parameter)
        
if TYPE_CHECKING:
    def check(parameter: ParameterProtocol, callback_on_set: Callable[[ParameterProtocol], None]):
        _: ParameterProtocol = CallbackParameter(parameter, callback_on_set)
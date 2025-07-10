
from qua_dashboards.core import ParameterProtocol
from typing import Callable, TYPE_CHECKING

__all__ = ["CallbackParameter"]


class CallbackParameter:
    """ Calls a callback function when the value of this parameter is changed. The callback takes the parameter changed as argument."""

    def __init__(self, parameter: ParameterProtocol, callback_on_set: Callable[[ParameterProtocol, float],None]):
        """ Wraps the given parameter with a class that calls a given callback when set() is called. 
        The callback is run after parameter.set() is called.
        
        Parameters:
        parameter: the parameter to which to add the callback
        callback_on_set: A callback that takes as argument parameter and the old value 
        """
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
        old_value = self.parameter.get_latest()
        self.parameter.set(value)
        self.callback(self.parameter,old_value)
        
if TYPE_CHECKING:
    def check(parameter: ParameterProtocol, callback_on_set: Callable[[ParameterProtocol], None]):
        _: ParameterProtocol = CallbackParameter(parameter, callback_on_set)
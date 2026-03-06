from dataclasses import dataclass
from typing import Union

@dataclass
class OPXOffsetParameter:
    element_name: str
    qm_provider: object = None
    _current_voltage: float = 0.0
    input: str = "single"

    def current_voltage(self, voltage: float = None) -> Union[float, None]: 
        if voltage is None: 
            return self._current_voltage
        self._current_voltage = voltage
        qm = getattr(self.qm_provider, "qm", None)
        if qm is not None: 
            qm.set_output_dc_offset_by_element(self.element_name, self.input, voltage)
        return None

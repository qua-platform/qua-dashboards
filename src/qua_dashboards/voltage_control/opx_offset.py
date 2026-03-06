import logging
from dataclasses import dataclass
from typing import Union

logger = logging.getLogger(__name__)


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

    def reapply(self) -> None:
        """Re-apply the stored DC offset to the hardware."""
        if self._current_voltage == 0.0:
            return
        qm = getattr(self.qm_provider, "qm", None)
        if qm is not None:
            qm.set_output_dc_offset_by_element(self.element_name, self.input, self._current_voltage)

from dash import Dash
from qua_dashboards.core import BaseUpdatableComponent
from qua_dashboards.utils.qua_types import QuaVariableFloat


from abc import ABC, abstractmethod
from typing import Tuple


class InnerLoopAction(BaseUpdatableComponent, ABC):
    def __init__(self, component_id: str = "inner-loop"):
        super().__init__(component_id=component_id)

    @abstractmethod
    def __call__(
        self, x: QuaVariableFloat, y: QuaVariableFloat
    ) -> Tuple[QuaVariableFloat, QuaVariableFloat]:
        pass

    def get_layout(self):
        return None

    def register_callbacks(self, app: Dash):
        pass

    def initial_action(self):
        pass

    def final_action(self):
        pass

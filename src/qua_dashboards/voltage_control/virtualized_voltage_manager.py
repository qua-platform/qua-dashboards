from typing import List, Optional
import numpy as np
from quam_builder.architecture.quantum_dots.components import VirtualGateSet

__all__ = ["VirtualizedVoltageManager", "VirtualizedVoltageParameter"]

class VirtualizedVoltageManager:


    def __init__(
        self, 
        physical_parameters: List,
        virtual_names: List[str], 
        gate_set:VirtualGateSet, 
        unit: str = "V", 
        use_pseudoinverse: bool = True, 
    ): 

        self._physical_parameters = physical_parameters
        self._virtual_names = virtual_names
        self._unit = unit
        self.use_pseudoinverse = use_pseudoinverse
        self.gate_set = gate_set

        self.first_layer = self.gate_set.layers[0]
        self.full_physical_list = self.first_layer.target_gates
        self.full_virtual_list = self.first_layer.source_gates

        virtual_indices = [self.full_virtual_list.index(gate) for gate in virtual_names]
        physical_names = [p.name for p in physical_parameters]
        physical_indices = [self.full_physical_list.index(name) for name in physical_names]

        full_matrix = np.array(self.first_layer.matrix, dtype=float)
        self.matrix = full_matrix[np.ix_(virtual_indices, physical_indices)]

        n_physical = len(self._physical_parameters)
        n_virtual = len(self._virtual_names)

        # Inverse
        expected_shape = (n_physical, n_virtual)
        try: 
            matrix_is_square = self.matrix.shape[0] == self.matrix.shape[1]
            if matrix_is_square and not self.use_pseudoinverse: 
                self.inv_matrix = np.linalg.inv(self.matrix)
            else: 
                self.inv_matrix = np.linalg.pinv(self.matrix)
            if self.inv_matrix.shape != expected_shape: 
                raise ValueError(
                    "Inverse matrix has incorrect dimensions. "
                    f"Expected {expected_shape[0]}x{expected_shape[1]}, "
                    f"got {self.inv_matrix.shape}."
                )
        except Exception as e:
            raise ValueError(f"Error calculating inverse matrix: {e}")


        self._virtual_parameters = [
            VirtualizedVoltageParameter(
                name = name, 
                label = name, 
                unit = self._unit, 
                virtual_index = i, 
                manager = self
            ) for i, name in enumerate(self._virtual_names)
        ]

    def get_virtual_offset_parameter(self, virtual_gate_name: str): 
        if virtual_gate_name not in self._virtual_names:
            return None
        parameter =  self._virtual_parameters[self._virtual_names.index(virtual_gate_name)]
        return parameter

    def get_physical_voltages(self) -> np.ndarray: 
        v_list = []
        for param in self._physical_parameters: 
            v = param.get_latest()
            if v is None: 
                v = 0.0
            v_list.append(v)
        return np.array(v_list, dtype = float)
    
    def get_all_virtual_voltages(self) -> np.ndarray: 
        physical_voltages = self.get_physical_voltages()
        return np.array(self.inv_matrix @ physical_voltages, dtype = float)

    def get_virtual_voltage(self, virtual_index: int): 
        return self.get_all_virtual_voltages()[virtual_index]

    def set_virtual_voltage(self, index: int, value: float) -> None: 
        current_virtual = self.get_all_virtual_voltages()
        current_virtual[index] = value
        new_physical = self.matrix @ current_virtual

        for param, physical_value in zip(self._physical_parameters, new_physical): 
            pv = float(physical_value)
            param.set(pv)

    def get_virtual_parameters(self) -> List["VirtualizedVoltageParameter"]: 
        return self._virtual_parameters
        

class VirtualizedVoltageParameter: 

    def __init__(
        self, 
        name: str, 
        label: str,
        unit: str, 
        virtual_index: int, 
        manager: VirtualizedVoltageManager
    ): 
        self.name = name
        self.label = label
        self.unit = unit
        self.virtual_index = virtual_index
        self.manager = manager
        
    def get_latest(self) -> float: 
        return self.manager.get_virtual_voltage(self.virtual_index)
    
    def set(self, value: float) -> None: 
        self.manager.set_virtual_voltage(self.virtual_index, value)
        
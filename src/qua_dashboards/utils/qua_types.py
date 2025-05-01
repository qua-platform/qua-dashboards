__all__ = [
    "QuaVariableFloat",
]

try:
    from qm.qua.type_hints import QuaVariable

    QuaVariableFloat = QuaVariable[float]

except ImportError:
    from qm.qua._dsl import QuaVariableType

    QuaVariableFloat = QuaVariableType

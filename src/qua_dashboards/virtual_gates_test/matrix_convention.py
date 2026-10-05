import numpy as np

def to_apply_matrix(quam_matrix):
    """QuAM stored M (source x target) -> apply matrix in the same layout.

    resolve_voltages uses V_phys = inv(M) @ V_virt. The GUI shows the apply
    coefficients: apply[s, t] = inv(M)[t, s], i.e. inv(M).T.

    Matrix contains staightforward virtual vectors, e.g., row [vP1] = 1*[P1] + (-0.2)*[sensor_gate]
    """
    inv_m = np.linalg.inv(np.asarray(quam_matrix, dtype=float))
    return inv_m.T.tolist()


def to_quam_matrix(apply_matrix):
    """GUI apply matrix -> QuAM stored M for layer.matrix.

    apply is source x target; inv(M) = apply.T, so M = inv(apply.T).
    """
    apply_T = np.asarray(apply_matrix, dtype=float).T
    return np.linalg.inv(apply_T).tolist()
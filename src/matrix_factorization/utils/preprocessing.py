import numpy as np


def normalize_matrix(A):
    """
    Normalize a matrix to the range [0, 1].
    """

    A = np.asarray(A, dtype=float)

    min_value = np.min(A)
    max_value = np.max(A)

    if max_value == min_value:
        return np.zeros_like(A)

    return (A - min_value) / (max_value - min_value)


def check_matrix(A):
    """
    Validate that the input is a 2D numeric matrix.
    """

    A = np.asarray(A)

    if A.ndim != 2:
        raise ValueError("Input must be a 2D matrix.")

    if not np.issubdtype(A.dtype, np.number):
        raise TypeError("Matrix must contain numeric values.")

    return True

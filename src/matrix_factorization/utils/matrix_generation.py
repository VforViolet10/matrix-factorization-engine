import numpy as np


def generate_random_matrix(rows, cols, seed=None):
    """
    Generate a random matrix.

    Parameters
    ----------
    rows : int
        Number of rows.
    cols : int
        Number of columns.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Generated random matrix.
    """

    if rows <= 0 or cols <= 0:
        raise ValueError("Rows and columns must be positive.")

    rng = np.random.default_rng(seed)

    return rng.random((rows, cols))


def generate_low_rank_matrix(rows, cols, rank, seed=None):
    """
    Generate a matrix with approximately specified rank.

    A = U @ V

    Parameters
    ----------
    rows : int
        Number of rows.
    cols : int
        Number of columns.
    rank : int
        Desired rank.
    seed : int, optional
        Random seed.

    Returns
    -------
    numpy.ndarray
        Low-rank matrix.
    """

    if rank <= 0:
        raise ValueError("Rank must be positive.")

    if rank > min(rows, cols):
        raise ValueError(
            "Rank cannot be greater than min(rows, cols)."
        )

    rng = np.random.default_rng(seed)

    U = rng.random((rows, rank))
    V = rng.random((rank, cols))

    return U @ V

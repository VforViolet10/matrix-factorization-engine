import numpy as np


def frobenius_norm(A):
    """
    Calculate the Frobenius norm of a matrix.

    Parameters
    ----------
    A : array-like
        Input matrix.

    Returns
    -------
    float
        Frobenius norm of A.
    """

    A = np.asarray(A, dtype=float)

    if A.ndim != 2:
        raise ValueError("Input must be a 2-dimensional matrix.")

    return np.linalg.norm(A, ord="fro")


def reconstruction_error(A, A_hat):
    """
    Calculate the absolute reconstruction error.

    E = ||A - A_hat||_F

    Parameters
    ----------
    A : array-like
        Original matrix.

    A_hat : array-like
        Reconstructed matrix.

    Returns
    -------
    float
        Frobenius reconstruction error.
    """

    A = np.asarray(A, dtype=float)
    A_hat = np.asarray(A_hat, dtype=float)

    if A.shape != A_hat.shape:
        raise ValueError(
            "A and A_hat must have the same shape."
        )

    return np.linalg.norm(
        A - A_hat,
        ord="fro"
    )


def relative_reconstruction_error(A, A_hat):
    """
    Calculate the relative reconstruction error.

    E_relative = ||A - A_hat||_F / ||A||_F

    Parameters
    ----------
    A : array-like
        Original matrix.

    A_hat : array-like
        Reconstructed matrix.

    Returns
    -------
    float
        Relative reconstruction error.
    """

    A = np.asarray(A, dtype=float)
    A_hat = np.asarray(A_hat, dtype=float)

    if A.shape != A_hat.shape:
        raise ValueError(
            "A and A_hat must have the same shape."
        )

    original_norm = np.linalg.norm(
        A,
        ord="fro"
    )

    if original_norm == 0:
        raise ValueError(
            "Relative error is undefined for a zero matrix."
        )

    return (
        np.linalg.norm(
            A - A_hat,
            ord="fro"
        )
        / original_norm
    )


def rmse(A, A_hat):
    """
    Calculate the Root Mean Squared Error.

    Parameters
    ----------
    A : array-like
        Original matrix.

    A_hat : array-like
        Reconstructed matrix.

    Returns
    -------
    float
        Root Mean Squared Error.
    """

    A = np.asarray(A, dtype=float)
    A_hat = np.asarray(A_hat, dtype=float)

    if A.shape != A_hat.shape:
        raise ValueError(
            "A and A_hat must have the same shape."
        )

    return np.sqrt(
        np.mean((A - A_hat) ** 2)
    )

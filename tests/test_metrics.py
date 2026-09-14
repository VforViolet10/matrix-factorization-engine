import numpy as np
import pytest

from matrix_factorization.metrics import (
    frobenius_norm,
    reconstruction_error,
    relative_reconstruction_error,
    rmse,
)


def test_frobenius_norm():
    A = np.array([
        [3, 4],
        [0, 0]
    ])

    assert np.isclose(
        frobenius_norm(A),
        5.0
    )


def test_reconstruction_error():
    A = np.array([
        [1, 2],
        [3, 4]
    ])

    A_hat = np.array([
        [1, 2],
        [3, 3]
    ])

    assert np.isclose(
        reconstruction_error(A, A_hat),
        1.0
    )


def test_relative_reconstruction_error():
    A = np.array([
        [3, 4],
        [0, 0]
    ])

    A_hat = np.array([
        [3, 3],
        [0, 0]
    ])

    expected = 1 / 5

    assert np.isclose(
        relative_reconstruction_error(A, A_hat),
        expected
    )


def test_rmse():
    A = np.array([
        [1, 2],
        [3, 4]
    ])

    A_hat = np.array([
        [1, 2],
        [3, 3]
    ])

    expected = 0.5

    assert np.isclose(
        rmse(A, A_hat),
        expected
    )


def test_shape_mismatch():
    A = np.ones((2, 2))
    A_hat = np.ones((3, 3))

    with pytest.raises(ValueError):
        reconstruction_error(A, A_hat)


def test_relative_error_zero_matrix():
    A = np.zeros((2, 2))
    A_hat = np.zeros((2, 2))

    with pytest.raises(ValueError):
        relative_reconstruction_error(A, A_hat)


def test_invalid_dimension():
    A = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        frobenius_norm(A)

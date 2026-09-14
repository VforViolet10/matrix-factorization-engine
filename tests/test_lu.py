import numpy as np
import pytest

from matrix_factorization.decompositions.lu import LU


def test_lu_shapes():
    """Test the dimensions of P, L, and U."""

    A = np.array([
        [4, 3],
        [6, 3]
    ], dtype=float)

    lu = LU().fit(A)

    assert lu.P.shape == (2, 2)
    assert lu.L.shape == (2, 2)
    assert lu.U.shape == (2, 2)


def test_l_is_lower_triangular():
    """Test that L is lower triangular."""

    A = np.array([
        [4, 3],
        [6, 3]
    ], dtype=float)

    lu = LU().fit(A)

    assert np.allclose(
        lu.L,
        np.tril(lu.L)
    )


def test_l_has_unit_diagonal():
    """Test that L has ones on its diagonal."""

    A = np.array([
        [4, 3],
        [6, 3]
    ], dtype=float)

    lu = LU().fit(A)

    assert np.allclose(
        np.diag(lu.L),
        np.ones(2)
    )


def test_u_is_upper_triangular():
    """Test that U is upper triangular."""

    A = np.array([
        [4, 3],
        [6, 3]
    ], dtype=float)

    lu = LU().fit(A)

    assert np.allclose(
        lu.U,
        np.triu(lu.U)
    )


def test_lu_reconstruction():
    """Test the fundamental relation P @ A = L @ U."""

    A = np.array([
        [4, 3],
        [6, 3]
    ], dtype=float)

    lu = LU().fit(A)

    assert np.allclose(
        lu.P @ A,
        lu.L @ lu.U
    )


def test_p_is_permutation_matrix():
    """Test that P is a valid permutation matrix."""

    A = np.array([
        [4, 3],
        [6, 3]
    ], dtype=float)

    lu = LU().fit(A)

    assert np.allclose(
        lu.P @ lu.P.T,
        np.eye(2)
    )


def test_pivoting():
    """
    Test that partial pivoting occurs when the first
    pivot is smaller than a later candidate.
    """

    A = np.array([
        [1e-12, 1],
        [1, 1]
    ], dtype=float)

    lu = LU().fit(A)

    assert np.allclose(
        lu.P @ A,
        lu.L @ lu.U
    )


def test_invalid_non_square_matrix():
    """Test that non-square matrices are rejected."""

    A = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    with pytest.raises(ValueError):
        LU().fit(A)


def test_invalid_one_dimensional_input():
    """Test that one-dimensional input is rejected."""

    A = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        LU().fit(A)


def test_singular_matrix():
    """Test that singular matrices are rejected."""

    A = np.array([
        [1, 2],
        [2, 4]
    ], dtype=float)

    with pytest.raises(ValueError):
        LU().fit(A)

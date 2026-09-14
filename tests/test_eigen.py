import numpy as np
import pytest

from matrix_factorization.decompositions.eigen import (
    EigenvalueDecomposition
)


def test_eigenvalue_shapes():
    """Test eigenvalue and eigenvector dimensions."""

    A = np.array([
        [4, 1],
        [2, 3]
    ])

    eigen = EigenvalueDecomposition().fit(A)

    assert eigen.eigenvalues.shape == (2,)
    assert eigen.eigenvectors.shape == (2, 2)


def test_known_eigenvalues():
    """Test eigenvalues using a matrix with known eigenvalues."""

    A = np.array([
        [4, 1],
        [2, 3]
    ])

    eigen = EigenvalueDecomposition().fit(A)

    expected = np.array([5, 2])

    assert np.allclose(
        np.sort(eigen.eigenvalues),
        np.sort(expected)
    )


def test_eigenvector_equation():
    """
    Test the fundamental eigenvector equation:

        A @ v = lambda @ v
    """

    A = np.array([
        [4, 1],
        [2, 3]
    ])

    eigen = EigenvalueDecomposition().fit(A)

    for i in range(len(eigen.eigenvalues)):
        eigenvalue = eigen.eigenvalues[i]
        eigenvector = eigen.eigenvectors[:, i]

        left = A @ eigenvector
        right = eigenvalue * eigenvector

        assert np.allclose(left, right)


def test_reconstruction():
    """Test reconstruction of the original matrix."""

    A = np.array([
        [4, 1],
        [2, 3]
    ])

    eigen = EigenvalueDecomposition().fit(A)

    reconstructed = eigen.reconstruct()

    assert np.allclose(
        reconstructed,
        A
    )


def test_eigenvalue_matrix():
    """Test that Lambda is diagonal."""

    A = np.array([
        [4, 1],
        [2, 3]
    ])

    eigen = EigenvalueDecomposition().fit(A)

    Lambda = eigen.eigenvalue_matrix

    assert np.allclose(
        Lambda,
        np.diag(np.diag(Lambda))
    )


def test_invalid_non_square_matrix():
    """Test that non-square matrices are rejected."""

    A = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    with pytest.raises(ValueError):
        EigenvalueDecomposition().fit(A)


def test_invalid_one_dimensional_input():
    """Test that one-dimensional input is rejected."""

    A = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        EigenvalueDecomposition().fit(A)

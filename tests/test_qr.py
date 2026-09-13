import numpy as np
import pytest

from matrix_factorization.decompositions.qr import QR


def test_qr_shapes():
    """Test the dimensions of Q and R."""

    A = np.random.default_rng(42).random((5, 3))

    qr = QR().fit(A)

    assert qr.Q.shape == (5, 3)
    assert qr.R.shape == (3, 3)


def test_q_is_orthonormal():
    """Test that Q has orthonormal columns."""

    A = np.random.default_rng(42).random((5, 3))

    qr = QR().fit(A)

    assert np.allclose(
        qr.Q.T @ qr.Q,
        np.eye(3),
        atol=1e-10
    )


def test_r_is_upper_triangular():
    """Test that R is upper triangular."""

    A = np.random.default_rng(42).random((5, 3))

    qr = QR().fit(A)

    assert np.allclose(
        qr.R,
        np.triu(qr.R)
    )


def test_qr_reconstruction():
    """Test that Q @ R reconstructs the original matrix."""

    A = np.random.default_rng(42).random((5, 3))

    qr = QR().fit(A)

    reconstructed = qr.reconstruct()

    assert np.allclose(
        reconstructed,
        A,
        atol=1e-10
    )


def test_orthogonality_error():
    """Test that Q has a very small orthogonality error."""

    A = np.random.default_rng(42).random((5, 3))

    qr = QR().fit(A)

    assert qr.orthogonality_error() < 1e-10


def test_rank_deficient_matrix():
    """Test that rank-deficient matrices are rejected."""

    A = np.array([
        [1, 2, 3],
        [2, 4, 6],
        [3, 6, 9],
        [4, 8, 12]
    ])

    with pytest.raises(ValueError):
        QR().fit(A)


def test_invalid_input():
    """Test that one-dimensional input is rejected."""

    A = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        QR().fit(A)

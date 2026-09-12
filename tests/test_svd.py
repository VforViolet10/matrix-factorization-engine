import numpy as np
import pytest

from matrix_factorization.decompositions import SVD


def test_svd_reconstruction():
    A = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=float)

    model = SVD()
    model.fit(A)

    reconstructed = model.reconstruct()

    assert np.allclose(A, reconstructed)


def test_svd_reduced_rank():
    A = np.random.default_rng(42).random((10, 8))

    model = SVD(n_components=3)
    model.fit(A)

    assert model.U.shape == (10, 3)
    assert model.singular_values.shape == (3,)
    assert model.Vt.shape == (3, 8)


def test_svd_invalid_components():
    A = np.random.default_rng(42).random((5, 4))

    model = SVD(n_components=10)

    with pytest.raises(ValueError):
        model.fit(A)


def test_explained_variance_ratio():
    A = np.random.default_rng(42).random((10, 5))

    model = SVD()
    model.fit(A)

    ratios = model.explained_variance_ratio()

    assert np.isclose(np.sum(ratios), 1.0)

import numpy as np
import pytest

from matrix_factorization.decompositions.incremental_svd import IncrementalSVD


def test_incremental_svd_fit():
    X = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])

    model = IncrementalSVD(n_components=2)
    model.fit(X)

    assert model.U_.shape == (3, 2)
    assert model.s_.shape == (2,)
    assert model.Vt_.shape == (2, 3)


def test_incremental_svd_reconstruction():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    model = IncrementalSVD(n_components=2)
    model.fit(X)

    reconstructed = model.reconstruct()

    assert reconstructed.shape == X.shape
    assert np.allclose(reconstructed, X, atol=1e-10)


def test_incremental_svd_partial_fit():
    X_initial = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    X_new = np.array([
        [5.0, 6.0],
        [7.0, 8.0],
    ])

    model = IncrementalSVD(n_components=2)

    model.fit(X_initial)
    model.partial_fit(X_new)

    assert model.n_samples_seen_ == 4
    assert model.reconstruct().shape == (4, 2)


def test_incremental_svd_transform():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    model = IncrementalSVD(n_components=1)
    model.fit(X)

    transformed = model.transform(X)

    assert transformed.shape == (3, 1)


def test_incremental_svd_explained_variance():
    X = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])

    model = IncrementalSVD(n_components=2)
    model.fit(X)

    ratios = model.explained_variance_ratio()

    assert np.isclose(np.sum(ratios), 1.0)


def test_incremental_svd_invalid_components():
    with pytest.raises(ValueError):
        IncrementalSVD(n_components=0)


def test_incremental_svd_invalid_input():
    model = IncrementalSVD(n_components=2)

    with pytest.raises(ValueError):
        model.fit(np.array([1.0, 2.0, 3.0]))

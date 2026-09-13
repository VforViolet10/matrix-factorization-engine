import numpy as np
import pytest

from matrix_factorization.decompositions import NMF


def test_nmf_shapes():
    A = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [2, 4, 6]
    ], dtype=float)

    model = NMF(
        n_components=2,
        max_iter=500,
        random_state=42
    )

    model.fit(A)

    assert model.W.shape == (4, 2)
    assert model.H.shape == (2, 3)


def test_nmf_non_negative_factors():
    A = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=float)

    model = NMF(
        n_components=2,
        random_state=42
    )

    model.fit(A)

    assert np.all(model.W >= 0)
    assert np.all(model.H >= 0)


def test_nmf_reconstruction():
    A = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=float)

    model = NMF(
        n_components=3,
        max_iter=1000,
        random_state=42
    )

    model.fit(A)

    reconstructed = model.reconstruct()

    assert reconstructed.shape == A.shape
    assert np.all(reconstructed >= 0)


def test_nmf_rejects_negative_values():
    A = np.array([
        [1, 2],
        [-1, 4]
    ], dtype=float)

    model = NMF(n_components=2)

    with pytest.raises(ValueError):
        model.fit(A)


def test_nmf_rejects_invalid_components():
    A = np.ones((3, 4))

    model = NMF(n_components=5)

    with pytest.raises(ValueError):
        model.fit(A)


def test_nmf_error_is_recorded():
    A = np.random.default_rng(42).random((10, 6))

    model = NMF(
        n_components=3,
        max_iter=100,
        random_state=42
    )

    model.fit(A)

    assert model.reconstruction_error_ >= 0
    assert len(model.error_history_) > 0
    assert model.n_iter_ > 0

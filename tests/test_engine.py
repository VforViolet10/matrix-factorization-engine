import numpy as np
import pytest

from matrix_factorization.engine import MatrixFactorizationEngine


def test_available_methods():
    methods = MatrixFactorizationEngine.available_methods()

    assert "svd" in methods
    assert "nmf" in methods
    assert "qr" in methods
    assert "eigen" in methods
    assert "lu" in methods
    assert "incremental_svd" in methods
    assert "robust_pca" in methods


def test_invalid_method():
    with pytest.raises(ValueError):
        MatrixFactorizationEngine(method="invalid")


def test_svd_engine():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    engine = MatrixFactorizationEngine(
        method="svd",
        n_components=2,
    )

    result = engine.fit(X)

    assert result is engine
    assert engine.get_model() is engine.model


def test_incremental_svd_engine():
    X_initial = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    X_new = np.array([
        [5.0, 6.0],
    ])

    engine = MatrixFactorizationEngine(
        method="incremental_svd",
        n_components=2,
    )

    engine.fit(X_initial)
    engine.partial_fit(X_new)

    assert engine.model.n_samples_seen_ == 3


def test_robust_pca_engine():
    X = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
    ])

    engine = MatrixFactorizationEngine(
        method="robust_pca",
        max_iter=500,
    )

    engine.fit(X)

    reconstructed = engine.reconstruct()

    assert reconstructed.shape == X.shape
    assert np.allclose(reconstructed, X, atol=1e-5)


def test_qr_engine():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    engine = MatrixFactorizationEngine(method="qr")
    engine.fit(X)

    reconstructed = engine.reconstruct()

    assert reconstructed.shape == X.shape
    assert np.allclose(reconstructed, X, atol=1e-10)


def test_eigen_engine():
    X = np.array([
        [4.0, 1.0],
        [1.0, 3.0],
    ])

    engine = MatrixFactorizationEngine(method="eigen")
    engine.fit(X)

    reconstructed = engine.reconstruct()

    assert reconstructed.shape == X.shape
    assert np.allclose(reconstructed, X, atol=1e-10)


def test_lu_engine():
    X = np.array([
        [4.0, 3.0],
        [6.0, 3.0],
    ])

    engine = MatrixFactorizationEngine(method="lu")
    engine.fit(X)

    reconstructed = engine.reconstruct()

    assert reconstructed.shape == X.shape
    assert np.allclose(reconstructed, X, atol=1e-10)

def test_transform_requires_support():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    engine = MatrixFactorizationEngine(method="lu")
    engine.fit(X)

    with pytest.raises(NotImplementedError):
        engine.transform(X)


def test_partial_fit_requires_incremental_svd():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    engine = MatrixFactorizationEngine(method="svd")
    engine.fit(X)

    with pytest.raises(NotImplementedError):
        engine.partial_fit(X)


def test_engine_requires_fit():
    engine = MatrixFactorizationEngine(method="svd")

    with pytest.raises(RuntimeError):
        engine.reconstruct()


def test_invalid_matrix():
    engine = MatrixFactorizationEngine(method="svd")

    with pytest.raises(ValueError):
        engine.fit(np.array([1.0, 2.0, 3.0]))

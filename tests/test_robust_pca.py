import numpy as np
import pytest

from matrix_factorization.decompositions.robust_pca import RobustPCA


def test_robust_pca_decomposes_matrix():
    rng = np.random.default_rng(42)

    low_rank = np.outer(
        rng.normal(size=8),
        rng.normal(size=6),
    )

    sparse = np.zeros((8, 6))
    sparse[1, 2] = 8.0
    sparse[5, 4] = -6.0

    X = low_rank + sparse

    model = RobustPCA(max_iter=500, tol=1e-6)
    model.fit(X)

    assert model.L_.shape == X.shape
    assert model.S_.shape == X.shape
    assert model.n_iter_ > 0


def test_robust_pca_reconstruction():
    rng = np.random.default_rng(42)

    low_rank = np.outer(
        rng.normal(size=6),
        rng.normal(size=5),
    )

    sparse = np.zeros((6, 5))
    sparse[2, 1] = 10.0

    X = low_rank + sparse

    model = RobustPCA(max_iter=1000, tol=1e-6)
    model.fit(X)

    reconstructed = model.reconstruct()

    assert reconstructed.shape == X.shape
    assert np.allclose(reconstructed, X, atol=1e-4)


def test_robust_pca_reconstruction_error():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    model = RobustPCA(max_iter=500)
    model.fit(X)

    error = model.reconstruction_error(X)

    assert error >= 0
    assert error < 1e-5


def test_robust_pca_rank():
    X = np.outer(
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 4.0]),
    )

    model = RobustPCA(max_iter=500)
    model.fit(X)

    assert model.rank() == 1


def test_robust_pca_sparsity():
    X = np.eye(4)

    model = RobustPCA(max_iter=500)
    model.fit(X)

    sparsity = model.sparsity()

    assert 0.0 <= sparsity <= 1.0


def test_robust_pca_zero_matrix():
    X = np.zeros((4, 4))

    model = RobustPCA()
    model.fit(X)

    assert np.allclose(model.L_, 0)
    assert np.allclose(model.S_, 0)
    assert model.reconstruction_error(X) == 0.0


def test_robust_pca_invalid_parameters():
    with pytest.raises(ValueError):
        RobustPCA(max_iter=0)

    with pytest.raises(ValueError):
        RobustPCA(tol=0)

    with pytest.raises(ValueError):
        RobustPCA(lam=0)

    with pytest.raises(ValueError):
        RobustPCA(mu=0)


def test_robust_pca_invalid_input():
    model = RobustPCA()

    with pytest.raises(ValueError):
        model.fit(np.array([1.0, 2.0, 3.0]))


def test_robust_pca_not_fitted():
    model = RobustPCA()

    with pytest.raises(RuntimeError):
        model.reconstruct()

    with pytest.raises(RuntimeError):
        model.rank()

    with pytest.raises(RuntimeError):
        model.sparsity()

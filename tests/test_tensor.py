import numpy as np

from matrix_factorization.decompositions.tensor import (
    TensorFactorization,
)


def test_tensor_factorization_reconstruction():
    rng = np.random.default_rng(42)

    tensor = rng.random((4, 4, 4))

    model = TensorFactorization(
        rank=2,
        max_iter=100,
        tol=1e-5,
        random_state=42,
    )

    model.fit(tensor)

    reconstructed = model.reconstruct_tensor()

    assert reconstructed.shape == tensor.shape
    assert np.all(np.isfinite(reconstructed))

    error = model.reconstruction_error(tensor)

    assert np.isfinite(error)
    assert error >= 0


def test_tensor_factorization_is_reproducible():
    rng = np.random.default_rng(42)

    tensor = rng.random((3, 3, 3))

    model_1 = TensorFactorization(
        rank=2,
        random_state=42,
    )

    model_2 = TensorFactorization(
        rank=2,
        random_state=42,
    )

    model_1.fit(tensor)
    model_2.fit(tensor)

    reconstruction_1 = model_1.reconstruct_tensor()
    reconstruction_2 = model_2.reconstruct_tensor()

    assert np.allclose(
        reconstruction_1,
        reconstruction_2,
    )


def test_tensor_factorization_rejects_invalid_rank():
    tensor = np.ones((3, 3, 3))

    model = TensorFactorization(
        rank=4,
    )

    try:
        model.fit(tensor)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_tensor_factorization_convergence_history():
    rng = np.random.default_rng(10)

    tensor = rng.random((4, 4, 4))

    model = TensorFactorization(
        rank=2,
        max_iter=20,
        random_state=10,
    )

    model.fit(tensor)

    assert len(model.errors_) > 0
    assert model.n_iter_ == len(model.errors_)

    assert all(
        error >= 0
        for error in model.errors_
    )

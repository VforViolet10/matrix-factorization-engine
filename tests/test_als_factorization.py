import numpy as np
import pytest

from applications.recommendation_system import (
    ALSMatrixFactorization,
    create_rating_matrix,
)


def test_model_initialization():
    model = ALSMatrixFactorization(
        n_components=2,
        random_state=42,
    )

    assert model.n_components == 2
    assert model.user_factors is None
    assert model.item_factors is None


def test_fit_shapes():
    ratings = create_rating_matrix()

    model = ALSMatrixFactorization(
        n_components=2,
        random_state=42,
    )

    model.fit(ratings)

    assert model.user_factors.shape == (
        ratings.shape[0],
        2,
    )

    assert model.item_factors.shape == (
        ratings.shape[1],
        2,
    )


def test_reconstruction_shape():
    ratings = create_rating_matrix()

    model = ALSMatrixFactorization(
        n_components=2,
        random_state=42,
    )

    model.fit(ratings)

    reconstructed = model.reconstruct()

    assert reconstructed.shape == ratings.shape


def test_predictions_are_finite():
    ratings = create_rating_matrix()

    model = ALSMatrixFactorization(
        n_components=2,
        random_state=42,
    )

    model.fit(ratings)

    predictions = model.reconstruct()

    assert np.all(
        np.isfinite(predictions)
    )


def test_predict_single_rating():
    ratings = create_rating_matrix()

    model = ALSMatrixFactorization(
        n_components=2,
        random_state=42,
    )

    model.fit(ratings)

    prediction = model.predict(
        0,
        0,
    )

    assert isinstance(
        prediction,
        float,
    )

    assert np.isfinite(
        prediction
    )


def test_loss_decreases():
    ratings = create_rating_matrix()

    model = ALSMatrixFactorization(
        n_components=2,
        regularization=0.1,
        max_iter=50,
        random_state=42,
    )

    model.fit(ratings)

    assert len(
        model.loss_history_
    ) > 1

    assert (
        model.loss_history_[-1]
        < model.loss_history_[0]
    )


def test_reproducibility():
    ratings = create_rating_matrix()

    model1 = ALSMatrixFactorization(
        n_components=2,
        random_state=42,
    )

    model2 = ALSMatrixFactorization(
        n_components=2,
        random_state=42,
    )

    model1.fit(ratings)
    model2.fit(ratings)

    assert np.allclose(
        model1.user_factors,
        model2.user_factors,
    )

    assert np.allclose(
        model1.item_factors,
        model2.item_factors,
    )


def test_reconstruction_error():
    ratings = create_rating_matrix()

    model = ALSMatrixFactorization(
        n_components=2,
        random_state=42,
    )

    model.fit(ratings)

    error = model.reconstruction_error(
        ratings
    )

    assert np.isfinite(error)
    assert error >= 0


@pytest.mark.parametrize(
    "parameter",
    [
        {"n_components": 0},
        {"regularization": -0.1},
        {"max_iter": 0},
        {"tolerance": -0.1},
    ],
)
def test_invalid_parameters(parameter):
    with pytest.raises(ValueError):
        ALSMatrixFactorization(
            **parameter
        )


def test_invalid_ratings_dimension():
    model = ALSMatrixFactorization(
        n_components=2,
    )

    with pytest.raises(ValueError):
        model.fit(
            np.array([1, 2, 3])
        )


def test_negative_ratings_rejected():
    model = ALSMatrixFactorization(
        n_components=2,
    )

    ratings = create_rating_matrix()

    ratings[0, 0] = -1

    with pytest.raises(ValueError):
        model.fit(ratings)


def test_empty_ratings_rejected():
    model = ALSMatrixFactorization(
        n_components=2,
    )

    ratings = np.zeros(
        (4, 4)
    )

    with pytest.raises(ValueError):
        model.fit(ratings)

import numpy as np
import pytest

from applications.recommendation_system import (
    MatrixFactorizationRecommender,
    create_rating_matrix,
    train_test_split_ratings,
    fill_missing_with_item_mean,
    rmse_on_observed_ratings,
    get_item_names,
)


def test_rating_matrix_shape():
    ratings = create_rating_matrix()

    assert ratings.shape == (8, 8)
    assert np.all(ratings >= 0)


def test_item_names_match_matrix():
    ratings = create_rating_matrix()
    item_names = get_item_names()

    assert len(item_names) == ratings.shape[1]


def test_train_test_split():
    ratings = create_rating_matrix()

    train, test = train_test_split_ratings(
        ratings,
        test_ratio=0.2,
        random_state=42,
    )

    assert train.shape == ratings.shape
    assert test.shape == ratings.shape

    assert np.all(
        (train == 0) | (test == 0)
    )

    assert np.sum(test > 0) > 0


def test_split_is_reproducible():
    ratings = create_rating_matrix()

    train1, test1 = train_test_split_ratings(
        ratings,
        test_ratio=0.2,
        random_state=42,
    )

    train2, test2 = train_test_split_ratings(
        ratings,
        test_ratio=0.2,
        random_state=42,
    )

    assert np.array_equal(
        train1,
        train2,
    )

    assert np.array_equal(
        test1,
        test2,
    )


def test_missing_value_imputation():
    ratings = create_rating_matrix()

    filled = fill_missing_with_item_mean(
        ratings
    )

    assert filled.shape == ratings.shape
    assert np.all(filled > 0)

    observed_mask = ratings > 0

    assert np.allclose(
        filled[observed_mask],
        ratings[observed_mask],
    )


@pytest.mark.parametrize(
    "method",
    ["svd", "nmf"],
)
def test_recommender_fit(method):
    ratings = create_rating_matrix()

    recommender = (
        MatrixFactorizationRecommender(
            method=method,
            n_components=2,
            random_state=42,
        )
    )

    recommender.fit(ratings)

    assert (
        recommender.predicted_ratings.shape
        == ratings.shape
    )

    assert np.all(
        recommender.predicted_ratings >= 1
    )

    assert np.all(
        recommender.predicted_ratings <= 5
    )


def test_recommendations_exclude_rated_items():
    ratings = create_rating_matrix()
    item_names = get_item_names()

    recommender = (
        MatrixFactorizationRecommender(
            method="svd",
            n_components=2,
            random_state=42,
        )
    )

    recommender.fit(ratings)

    recommendations = recommender.recommend(
        user_index=0,
        item_names=item_names,
        ratings=ratings,
        n=3,
    )

    recommended_items = [
        item
        for item, _ in recommendations
    ]

    rated_items = [
        item_names[index]
        for index in np.where(
            ratings[0] > 0
        )[0]
    ]

    for item in recommended_items:
        assert item not in rated_items


def test_rmse():
    predictions = np.array(
        [
            [4.0, 3.0],
            [5.0, 2.0],
        ]
    )

    test = np.array(
        [
            [4.0, 0.0],
            [0.0, 1.0],
        ]
    )

    score = rmse_on_observed_ratings(
        predictions,
        test,
    )

    expected = np.sqrt(
        ((4.0 - 4.0) ** 2
         + (2.0 - 1.0) ** 2) / 2
    )

    assert np.isclose(
        score,
        expected,
    )

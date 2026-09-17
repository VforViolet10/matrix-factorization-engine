import numpy as np
import pandas as pd

from applications.recommendation_system.movielens_dataset import (
    create_user_item_matrix,
    dataset_summary,
    load_movies,
    load_ratings,
)


def test_load_ratings():
    ratings = load_ratings()

    assert isinstance(ratings, pd.DataFrame)
    assert list(ratings.columns) == [
        "user_id",
        "item_id",
        "rating",
        "timestamp",
    ]
    assert len(ratings) == 100000
    assert ratings["user_id"].nunique() == 943
    assert ratings["item_id"].nunique() == 1682


def test_rating_range():
    ratings = load_ratings()

    assert ratings["rating"].min() >= 1
    assert ratings["rating"].max() <= 5


def test_load_movies():
    movies = load_movies()

    assert isinstance(movies, pd.DataFrame)
    assert len(movies) == 1682
    assert "item_id" in movies.columns
    assert "title" in movies.columns


def test_create_user_item_matrix():
    ratings = load_ratings()

    matrix = create_user_item_matrix(ratings)

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (943, 1682)
    assert matrix.dtype == float


def test_matrix_contains_valid_ratings():
    ratings = load_ratings()

    matrix = create_user_item_matrix(ratings)

    observed = matrix[matrix > 0]

    assert observed.min() >= 1
    assert observed.max() <= 5


def test_dataset_summary():
    ratings = load_ratings()

    summary = dataset_summary(ratings)

    assert summary["num_ratings"] == 100000
    assert summary["num_users"] == 943
    assert summary["num_items"] == 1682
    assert summary["rating_min"] == 1

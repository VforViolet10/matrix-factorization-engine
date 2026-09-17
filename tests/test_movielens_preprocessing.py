import numpy as np

from applications.recommendation_system.movielens_dataset import (
    load_ratings,
)

from applications.recommendation_system.movielens_preprocessing import (
    train_test_split_movielens,
    create_train_test_matrices,
)


def test_train_test_split_preserves_all_ratings():
    ratings = load_ratings()

    train, test = train_test_split_movielens(
        ratings,
        test_ratio=0.2,
        random_state=42,
    )

    assert len(train) + len(test) == len(ratings)


def test_every_user_has_train_and_test_ratings():
    ratings = load_ratings()

    train, test = train_test_split_movielens(
        ratings,
        test_ratio=0.2,
        random_state=42,
    )

    train_users = set(train["user_id"])
    test_users = set(test["user_id"])

    assert train_users == test_users


def test_split_is_reproducible():
    ratings = load_ratings()

    train_1, test_1 = train_test_split_movielens(
        ratings,
        test_ratio=0.2,
        random_state=42,
    )

    train_2, test_2 = train_test_split_movielens(
        ratings,
        test_ratio=0.2,
        random_state=42,
    )

    assert train_1.equals(train_2)
    assert test_1.equals(test_2)


def test_train_test_matrices():
    ratings = load_ratings()

    train, test = train_test_split_movielens(
        ratings,
        test_ratio=0.2,
        random_state=42,
    )

    train_matrix, test_matrix, user_ids, item_ids = (
        create_train_test_matrices(train, test)
    )

    assert isinstance(train_matrix, np.ndarray)
    assert isinstance(test_matrix, np.ndarray)

    assert train_matrix.shape == test_matrix.shape
    assert train_matrix.shape[0] == 943

    assert len(user_ids) == 943
    assert train_matrix.shape[1] == len(item_ids)


def test_train_and_test_do_not_overlap():
    ratings = load_ratings()

    train, test = train_test_split_movielens(
        ratings,
        test_ratio=0.2,
        random_state=42,
    )

    train_pairs = set(
        zip(train["user_id"], train["item_id"])
    )

    test_pairs = set(
        zip(test["user_id"], test["item_id"])
    )

    assert train_pairs.isdisjoint(test_pairs)

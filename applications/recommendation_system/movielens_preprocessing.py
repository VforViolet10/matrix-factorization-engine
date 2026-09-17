import numpy as np
import pandas as pd


def train_test_split_movielens(
    ratings,
    test_ratio=0.2,
    random_state=42,
):
    """
    Split MovieLens ratings into train and test sets.

    Ratings are split independently for each user so that
    every user has ratings in both sets.

    Parameters
    ----------
    ratings : pandas.DataFrame
        MovieLens ratings with columns:
        user_id, item_id, rating, timestamp.

    test_ratio : float
        Fraction of each user's ratings assigned to test data.

    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    train : pandas.DataFrame
    test : pandas.DataFrame
    """

    required_columns = {
        "user_id",
        "item_id",
        "rating",
        "timestamp",
    }

    missing_columns = required_columns - set(ratings.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if not 0 < test_ratio < 1:
        raise ValueError(
            "test_ratio must be between 0 and 1."
        )

    rng = np.random.default_rng(random_state)

    train_parts = []
    test_parts = []

    for _, user_ratings in ratings.groupby("user_id"):
        indices = np.arange(len(user_ratings))
        rng.shuffle(indices)

        test_size = max(
            1,
            int(round(len(indices) * test_ratio)),
        )

        # Ensure every user keeps at least one
        # rating in the training set.
        test_size = min(
            test_size,
            len(indices) - 1,
        )

        test_indices = indices[:test_size]
        train_indices = indices[test_size:]

        train_parts.append(
            user_ratings.iloc[train_indices]
        )

        test_parts.append(
            user_ratings.iloc[test_indices]
        )

    train = pd.concat(
        train_parts,
        ignore_index=True,
    )

    test = pd.concat(
        test_parts,
        ignore_index=True,
    )

    return train, test


def create_train_test_matrices(
    train,
    test,
):
    """
    Convert train and test DataFrames into aligned
    user-item matrices.

    Only items appearing in the training set are included.
    This prevents cold-start items from entering the
    factorization benchmark.

    Zero represents an unobserved rating.

    Returns
    -------
    train_matrix : numpy.ndarray
    test_matrix : numpy.ndarray
    user_ids : numpy.ndarray
    item_ids : numpy.ndarray
    """

    user_ids = np.sort(
        train["user_id"].unique()
    )

    item_ids = np.sort(
        train["item_id"].unique()
    )

    train_matrix_df = train.pivot_table(
        index="user_id",
        columns="item_id",
        values="rating",
        fill_value=0,
    )

    test_matrix_df = test.pivot_table(
        index="user_id",
        columns="item_id",
        values="rating",
        fill_value=0,
    )

    train_matrix_df = train_matrix_df.reindex(
        index=user_ids,
        columns=item_ids,
        fill_value=0,
    )

    test_matrix_df = test_matrix_df.reindex(
        index=user_ids,
        columns=item_ids,
        fill_value=0,
    )

    return (
        train_matrix_df.to_numpy(dtype=float),
        test_matrix_df.to_numpy(dtype=float),
        user_ids,
        item_ids,
    )

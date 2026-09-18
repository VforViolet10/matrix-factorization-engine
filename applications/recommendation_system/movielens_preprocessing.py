```python
import numpy as np
import pandas as pd


def train_test_split_movielens(
    ratings,
    test_ratio=0.2,
    random_state=42,
):
    if not 0 < test_ratio < 1:
        raise ValueError(
            "test_ratio must be between 0 and 1."
        )

    required_columns = {
        "user_id",
        "item_id",
        "rating",
    }

    missing_columns = (
        required_columns
        - set(ratings.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    rng = np.random.default_rng(
        random_state
    )

    train_parts = []
    test_parts = []

    for user_id, user_ratings in ratings.groupby(
        "user_id"
    ):
        user_ratings = user_ratings.sample(
            frac=1,
            random_state=int(
                rng.integers(0, 1_000_000)
            ),
        )

        n_ratings = len(user_ratings)

        if n_ratings < 2:
            train_parts.append(
                user_ratings
            )
            continue

        n_test = max(
            1,
            int(
                round(
                    n_ratings
                    * test_ratio
                )
            ),
        )

        n_test = min(
            n_test,
            n_ratings - 1,
        )

        test_parts.append(
            user_ratings.iloc[:n_test]
        )

        train_parts.append(
            user_ratings.iloc[n_test:]
        )

    train = pd.concat(
        train_parts,
        ignore_index=True,
    )

    if test_parts:
        test = pd.concat(
            test_parts,
            ignore_index=True,
        )
    else:
        test = pd.DataFrame(
            columns=ratings.columns
        )

    return train, test


def create_train_test_matrices(
    train,
    test,
):
    required_columns = {
        "user_id",
        "item_id",
        "rating",
    }

    for name, dataframe in [
        ("train", train),
        ("test", test),
    ]:
        missing_columns = (
            required_columns
            - set(dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{name} is missing required "
                f"columns: {sorted(missing_columns)}"
            )

    user_ids = np.sort(
        train["user_id"].unique()
    )

    item_ids = np.sort(
        train["item_id"].unique()
    )

    train_matrix = (
        train.pivot_table(
            index="user_id",
            columns="item_id",
            values="rating",
            fill_value=0,
        )
        .reindex(
            index=user_ids,
            columns=item_ids,
            fill_value=0,
        )
        .to_numpy(dtype=float)
    )

    test_matrix = (
        test.pivot_table(
            index="user_id",
            columns="item_id",
            values="rating",
            fill_value=0,
        )
        .reindex(
            index=user_ids,
            columns=item_ids,
            fill_value=0,
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

    return (
        train_matrix,
        test_matrix,
        user_ids,
        item_ids,
    )
```

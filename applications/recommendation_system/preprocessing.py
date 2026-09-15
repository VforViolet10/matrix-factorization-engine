import numpy as np


def train_test_split_ratings(
    ratings,
    test_ratio=0.2,
    random_state=42,
):
    """
    Split observed ratings into training and test matrices.

    Zero values represent missing ratings and are never
    selected as test observations.

    Parameters
    ----------
    ratings : numpy.ndarray
        User-item rating matrix.

    test_ratio : float
        Fraction of observed ratings assigned to the test set.

    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    train : numpy.ndarray
        Training rating matrix.

    test : numpy.ndarray
        Test rating matrix containing only held-out ratings.
    """

    ratings = np.asarray(
        ratings,
        dtype=float,
    )

    if ratings.ndim != 2:
        raise ValueError(
            "ratings must be a 2D matrix."
        )

    if not 0 < test_ratio < 1:
        raise ValueError(
            "test_ratio must be between 0 and 1."
        )

    rng = np.random.default_rng(
        random_state
    )

    observed = np.argwhere(
        ratings > 0
    )

    n_test = max(
        1,
        int(len(observed) * test_ratio)
    )

    selected_indices = rng.choice(
        len(observed),
        size=n_test,
        replace=False,
    )

    test_positions = observed[
        selected_indices
    ]

    train = ratings.copy()
    test = np.zeros_like(ratings)

    for row, column in test_positions:
        test[row, column] = ratings[
            row,
            column,
        ]
        train[row, column] = 0

    return train, test


def fill_missing_with_item_mean(ratings):
    """
    Replace missing ratings with item means.

    Missing ratings are represented by zero.

    If an item has no observed ratings, the global
    mean rating is used.
    """

    ratings = np.asarray(
        ratings,
        dtype=float,
    )

    if ratings.ndim != 2:
        raise ValueError(
            "ratings must be a 2D matrix."
        )

    result = ratings.copy()

    observed_values = ratings[
        ratings > 0
    ]

    if len(observed_values) == 0:
        raise ValueError(
            "ratings must contain at least one observed rating."
        )

    global_mean = np.mean(
        observed_values
    )

    for column in range(
        ratings.shape[1]
    ):
        observed = ratings[
            :, column
        ]

        observed = observed[
            observed > 0
        ]

        if len(observed) > 0:
            mean_rating = np.mean(
                observed
            )
        else:
            mean_rating = global_mean

        missing = ratings[
            :, column
        ] == 0

        result[
            missing,
            column
        ] = mean_rating

    return result

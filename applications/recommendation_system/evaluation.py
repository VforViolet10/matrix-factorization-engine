import numpy as np


def rmse_on_observed_ratings(
    predictions,
    test_ratings,
):
    predictions = np.asarray(predictions, dtype=float)
    test_ratings = np.asarray(test_ratings, dtype=float)

    if predictions.shape != test_ratings.shape:
        raise ValueError(
            "predictions and test_ratings must have the same shape."
        )

    observed = test_ratings > 0

    if not np.any(observed):
        raise ValueError(
            "test_ratings must contain at least one observed rating."
        )

    errors = predictions[observed] - test_ratings[observed]

    return float(
        np.sqrt(np.mean(errors ** 2))
    )


def precision_at_k(
    predictions,
    test_ratings,
    k=3,
    relevance_threshold=4.0,
):
    """
    Calculate Precision@K for a user-item recommendation matrix.

    A test item is considered relevant when its actual rating is
    greater than or equal to relevance_threshold.

    Only items present in the test set are used as ground truth.
    """

    predictions = np.asarray(predictions, dtype=float)
    test_ratings = np.asarray(test_ratings, dtype=float)

    if predictions.shape != test_ratings.shape:
        raise ValueError(
            "predictions and test_ratings must have the same shape."
        )

    if k <= 0:
        raise ValueError(
            "k must be greater than zero."
        )

    if relevance_threshold < 0:
        raise ValueError(
            "relevance_threshold must be non-negative."
        )

    precisions = []

    for user in range(test_ratings.shape[0]):

        test_items = np.where(
            test_ratings[user] > 0
        )[0]

        if len(test_items) == 0:
            continue

        ranked_items = test_items[
            np.argsort(
                predictions[user, test_items]
            )[::-1]
        ]

        top_k = ranked_items[:k]

        relevant = (
            test_ratings[user, top_k]
            >= relevance_threshold
        )

        precisions.append(
            np.mean(relevant)
        )

    if not precisions:
        raise ValueError(
            "No users contain test ratings."
        )

    return float(
        np.mean(precisions)
    )


def recall_at_k(
    predictions,
    test_ratings,
    k=3,
    relevance_threshold=4.0,
):
    """
    Calculate Recall@K for a user-item recommendation matrix.

    A test item is considered relevant when its actual rating is
    greater than or equal to relevance_threshold.
    """

    predictions = np.asarray(predictions, dtype=float)
    test_ratings = np.asarray(test_ratings, dtype=float)

    if predictions.shape != test_ratings.shape:
        raise ValueError(
            "predictions and test_ratings must have the same shape."
        )

    if k <= 0:
        raise ValueError(
            "k must be greater than zero."
        )

    if relevance_threshold < 0:
        raise ValueError(
            "relevance_threshold must be non-negative."
        )

    recalls = []

    for user in range(test_ratings.shape[0]):

        test_items = np.where(
            test_ratings[user] > 0
        )[0]

        if len(test_items) == 0:
            continue

        relevant_items = test_items[
            test_ratings[user, test_items]
            >= relevance_threshold
        ]

        if len(relevant_items) == 0:
            continue

        ranked_items = test_items[
            np.argsort(
                predictions[user, test_items]
            )[::-1]
        ]

        top_k = ranked_items[:k]

        hits = np.sum(
            np.isin(
                top_k,
                relevant_items,
            )
        )

        recalls.append(
            hits / len(relevant_items)
        )

    if not recalls:
        raise ValueError(
            "No relevant test items found."
        )

    return float(
        np.mean(recalls)
    )

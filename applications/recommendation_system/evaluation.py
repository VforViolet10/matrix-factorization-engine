```python
import numpy as np


def _validate_inputs(predictions, train_ratings, test_ratings):
    predictions = np.asarray(predictions, dtype=float)
    train_ratings = np.asarray(train_ratings, dtype=float)
    test_ratings = np.asarray(test_ratings, dtype=float)

    if predictions.ndim != 2:
        raise ValueError("predictions must be a 2D matrix.")

    if train_ratings.ndim != 2:
        raise ValueError("train_ratings must be a 2D matrix.")

    if test_ratings.ndim != 2:
        raise ValueError("test_ratings must be a 2D matrix.")

    if (
        predictions.shape
        != train_ratings.shape
        or predictions.shape
        != test_ratings.shape
    ):
        raise ValueError(
            "predictions, train_ratings, and test_ratings "
            "must have the same shape."
        )

    return predictions, train_ratings, test_ratings


def rmse_on_observed_ratings(predictions, test_ratings):
    predictions = np.asarray(predictions, dtype=float)
    test_ratings = np.asarray(test_ratings, dtype=float)

    if predictions.shape != test_ratings.shape:
        raise ValueError(
            "predictions and test_ratings must have the same shape."
        )

    mask = test_ratings > 0

    if not np.any(mask):
        raise ValueError(
            "test_ratings must contain at least one observed rating."
        )

    errors = predictions[mask] - test_ratings[mask]

    return float(np.sqrt(np.mean(errors ** 2)))


def _get_top_k_items(
    predictions,
    train_ratings,
    user_index,
    k,
):
    unseen_items = np.where(
        train_ratings[user_index] == 0
    )[0]

    if len(unseen_items) == 0:
        return np.array([], dtype=int)

    scores = predictions[
        user_index,
        unseen_items,
    ]

    ranked_indices = np.argsort(
        scores
    )[::-1]

    top_k = ranked_indices[:k]

    return unseen_items[top_k]


def precision_at_k(
    predictions,
    train_ratings,
    test_ratings,
    k=5,
    relevance_threshold=4.0,
):
    predictions, train_ratings, test_ratings = _validate_inputs(
        predictions,
        train_ratings,
        test_ratings,
    )

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    precisions = []

    for user_index in range(predictions.shape[0]):
        top_k_items = _get_top_k_items(
            predictions,
            train_ratings,
            user_index,
            k,
        )

        if len(top_k_items) == 0:
            continue

        relevant = test_ratings[
            user_index,
            top_k_items,
        ] >= relevance_threshold

        precisions.append(
            np.mean(relevant)
        )

    if not precisions:
        return 0.0

    return float(np.mean(precisions))


def recall_at_k(
    predictions,
    train_ratings,
    test_ratings,
    k=5,
    relevance_threshold=4.0,
):
    predictions, train_ratings, test_ratings = _validate_inputs(
        predictions,
        train_ratings,
        test_ratings,
    )

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    recalls = []

    for user_index in range(predictions.shape[0]):
        relevant_items = np.where(
            (
                test_ratings[user_index]
                >= relevance_threshold
            )
            & (
                train_ratings[user_index]
                == 0
            )
        )[0]

        if len(relevant_items) == 0:
            continue

        top_k_items = _get_top_k_items(
            predictions,
            train_ratings,
            user_index,
            k,
        )

        hits = np.intersect1d(
            top_k_items,
            relevant_items,
        )

        recalls.append(
            len(hits) / len(relevant_items)
        )

    if not recalls:
        return 0.0

    return float(np.mean(recalls))


def ndcg_at_k(
    predictions,
    train_ratings,
    test_ratings,
    k=5,
    relevance_threshold=4.0,
):
    predictions, train_ratings, test_ratings = _validate_inputs(
        predictions,
        train_ratings,
        test_ratings,
    )

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    ndcg_scores = []

    for user_index in range(predictions.shape[0]):
        top_k_items = _get_top_k_items(
            predictions,
            train_ratings,
            user_index,
            k,
        )

        if len(top_k_items) == 0:
            continue

        relevance = (
            test_ratings[
                user_index,
                top_k_items,
            ]
            >= relevance_threshold
        ).astype(float)

        discounts = np.log2(
            np.arange(2, len(relevance) + 2)
        )

        dcg = np.sum(
            relevance / discounts
        )

        all_relevant = np.sum(
            (
                test_ratings[user_index]
                >= relevance_threshold
            )
            & (
                train_ratings[user_index]
                == 0
            )
        )

        if all_relevant == 0:
            continue

        ideal_length = min(
            k,
            all_relevant,
        )

        ideal_relevance = np.ones(
            ideal_length,
            dtype=float,
        )

        ideal_discounts = np.log2(
            np.arange(2, ideal_length + 2)
        )

        idcg = np.sum(
            ideal_relevance / ideal_discounts
        )

        if idcg > 0:
            ndcg_scores.append(
                dcg / idcg
            )

    if not ndcg_scores:
        return 0.0

    return float(np.mean(ndcg_scores))
```

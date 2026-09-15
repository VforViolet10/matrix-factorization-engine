import numpy as np
import pytest

from applications.recommendation_system import (
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k():
    predictions = np.array([
        [5, 4, 3, 2],
    ])

    test_ratings = np.array([
        [5, 4, 0, 0],
    ])

    result = precision_at_k(
        predictions,
        test_ratings,
        k=2,
        relevance_threshold=4,
    )

    assert result == pytest.approx(1.0)


def test_recall_at_k():
    predictions = np.array([
        [5, 4, 3, 2],
    ])

    test_ratings = np.array([
        [5, 4, 0, 0],
    ])

    result = recall_at_k(
        predictions,
        test_ratings,
        k=1,
        relevance_threshold=4,
    )

    assert result == pytest.approx(0.5)


def test_precision_invalid_k():
    predictions = np.ones((2, 4))
    test_ratings = np.ones((2, 4))

    with pytest.raises(ValueError):
        precision_at_k(
            predictions,
            test_ratings,
            k=0,
        )


def test_recall_invalid_k():
    predictions = np.ones((2, 4))
    test_ratings = np.ones((2, 4))

    with pytest.raises(ValueError):
        recall_at_k(
            predictions,
            test_ratings,
            k=0,
        )


def test_metrics_require_matching_shapes():
    predictions = np.ones((2, 4))
    test_ratings = np.ones((2, 3))

    with pytest.raises(ValueError):
        precision_at_k(
            predictions,
            test_ratings,
        )

    with pytest.raises(ValueError):
        recall_at_k(
            predictions,
            test_ratings,
        )

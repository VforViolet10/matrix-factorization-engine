import csv
import os

import numpy as np

from applications.recommendation_system import (
    ALSMatrixFactorization,
    MaskedMatrixFactorization,
    MatrixFactorizationRecommender,
    create_rating_matrix,
    precision_at_k,
    recall_at_k,
    rmse_on_observed_ratings,
    train_test_split_ratings,
)


METHODS = [
    "svd",
    "nmf",
    "masked_sgd",
    "als",
]

COMPONENTS = [
    1,
    2,
    3,
    4,
]

RANDOM_SEEDS = [
    42,
    7,
    21,
    100,
    123,
]

K = 3
RELEVANCE_THRESHOLD = 4.0


def train_model(
    method,
    train_ratings,
    components,
    seed,
):
    if method == "svd":
        model = MatrixFactorizationRecommender(
            method="svd",
            n_components=components,
            random_state=seed,
        )

        model.fit(train_ratings)

        return model.predicted_ratings

    if method == "nmf":
        model = MatrixFactorizationRecommender(
            method="nmf",
            n_components=components,
            random_state=seed,
        )

        model.fit(train_ratings)

        return model.predicted_ratings

    if method == "masked_sgd":
        model = MaskedMatrixFactorization(
            n_components=components,
            learning_rate=0.001,
            regularization=0.01,
            max_iter=2000,
            tolerance=1e-5,
            random_state=seed,
        )

        model.fit(train_ratings)

        return model.reconstruct()

    if method == "als":
        model = ALSMatrixFactorization(
            n_components=components,
            regularization=0.1,
            max_iter=50,
            tolerance=1e-5,
            random_state=seed,
        )

        model.fit(train_ratings)

        return model.reconstruct()

    raise ValueError(
        f"Unknown method: {method}"
    )


def run_experiment():
    ratings = create_rating_matrix()

    results = []

    for method in METHODS:

        for components in COMPONENTS:

            rmse_scores = []
            precision_scores = []
            recall_scores = []

            for seed in RANDOM_SEEDS:

                train_ratings, test_ratings = (
                    train_test_split_ratings(
                        ratings,
                        test_ratio=0.2,
                        random_state=seed,
                    )
                )

                predictions = train_model(
                    method,
                    train_ratings,
                    components,
                    seed,
                )

                rmse = rmse_on_observed_ratings(
                    predictions,
                    test_ratings,
                )

                precision = precision_at_k(
                    predictions,
                    test_ratings,
                    k=K,
                    relevance_threshold=RELEVANCE_THRESHOLD,
                )

                recall = recall_at_k(
                    predictions,
                    test_ratings,
                    k=K,
                    relevance_threshold=RELEVANCE_THRESHOLD,
                )

                rmse_scores.append(rmse)
                precision_scores.append(precision)
                recall_scores.append(recall)

            result = {
                "method": method,
                "components": components,
                "mean_rmse": np.mean(rmse_scores),
                "std_rmse": np.std(rmse_scores),
                "mean_precision_at_k": np.mean(
                    precision_scores
                ),
                "std_precision_at_k": np.std(
                    precision_scores
                ),
                "mean_recall_at_k": np.mean(
                    recall_scores
                ),
                "std_recall_at_k": np.std(
                    recall_scores
                ),
            }

            results.append(result)

            print(
                f"{method.upper():<10} | "
                f"k={components} | "
                f"RMSE={result['mean_rmse']:.4f} | "
                f"P@{K}={result['mean_precision_at_k']:.4f} | "
                f"R@{K}={result['mean_recall_at_k']:.4f}"
            )

    return results


def save_results(results):
    output_dir = (
        "applications/"
        "recommendation_system/"
        "results"
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    output_path = os.path.join(
        output_dir,
        "recommendation_ranking_comparison.csv",
    )

    with open(
        output_path,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "method",
                "components",
                "mean_rmse",
                "std_rmse",
                "mean_precision_at_k",
                "std_precision_at_k",
                "mean_recall_at_k",
                "std_recall_at_k",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    return output_path


def main():
    print("=" * 85)
    print("RECOMMENDATION RANKING EVALUATION")
    print("=" * 85)

    print(
        f"Evaluation metrics: RMSE, Precision@{K}, Recall@{K}"
    )
    print(
        f"Relevance threshold: rating >= {RELEVANCE_THRESHOLD}"
    )
    print()

    results = run_experiment()

    output_path = save_results(results)

    print("\n" + "=" * 85)
    print(
        f"Results saved to: {output_path}"
    )
    print("=" * 85)


if __name__ == "__main__":
    main()

import csv
import os
import time

import numpy as np

from applications.recommendation_system import (
    ALSMatrixFactorization,
    MaskedMatrixFactorization,
    MatrixFactorizationRecommender,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse_on_observed_ratings,
)

from applications.recommendation_system.movielens_dataset import (
    load_ratings,
)

from applications.recommendation_system.movielens_preprocessing import (
    train_test_split_movielens,
    create_train_test_matrices,
)


METHODS = [
    "svd",
    "nmf",
    "masked_sgd",
    "als",
]

COMPONENTS = [
    5,
    10,
    20,
]

RANDOM_SEEDS = [
    42,
]

K = 5
RELEVANCE_THRESHOLD = 4.0

RESULTS_DIR = os.path.join(
    "applications",
    "recommendation_system",
    "results",
)

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "movielens_model_comparison.csv",
)


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
            max_iter=300,
            tolerance=1e-5,
            random_state=seed,
        )

        model.fit(train_ratings)

        return model.reconstruct()

    if method == "als":
        model = ALSMatrixFactorization(
            n_components=components,
            regularization=0.1,
            max_iter=30,
            tolerance=1e-5,
            random_state=seed,
        )

        model.fit(train_ratings)

        return model.reconstruct()

    raise ValueError(
        f"Unknown method: {method}"
    )


def evaluate_model(
    predictions,
    train_matrix,
    test_matrix,
):
    rmse = rmse_on_observed_ratings(
        predictions,
        test_matrix,
    )

    precision = precision_at_k(
        predictions,
        train_matrix,
        test_matrix,
        k=K,
        relevance_threshold=RELEVANCE_THRESHOLD,
    )

    recall = recall_at_k(
        predictions,
        train_matrix,
        test_matrix,
        k=K,
        relevance_threshold=RELEVANCE_THRESHOLD,
    )

    ndcg = ndcg_at_k(
        predictions,
        train_matrix,
        test_matrix,
        k=K,
        relevance_threshold=RELEVANCE_THRESHOLD,
    )

    return (
        rmse,
        precision,
        recall,
        ndcg,
    )


def main():
    ratings = load_ratings()

    train, test = train_test_split_movielens(
        ratings,
        test_ratio=0.2,
        random_state=42,
    )

    (
        train_matrix,
        test_matrix,
        user_ids,
        item_ids,
    ) = create_train_test_matrices(
        train,
        test,
    )

    print("=" * 75)
    print(
        "MovieLens 100K - Matrix Factorization Model Comparison"
    )
    print("=" * 75)

    print(
        f"Users: {len(user_ids)}"
    )

    print(
        f"Items: {len(item_ids)}"
    )

    print(
        f"Train matrix: {train_matrix.shape}"
    )

    print(
        f"Test matrix:  {test_matrix.shape}"
    )

    print(
        f"Models: {', '.join(METHODS)}"
    )

    print(
        f"Components: {COMPONENTS}"
    )

    print(
        f"Ranking metrics: Precision@{K}, "
        f"Recall@{K}, NDCG@{K}"
    )

    print(
        "Seeds: 1"
    )

    print("=" * 75)

    results = []

    total_runs = (
        len(METHODS)
        * len(COMPONENTS)
        * len(RANDOM_SEEDS)
    )

    current_run = 0

    for method in METHODS:
        for components in COMPONENTS:

            rmse_scores = []
            precision_scores = []
            recall_scores = []
            ndcg_scores = []
            runtime_scores = []

            for seed in RANDOM_SEEDS:

                current_run += 1

                print()
                print(
                    f"[{current_run}/{total_runs}] "
                    f"{method.upper()} "
                    f"k={components}"
                )

                start_time = time.perf_counter()

                predictions = train_model(
                    method,
                    train_matrix,
                    components,
                    seed,
                )

                runtime = (
                    time.perf_counter()
                    - start_time
                )

                (
                    rmse,
                    precision,
                    recall,
                    ndcg,
                ) = evaluate_model(
                    predictions,
                    train_matrix,
                    test_matrix,
                )

                rmse_scores.append(rmse)
                precision_scores.append(precision)
                recall_scores.append(recall)
                ndcg_scores.append(ndcg)
                runtime_scores.append(runtime)

                print(
                    f"RMSE={rmse:.4f} | "
                    f"P@{K}={precision:.4f} | "
                    f"R@{K}={recall:.4f} | "
                    f"NDCG@{K}={ndcg:.4f} | "
                    f"Time={runtime:.3f}s"
                )

            result = {
                "method": method,
                "components": components,
                "mean_rmse": float(
                    np.mean(rmse_scores)
                ),
                "std_rmse": float(
                    np.std(rmse_scores)
                ),
                "mean_precision_at_k": float(
                    np.mean(precision_scores)
                ),
                "std_precision_at_k": float(
                    np.std(precision_scores)
                ),
                "mean_recall_at_k": float(
                    np.mean(recall_scores)
                ),
                "std_recall_at_k": float(
                    np.std(recall_scores)
                ),
                "mean_ndcg_at_k": float(
                    np.mean(ndcg_scores)
                ),
                "std_ndcg_at_k": float(
                    np.std(ndcg_scores)
                ),
                "mean_runtime_seconds": float(
                    np.mean(runtime_scores)
                ),
                "std_runtime_seconds": float(
                    np.std(runtime_scores)
                ),
            }

            results.append(result)

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    fieldnames = [
        "method",
        "components",
        "mean_rmse",
        "std_rmse",
        "mean_precision_at_k",
        "std_precision_at_k",
        "mean_recall_at_k",
        "std_recall_at_k",
        "mean_ndcg_at_k",
        "std_ndcg_at_k",
        "mean_runtime_seconds",
        "std_runtime_seconds",
    ]

    with open(
        RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("=" * 75)
    print("Benchmark complete.")
    print()
    print("Results saved to:")
    print(RESULTS_FILE)
    print("=" * 75)


if __name__ == "__main__":
    main()

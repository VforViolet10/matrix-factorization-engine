import csv
import os

import numpy as np

from applications.recommendation_system import (
    ALSMatrixFactorization,
    MaskedMatrixFactorization,
    MatrixFactorizationRecommender,
    create_rating_matrix,
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

            scores = []

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

                score = (
                    rmse_on_observed_ratings(
                        predictions,
                        test_ratings,
                    )
                )

                scores.append(score)

            mean_rmse = np.mean(scores)
            std_rmse = np.std(scores)

            results.append(
                {
                    "method": method,
                    "components": components,
                    "mean_rmse": mean_rmse,
                    "std_rmse": std_rmse,
                }
            )

            print(
                f"{method.upper():<10} | "
                f"k={components} | "
                f"Mean RMSE={mean_rmse:.4f} | "
                f"Std={std_rmse:.4f}"
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
        "recommendation_model_comparison.csv",
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
            ],
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    return output_path


def main():
    print("=" * 75)
    print(
        "RECOMMENDATION MODEL COMPARISON"
    )
    print("=" * 75)

    results = run_experiment()

    output_path = save_results(
        results
    )

    print("\n" + "=" * 75)
    print(
        f"Results saved to: {output_path}"
    )
    print("=" * 75)


if __name__ == "__main__":
    main()

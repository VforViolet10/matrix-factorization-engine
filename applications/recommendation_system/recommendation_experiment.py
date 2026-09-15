import csv
import os

import numpy as np

from applications.recommendation_system import (
    MatrixFactorizationRecommender,
    create_rating_matrix,
    train_test_split_ratings,
    rmse_on_observed_ratings,
)


METHODS = ["svd", "nmf"]
COMPONENTS = [1, 2, 3, 4]
RANDOM_SEEDS = [42, 7, 21, 100, 123]


def run_experiment():
    """
    Compare SVD and NMF across multiple
    latent dimensions and train/test splits.
    """

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

                recommender = (
                    MatrixFactorizationRecommender(
                        method=method,
                        n_components=components,
                        random_state=seed,
                    )
                )

                recommender.fit(
                    train_ratings
                )

                score = (
                    rmse_on_observed_ratings(
                        recommender.predicted_ratings,
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
                f"{method.upper()} | "
                f"k={components} | "
                f"Mean RMSE={mean_rmse:.4f} | "
                f"Std={std_rmse:.4f}"
            )

    return results


def save_results(results):
    """
    Save experiment results to CSV.
    """

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
        "recommendation_results.csv",
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
        writer.writerows(results)

    return output_path


def main():

    print("=" * 65)
    print("RECOMMENDATION SYSTEM EXPERIMENT")
    print("=" * 65)

    results = run_experiment()

    output_path = save_results(
        results
    )

    print("\n" + "=" * 65)
    print(
        f"Results saved to: {output_path}"
    )
    print("=" * 65)


if __name__ == "__main__":
    main()

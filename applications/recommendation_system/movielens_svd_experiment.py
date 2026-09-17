import numpy as np

from applications.recommendation_system.movielens_dataset import (
    load_ratings,
)
from applications.recommendation_system.movielens_preprocessing import (
    train_test_split_movielens,
    create_train_test_matrices,
)
from applications.recommendation_system.recommender import (
    MatrixFactorizationRecommender,
)
from applications.recommendation_system.evaluation import (
    rmse_on_observed_ratings,
)


def main():
    ratings = load_ratings()

    train, test = train_test_split_movielens(
        ratings,
        test_ratio=0.2,
        random_state=42,
    )

    train_matrix, test_matrix, user_ids, item_ids = (
        create_train_test_matrices(train, test)
    )

    print("=" * 60)
    print("MovieLens 100K - SVD Baseline")
    print("=" * 60)

    print(f"Train matrix: {train_matrix.shape}")
    print(f"Test matrix:  {test_matrix.shape}")

    # Use a moderate rank for the first real-data experiment.
    n_components = 20

    model = MatrixFactorizationRecommender(
        method="svd",
        n_components=n_components,
    )

    print()
    print(f"Training SVD with {n_components} components...")

    model.fit(train_matrix)

    predictions = model.model.reconstruct()

    rmse = rmse_on_observed_ratings(
        test_matrix,
        predictions,
    )

    print()
    print("Results")
    print("-" * 60)
    print(f"Components: {n_components}")
    print(f"Test RMSE:  {rmse:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

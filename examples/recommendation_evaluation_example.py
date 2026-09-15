from applications.recommendation_system import (
    MatrixFactorizationRecommender,
    create_rating_matrix,
    train_test_split_ratings,
    rmse_on_observed_ratings,
)


def main():

    ratings = create_rating_matrix()

    train_ratings, test_ratings = (
        train_test_split_ratings(
            ratings,
            test_ratio=0.2,
            random_state=42,
        )
    )

    print("=" * 60)
    print("RECOMMENDATION SYSTEM EVALUATION")
    print("=" * 60)

    print(
        f"\nOriginal ratings: "
        f"{int((ratings > 0).sum())}"
    )

    print(
        f"Training ratings: "
        f"{int((train_ratings > 0).sum())}"
    )

    print(
        f"Test ratings: "
        f"{int((test_ratings > 0).sum())}"
    )

    for method in ["svd", "nmf"]:

        recommender = (
            MatrixFactorizationRecommender(
                method=method,
                n_components=2,
                random_state=42,
            )
        )

        recommender.fit(
            train_ratings
        )

        score = rmse_on_observed_ratings(
            recommender.predicted_ratings,
            test_ratings,
        )

        print(
            f"\n{method.upper()} RMSE: "
            f"{score:.4f}"
        )


if __name__ == "__main__":
    main()

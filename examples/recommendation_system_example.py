from applications.recommendation_system import (
    MatrixFactorizationRecommender,
    create_rating_matrix,
    get_item_names,
)


def main():
    ratings = create_rating_matrix()
    item_names = get_item_names()

    print("=" * 60)
    print("MATRIX FACTORIZATION RECOMMENDATION SYSTEM")
    print("=" * 60)

    print("\nRating Matrix:")
    print(ratings)

    for method in ["svd", "nmf"]:

        print("\n" + "-" * 60)
        print(f"{method.upper()} RECOMMENDATIONS")
        print("-" * 60)

        recommender = MatrixFactorizationRecommender(
            method=method,
            n_components=2,
            random_state=42,
        )

        recommender.fit(ratings)

        for user_index in range(ratings.shape[0]):

            recommendations = recommender.recommend(
                user_index=user_index,
                item_names=item_names,
                ratings=ratings,
                n=2,
            )

            print(
                f"\nUser {user_index + 1}:"
            )

            for item, predicted_rating in recommendations:
                print(
                    f"  {item}: "
                    f"{predicted_rating:.2f}"
                )


if __name__ == "__main__":
    main()

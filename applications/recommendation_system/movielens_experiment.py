from applications.recommendation_system.movielens_dataset import (
    load_ratings,
    dataset_summary,
)
from applications.recommendation_system.movielens_preprocessing import (
    train_test_split_movielens,
    create_train_test_matrices,
)


def main():
    ratings = load_ratings()

    summary = dataset_summary(ratings)

    train, test = train_test_split_movielens(
        ratings,
        test_ratio=0.2,
        random_state=42,
    )

    train_matrix, test_matrix, user_ids, item_ids = (
        create_train_test_matrices(train, test)
    )

    print("=" * 60)
    print("MovieLens 100K Dataset")
    print("=" * 60)

    print(f"Users:          {summary['num_users']}")
    print(f"Movies:         {summary['num_items']}")
    print(f"Ratings:        {summary['num_ratings']}")
    print(f"Rating range:   {summary['rating_min']} - "
          f"{summary['rating_max']}")
    print(f"Average rating: {summary['rating_mean']:.3f}")

    print()
    print("Train/Test Split")
    print("-" * 60)

    print(f"Train ratings:  {len(train)}")
    print(f"Test ratings:   {len(test)}")
    print(f"Train matrix:   {train_matrix.shape}")
    print(f"Test matrix:    {test_matrix.shape}")

    train_density = (
        (train_matrix > 0).sum()
        / train_matrix.size
    )

    test_density = (
        (test_matrix > 0).sum()
        / test_matrix.size
    )

    print(f"Train density:  {train_density:.4f}")
    print(f"Test density:   {test_density:.4f}")

    print()
    print("Validation")
    print("-" * 60)

    print(f"Users in train: {len(user_ids)}")
    print(f"Items in train: {len(item_ids)}")

    overlap = (
        (train_matrix > 0)
        & (test_matrix > 0)
    ).sum()

    print(f"Train/test overlap: {overlap}")

    print("=" * 60)


if __name__ == "__main__":
    main()

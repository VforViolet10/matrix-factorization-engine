from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "movielens"
    / "ml-100k"
)


def create_rating_matrix():
    """
    Create a synthetic user-item rating matrix.
    """

    ratings = np.array(
        [
            [5, 4, 0, 0, 2, 1, 0, 0],
            [4, 5, 0, 0, 1, 2, 0, 0],
            [1, 2, 5, 4, 0, 0, 0, 1],
            [0, 1, 4, 5, 0, 0, 2, 1],
            [5, 4, 0, 0, 5, 4, 0, 0],
            [4, 5, 0, 0, 4, 5, 0, 0],
            [0, 0, 5, 4, 1, 0, 5, 4],
            [0, 0, 4, 5, 2, 1, 4, 5],
        ],
        dtype=float,
    )

    return ratings


def get_item_names():
    """
    Return names corresponding to the items
    in the synthetic rating matrix.
    """

    return [
        "Inception",
        "Interstellar",
        "The Dark Knight",
        "Dune",
        "Avengers",
        "Iron Man",
        "The Matrix",
        "Gladiator",
    ]


def get_user_names():
    """
    Return names corresponding to users
    in the synthetic rating matrix.
    """

    return [
        "User 1",
        "User 2",
        "User 3",
        "User 4",
        "User 5",
        "User 6",
        "User 7",
        "User 8",
    ]


def load_ratings(data_path=DEFAULT_DATA_PATH):
    """
    Load MovieLens 100K ratings from u.data.
    """

    data_path = Path(data_path)
    ratings_file = data_path / "u.data"

    if not ratings_file.exists():
        raise FileNotFoundError(
            f"MovieLens ratings file not found: "
            f"{ratings_file}"
        )

    ratings = pd.read_csv(
        ratings_file,
        sep="\t",
        names=[
            "user_id",
            "item_id",
            "rating",
            "timestamp",
        ],
        engine="python",
    )

    return ratings


def load_movies(data_path=DEFAULT_DATA_PATH):
    """
    Load MovieLens 100K movie metadata from u.item.
    """

    data_path = Path(data_path)
    movies_file = data_path / "u.item"

    if not movies_file.exists():
        raise FileNotFoundError(
            f"MovieLens movie file not found: "
            f"{movies_file}"
        )

    genre_count = 19

    columns = [
        "item_id",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url",
    ]

    columns.extend(
        [
            f"genre_{i}"
            for i in range(genre_count)
        ]
    )

    movies = pd.read_csv(
        movies_file,
        sep="|",
        names=columns,
        encoding="latin-1",
        engine="python",
    )

    return movies


def create_user_item_matrix(ratings):
    """
    Convert MovieLens ratings into a dense user-item matrix.
    Missing ratings are represented by zero.
    """

    required_columns = {
        "user_id",
        "item_id",
        "rating",
    }

    missing_columns = (
        required_columns
        - set(ratings.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    matrix = ratings.pivot_table(
        index="user_id",
        columns="item_id",
        values="rating",
        fill_value=0,
    )

    return matrix.to_numpy(dtype=float)


def dataset_summary(ratings):
    """
    Return basic statistics for a MovieLens ratings DataFrame.
    """

    return {
        "num_ratings": len(ratings),
        "num_users": ratings["user_id"].nunique(),
        "num_items": ratings["item_id"].nunique(),
        "rating_min": ratings["rating"].min(),
        "rating_max": ratings["rating"].max(),
        "rating_mean": ratings["rating"].mean(),
    }

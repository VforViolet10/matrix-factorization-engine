from pathlib import Path

import pandas as pd


DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "movielens"
    / "ml-100k"
)


def load_ratings(data_path=DEFAULT_DATA_PATH):
    """
    Load MovieLens 100K ratings.

    Returns
    -------
    pandas.DataFrame
        Columns:
        user_id, item_id, rating, timestamp
    """

    data_path = Path(data_path)
    ratings_file = data_path / "u.data"

    if not ratings_file.exists():
        raise FileNotFoundError(
            f"MovieLens ratings file not found: {ratings_file}"
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
    Load MovieLens movie metadata.

    Returns
    -------
    pandas.DataFrame
        Movie ID, title, and genre columns.
    """

    data_path = Path(data_path)
    movies_file = data_path / "u.item"

    if not movies_file.exists():
        raise FileNotFoundError(
            f"MovieLens movie file not found: {movies_file}"
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
        [f"genre_{i}" for i in range(genre_count)]
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
    Convert ratings into a user-item matrix.

    Rows represent users.
    Columns represent items.
    Zero represents an unrated item.
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
    Return basic MovieLens dataset statistics.
    """

    return {
        "num_ratings": len(ratings),
        "num_users": ratings["user_id"].nunique(),
        "num_items": ratings["item_id"].nunique(),
        "rating_min": ratings["rating"].min(),
        "rating_max": ratings["rating"].max(),
        "rating_mean": ratings["rating"].mean(),
    }

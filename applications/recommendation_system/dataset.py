import numpy as np


def create_rating_matrix():
    """
    Create a synthetic user-item rating matrix.

    Rows represent users.
    Columns represent movies/items.

    Ratings range from 1 to 5.
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
    in the rating matrix.
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
    in the rating matrix.
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

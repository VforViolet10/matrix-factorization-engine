import numpy as np

from matrix_factorization.decompositions import SVD, NMF

from .preprocessing import fill_missing_with_item_mean


class MatrixFactorizationRecommender:
    """
    Recommendation system based on matrix factorization.

    Supports:
    - SVD
    - NMF

    Missing ratings are represented by zero and are
    imputed using item means before factorization.
    """

    def __init__(
        self,
        method="svd",
        n_components=2,
        max_iter=500,
        random_state=42,
    ):
        method = method.lower()

        if method not in {"svd", "nmf"}:
            raise ValueError(
                "method must be either 'svd' or 'nmf'."
            )

        if n_components <= 0:
            raise ValueError(
                "n_components must be greater than zero."
            )

        self.method = method
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state

        self.model = None
        self.predicted_ratings = None
        self.training_ratings = None

    def fit(self, ratings):
        """
        Fit the recommendation model.
        """

        ratings = np.asarray(
            ratings,
            dtype=float,
        )

        if ratings.ndim != 2:
            raise ValueError(
                "ratings must be a 2D matrix."
            )

        if np.any(ratings < 0):
            raise ValueError(
                "ratings must contain non-negative values."
            )

        self.training_ratings = (
            fill_missing_with_item_mean(
                ratings
            )
        )

        if self.method == "svd":
            self.model = SVD(
                n_components=self.n_components
            )

        else:
            self.model = NMF(
                n_components=self.n_components,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )

        self.model.fit(
            self.training_ratings
        )

        predictions = (
            self.model.reconstruct()
        )

        self.predicted_ratings = np.clip(
            predictions,
            1.0,
            5.0,
        )

        return self

    def predict(
        self,
        user_index,
        item_index,
    ):
        """
        Predict a rating for a user-item pair.
        """

        if self.predicted_ratings is None:
            raise ValueError(
                "Model must be fitted before prediction."
            )

        return float(
            self.predicted_ratings[
                user_index,
                item_index,
            ]
        )

    def recommend(
        self,
        user_index,
        item_names,
        ratings,
        n=3,
    ):
        """
        Generate Top-N recommendations.

        Items already rated by the user are excluded.
        """

        if self.predicted_ratings is None:
            raise ValueError(
                "Model must be fitted before recommendation."
            )

        ratings = np.asarray(
            ratings,
            dtype=float,
        )

        if user_index < 0 or user_index >= ratings.shape[0]:
            raise IndexError(
                "user_index is out of range."
            )

        if len(item_names) != ratings.shape[1]:
            raise ValueError(
                "item_names must match the number of items."
            )

        if n <= 0:
            raise ValueError(
                "n must be greater than zero."
            )

        unrated_items = np.where(
            ratings[user_index] == 0
        )[0]

        predictions = [
            (
                item_names[item_index],
                self.predicted_ratings[
                    user_index,
                    item_index,
                ],
            )
            for item_index in unrated_items
        ]

        predictions.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        return predictions[:n]

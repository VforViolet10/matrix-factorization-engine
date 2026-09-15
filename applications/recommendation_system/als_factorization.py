import numpy as np


class ALSMatrixFactorization:
    """
    Matrix factorization using Alternating Least Squares (ALS).

    The model learns:

        R ≈ U @ V.T

    where:
        R = user-item rating matrix
        U = user latent-factor matrix
        V = item latent-factor matrix

    Only observed ratings contribute to the optimization.
    Zero values are treated as missing ratings.
    """

    def __init__(
        self,
        n_components=2,
        regularization=0.1,
        max_iter=50,
        tolerance=1e-5,
        random_state=42,
    ):
        if n_components <= 0:
            raise ValueError(
                "n_components must be greater than zero."
            )

        if regularization < 0:
            raise ValueError(
                "regularization must be non-negative."
            )

        if max_iter <= 0:
            raise ValueError(
                "max_iter must be greater than zero."
            )

        if tolerance < 0:
            raise ValueError(
                "tolerance must be non-negative."
            )

        self.n_components = n_components
        self.regularization = regularization
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_state = random_state

        self.user_factors = None
        self.item_factors = None
        self.loss_history_ = []
        self.n_iter_ = 0

    def fit(self, ratings):
        """
        Fit the ALS matrix factorization model.

        Parameters
        ----------
        ratings : numpy.ndarray
            User-item rating matrix.
            Zero values represent missing ratings.

        Returns
        -------
        ALSMatrixFactorization
            Fitted model.
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

        observed = ratings > 0

        if not np.any(observed):
            raise ValueError(
                "ratings must contain at least one observed rating."
            )

        n_users, n_items = ratings.shape

        rng = np.random.default_rng(
            self.random_state
        )

        self.user_factors = rng.normal(
            0,
            0.01,
            size=(
                n_users,
                self.n_components,
            ),
        )

        self.item_factors = rng.normal(
            0,
            0.01,
            size=(
                n_items,
                self.n_components,
            ),
        )

        self.loss_history_ = []
        self.n_iter_ = 0

        identity = np.eye(
            self.n_components
        )

        previous_loss = None

        for iteration in range(
            self.max_iter
        ):

            # ---------------------------------
            # Update user factors
            # ---------------------------------

            for user in range(
                n_users
            ):

                item_indices = np.where(
                    observed[user]
                )[0]

                if len(item_indices) == 0:
                    continue

                V = self.item_factors[
                    item_indices
                ]

                r = ratings[
                    user,
                    item_indices
                ]

                A = (
                    V.T @ V
                    + self.regularization
                    * identity
                )

                b = V.T @ r

                self.user_factors[user] = (
                    np.linalg.solve(
                        A,
                        b,
                    )
                )

            # ---------------------------------
            # Update item factors
            # ---------------------------------

            for item in range(
                n_items
            ):

                user_indices = np.where(
                    observed[:, item]
                )[0]

                if len(user_indices) == 0:
                    continue

                U = self.user_factors[
                    user_indices
                ]

                r = ratings[
                    user_indices,
                    item
                ]

                A = (
                    U.T @ U
                    + self.regularization
                    * identity
                )

                b = U.T @ r

                self.item_factors[item] = (
                    np.linalg.solve(
                        A,
                        b,
                    )
                )

            # ---------------------------------
            # Calculate observed-rating loss
            # ---------------------------------

            predictions = (
                self.user_factors
                @ self.item_factors.T
            )

            errors = np.zeros_like(
                ratings
            )

            errors[observed] = (
                ratings[observed]
                - predictions[observed]
            )

            loss = (
                np.sum(
                    errors[observed] ** 2
                )
                + self.regularization
                * (
                    np.sum(
                        self.user_factors ** 2
                    )
                    + np.sum(
                        self.item_factors ** 2
                    )
                )
            )

            self.loss_history_.append(
                float(loss)
            )

            self.n_iter_ = iteration + 1

            if previous_loss is not None:

                improvement = (
                    previous_loss
                    - loss
                )

                if (
                    improvement >= 0
                    and improvement < self.tolerance
                ):
                    break

            previous_loss = loss

        return self

    def reconstruct(self):
        """
        Reconstruct the complete rating matrix.
        """

        if (
            self.user_factors is None
            or self.item_factors is None
        ):
            raise ValueError(
                "Model must be fitted before reconstruction."
            )

        return (
            self.user_factors
            @ self.item_factors.T
        )

    def predict(
        self,
        user_index,
        item_index,
    ):
        """
        Predict one user-item rating.
        """

        predictions = self.reconstruct()

        return float(
            predictions[
                user_index,
                item_index,
            ]
        )

    def reconstruction_error(
        self,
        ratings,
    ):
        """
        Calculate RMSE only on observed ratings.
        """

        ratings = np.asarray(
            ratings,
            dtype=float,
        )

        predictions = self.reconstruct()

        if ratings.shape != predictions.shape:
            raise ValueError(
                "ratings must have the same shape "
                "as the reconstructed matrix."
            )

        observed = ratings > 0

        if not np.any(observed):
            raise ValueError(
                "ratings must contain at least one "
                "observed rating."
            )

        errors = (
            predictions[observed]
            - ratings[observed]
        )

        return float(
            np.sqrt(
                np.mean(
                    errors ** 2
                )
            )
        )

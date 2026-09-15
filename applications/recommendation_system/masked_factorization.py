import numpy as np


class MaskedMatrixFactorization:
    """
    Matrix factorization for sparse user-item ratings.

    The model learns:

        R ≈ U @ V.T

    Only observed ratings contribute to the loss.
    Zero values are treated as missing ratings.

    Gradient descent is used to learn the latent
    user and item factors.
    """

    def __init__(
        self,
        n_components=2,
        learning_rate=0.001,
        regularization=0.01,
        max_iter=1000,
        tolerance=1e-5,
        random_state=42,
    ):
        if n_components <= 0:
            raise ValueError(
                "n_components must be greater than zero."
            )

        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than zero."
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
        self.learning_rate = learning_rate
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
        Fit the matrix factorization model.

        Parameters
        ----------
        ratings : numpy.ndarray
            User-item rating matrix.
            Zero values represent missing ratings.

        Returns
        -------
        MaskedMatrixFactorization
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
            loc=0.0,
            scale=0.01,
            size=(
                n_users,
                self.n_components,
            ),
        )

        self.item_factors = rng.normal(
            loc=0.0,
            scale=0.01,
            size=(
                n_items,
                self.n_components,
            ),
        )

        self.loss_history_ = []
        self.n_iter_ = 0

        previous_loss = None

        for iteration in range(
            self.max_iter
        ):

            predictions = (
                self.user_factors
                @ self.item_factors.T
            )

            error = np.zeros_like(
                ratings
            )

            error[observed] = (
                ratings[observed]
                - predictions[observed]
            )

            loss = (
                np.sum(
                    error[observed] ** 2
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

            user_gradient = (
                -2.0
                * (
                    error
                    @ self.item_factors
                )
                + 2.0
                * self.regularization
                * self.user_factors
            )

            item_gradient = (
                -2.0
                * (
                    error.T
                    @ self.user_factors
                )
                + 2.0
                * self.regularization
                * self.item_factors
            )

            self.user_factors -= (
                self.learning_rate
                * user_gradient
            )

            self.item_factors -= (
                self.learning_rate
                * item_gradient
            )

            self.n_iter_ = iteration + 1

            if not np.all(
                np.isfinite(
                    self.user_factors
                )
            ) or not np.all(
                np.isfinite(
                    self.item_factors
                )
            ):
                raise FloatingPointError(
                    "Matrix factorization became "
                    "numerically unstable. "
                    "Try reducing learning_rate."
                )

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

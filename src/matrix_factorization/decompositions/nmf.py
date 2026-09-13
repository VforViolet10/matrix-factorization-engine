import numpy as np


class NMF:
    """
    Non-negative Matrix Factorization using multiplicative updates.

    Factorizes a non-negative matrix A approximately as:

        A ≈ W @ H

    where:
        W >= 0
        H >= 0
    """

    def __init__(
        self,
        n_components,
        max_iter=500,
        tol=1e-4,
        random_state=None,
        epsilon=1e-10,
    ):
        if n_components <= 0:
            raise ValueError("n_components must be positive.")

        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")

        if tol <= 0:
            raise ValueError("tol must be positive.")

        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.epsilon = epsilon

        self.W = None
        self.H = None
        self.reconstruction_error_ = None
        self.n_iter_ = 0
        self.error_history_ = []

    def _validate_input(self, A):
        """Validate that A is a non-negative 2D matrix."""

        A = np.asarray(A, dtype=float)

        if A.ndim != 2:
            raise ValueError("Input must be a 2D matrix.")

        if not np.all(np.isfinite(A)):
            raise ValueError("Input contains NaN or infinite values.")

        if np.any(A < 0):
            raise ValueError("NMF requires a non-negative matrix.")

        return A

    def fit(self, A):
        """
        Fit NMF to matrix A.

        Parameters
        ----------
        A : numpy.ndarray
            Non-negative input matrix.

        Returns
        -------
        self
        """

        A = self._validate_input(A)

        m, n = A.shape

        if self.n_components > min(m, n):
            raise ValueError(
                "n_components cannot exceed min(rows, columns)."
            )

        rng = np.random.default_rng(self.random_state)

        self.W = rng.random((m, self.n_components))
        self.H = rng.random((self.n_components, n))

        self.error_history_ = []

        previous_error = None

        for iteration in range(1, self.max_iter + 1):

            # Update H
            numerator = self.W.T @ A
            denominator = self.W.T @ self.W @ self.H

            self.H *= numerator / (denominator + self.epsilon)

            # Update W
            numerator = A @ self.H.T
            denominator = self.W @ self.H @ self.H.T

            self.W *= numerator / (denominator + self.epsilon)

            # Calculate reconstruction error
            reconstructed = self.W @ self.H

            error = np.linalg.norm(A - reconstructed, ord="fro")

            self.error_history_.append(error)

            if previous_error is not None:
                relative_change = abs(previous_error - error) / (
                    previous_error + self.epsilon
                )

                if relative_change < self.tol:
                    break

            previous_error = error

        self.n_iter_ = iteration
        self.reconstruction_error_ = self.error_history_[-1]

        return self

    def reconstruct(self):
        """
        Reconstruct the original matrix.

        Returns
        -------
        numpy.ndarray
            Reconstructed matrix.
        """

        if self.W is None or self.H is None:
            raise ValueError("Model has not been fitted yet.")

        return self.W @ self.H

    def fit_transform(self, A):
        """
        Fit the model and return W.
        """

        self.fit(A)
        return self.W

    def reconstruction_error(self, A):
        """
        Calculate Frobenius reconstruction error.
        """

        if self.W is None or self.H is None:
            raise ValueError("Model has not been fitted yet.")

        A = self._validate_input(A)

        return np.linalg.norm(A - self.reconstruct(), ord="fro")

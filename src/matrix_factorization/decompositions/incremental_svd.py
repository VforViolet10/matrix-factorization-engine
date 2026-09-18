"""
Incremental Singular Value Decomposition.

Provides an IncrementalSVD implementation for updating a low-rank
matrix factorization as new rows are observed.
"""

import numpy as np


class IncrementalSVD:
    """
    Incremental SVD using rank-k truncated updates.

    Parameters
    ----------
    n_components : int
        Number of singular components to retain.
    """

    def __init__(self, n_components=2):
        if n_components < 1:
            raise ValueError("n_components must be at least 1")

        self.n_components = n_components
        self.U_ = None
        self.s_ = None
        self.Vt_ = None
        self.n_samples_seen_ = 0

    def fit(self, X):
        """Fit the initial SVD decomposition."""
        X = self._validate_input(X)

        U, s, Vt = np.linalg.svd(X, full_matrices=False)

        rank = min(
            self.n_components,
            len(s),
            X.shape[0],
            X.shape[1],
        )

        self.U_ = U[:, :rank]
        self.s_ = s[:rank]
        self.Vt_ = Vt[:rank, :]
        self.n_samples_seen_ = X.shape[0]

        return self

    def partial_fit(self, X):
        """Update the factorization with additional rows."""
        X = self._validate_input(X)

        if self.U_ is None:
            return self.fit(X)

        if X.shape[1] != self.Vt_.shape[1]:
            raise ValueError(
                "New data must have the same number of features "
                "as the original data"
            )

        old_matrix = self.reconstruct()
        combined = np.vstack([old_matrix, X])

        U, s, Vt = np.linalg.svd(combined, full_matrices=False)

        rank = min(
            self.n_components,
            len(s),
            combined.shape[0],
            combined.shape[1],
        )

        self.U_ = U[:, :rank]
        self.s_ = s[:rank]
        self.Vt_ = Vt[:rank, :]
        self.n_samples_seen_ = combined.shape[0]

        return self

    def transform(self, X):
        """Project observations into the learned latent space."""
        self._check_fitted()

        X = self._validate_input(X)

        if X.shape[1] != self.Vt_.shape[1]:
            raise ValueError(
                "Input must have the same number of features "
                "as the fitted data"
            )

        return X @ self.Vt_.T

    def reconstruct(self):
        """Reconstruct the matrix represented by the current factors."""
        self._check_fitted()

        return self.U_ @ np.diag(self.s_) @ self.Vt_

    def explained_variance_ratio(self):
        """Return the fraction of captured squared singular values."""
        self._check_fitted()

        squared = self.s_ ** 2
        total = np.sum(squared)

        if total == 0:
            return np.zeros_like(self.s_)

        return squared / total

    def _check_fitted(self):
        if self.U_ is None or self.s_ is None or self.Vt_ is None:
            raise RuntimeError(
                "IncrementalSVD must be fitted before this operation"
            )

    @staticmethod
    def _validate_input(X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional matrix")

        if X.size == 0:
            raise ValueError("X cannot be empty")

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")

        return X

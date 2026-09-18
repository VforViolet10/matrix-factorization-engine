"""
Robust Principal Component Analysis.

Decomposes a matrix into a low-rank component and a sparse component
using Principal Component Pursuit solved with ADMM.
"""

import numpy as np


class RobustPCA:
    """
    Robust PCA using Principal Component Pursuit.

    Solves approximately:

        X = L + S

    where L is low-rank and S is sparse.

    Parameters
    ----------
    mu : float or None
        ADMM penalty parameter. If None, a value is estimated from X.
    lam : float or None
        Sparsity regularization parameter. If None, uses
        1 / sqrt(max(n_samples, n_features)).
    max_iter : int
        Maximum number of ADMM iterations.
    tol : float
        Convergence tolerance.
    """

    def __init__(self, mu=None, lam=None, max_iter=1000, tol=1e-7):
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")

        if tol <= 0:
            raise ValueError("tol must be positive")

        if mu is not None and mu <= 0:
            raise ValueError("mu must be positive")

        if lam is not None and lam <= 0:
            raise ValueError("lam must be positive")

        self.mu = mu
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol

        self.L_ = None
        self.S_ = None
        self.n_iter_ = 0
        self.convergence_history_ = []

    def fit(self, X):
        """
        Decompose X into low-rank and sparse components.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = self._validate_input(X)

        n_samples, n_features = X.shape

        lam = self.lam
        if lam is None:
            lam = 1.0 / np.sqrt(max(n_samples, n_features))

        mu = self.mu
        if mu is None:
            mu = (
                n_samples * n_features
            ) / (
                4.0 * np.sum(np.abs(X)) + 1e-12
            )

        L = np.zeros_like(X)
        S = np.zeros_like(X)
        Y = np.zeros_like(X)

        norm_X = np.linalg.norm(X, ord="fro")

        if norm_X == 0:
            self.L_ = np.zeros_like(X)
            self.S_ = np.zeros_like(X)
            self.n_iter_ = 0
            self.convergence_history_ = [0.0]
            return self

        self.convergence_history_ = []

        for iteration in range(1, self.max_iter + 1):
            # Low-rank update using singular-value thresholding.
            U, singular_values, Vt = np.linalg.svd(
                X - S + Y / mu,
                full_matrices=False,
            )

            thresholded = np.maximum(
                singular_values - 1.0 / mu,
                0.0,
            )

            rank = np.sum(thresholded > 0)

            if rank == 0:
                L = np.zeros_like(X)
            else:
                L = (
                    U[:, :rank]
                    @ np.diag(thresholded[:rank])
                    @ Vt[:rank, :]
                )

            # Sparse update using element-wise soft thresholding.
            residual = X - L + Y / mu

            S = self._soft_threshold(
                residual,
                lam / mu,
            )

            # Dual-variable update.
            constraint = X - L - S
            Y = Y + mu * constraint

            error = np.linalg.norm(
                constraint,
                ord="fro",
            ) / norm_X

            self.convergence_history_.append(error)

            if error < self.tol:
                self.n_iter_ = iteration
                break
        else:
            self.n_iter_ = self.max_iter

        self.L_ = L
        self.S_ = S

        return self

    def reconstruct(self):
        """
        Reconstruct the original matrix from L + S.

        Returns
        -------
        ndarray
            Reconstructed matrix.
        """
        self._check_fitted()

        return self.L_ + self.S_

    def reconstruction_error(self, X):
        """
        Calculate relative reconstruction error.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        float
            Relative Frobenius reconstruction error.
        """
        self._check_fitted()

        X = self._validate_input(X)

        if X.shape != self.L_.shape:
            raise ValueError(
                "X must have the same shape as the fitted matrix"
            )

        denominator = np.linalg.norm(X, ord="fro")

        if denominator == 0:
            return 0.0

        return np.linalg.norm(
            X - self.reconstruct(),
            ord="fro",
        ) / denominator

    def rank(self):
        """
        Return the numerical rank of the low-rank component.

        Returns
        -------
        int
            Numerical rank.
        """
        self._check_fitted()

        singular_values = np.linalg.svd(
            self.L_,
            compute_uv=False,
        )

        if singular_values.size == 0:
            return 0

        threshold = (
            np.finfo(float).eps
            * max(self.L_.shape)
            * singular_values[0]
        )

        return int(np.sum(singular_values > threshold))

    def sparsity(self, threshold=1e-8):
        """
        Return the fraction of entries considered sparse.

        Parameters
        ----------
        threshold : float
            Absolute-value threshold for considering an entry non-zero.

        Returns
        -------
        float
            Fraction of non-zero entries in S.
        """
        self._check_fitted()

        if threshold < 0:
            raise ValueError("threshold must be non-negative")

        return float(
            np.mean(np.abs(self.S_) > threshold)
        )

    def _check_fitted(self):
        if self.L_ is None or self.S_ is None:
            raise RuntimeError(
                "RobustPCA must be fitted before this operation"
            )

    @staticmethod
    def _soft_threshold(X, threshold):
        return np.sign(X) * np.maximum(
            np.abs(X) - threshold,
            0.0,
        )

    @staticmethod
    def _validate_input(X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2-dimensional matrix"
            )

        if X.size == 0:
            raise ValueError("X cannot be empty")

        if not np.all(np.isfinite(X)):
            raise ValueError(
                "X must contain only finite values"
            )

        return X

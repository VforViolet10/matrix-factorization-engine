"""
Unified Matrix Factorization Engine.

Provides a common interface for selecting and running supported
matrix factorization algorithms.
"""

import numpy as np

from .decompositions import (
    SVD,
    NMF,
    QR,
    EigenvalueDecomposition,
    LU,
    IncrementalSVD,
    RobustPCA,
)


class MatrixFactorizationEngine:
    """
    Unified interface for matrix factorization algorithms.

    Parameters
    ----------
    method : str
        Factorization method to use.

        Supported methods:
        - "svd"
        - "nmf"
        - "qr"
        - "eigen"
        - "lu"
        - "incremental_svd"
        - "robust_pca"

    **kwargs
        Parameters passed to the selected decomposition.
    """

    _METHODS = {
        "svd": SVD,
        "nmf": NMF,
        "qr": QR,
        "eigen": EigenvalueDecomposition,
        "lu": LU,
        "incremental_svd": IncrementalSVD,
        "robust_pca": RobustPCA,
    }

    def __init__(self, method="svd", **kwargs):
        method = method.lower()

        if method not in self._METHODS:
            supported = ", ".join(sorted(self._METHODS))
            raise ValueError(
                f"Unknown factorization method '{method}'. "
                f"Supported methods: {supported}"
            )

        self.method = method
        self.model = self._METHODS[method](**kwargs)
        self._fitted = False

    @classmethod
    def available_methods(cls):
        """Return the names of all supported methods."""
        return sorted(cls._METHODS)

    def fit(self, X):
        """
        Fit the selected factorization.

        Parameters
        ----------
        X : array-like
            Input matrix.

        Returns
        -------
        MatrixFactorizationEngine
            The fitted engine.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional matrix")

        if X.size == 0:
            raise ValueError("X cannot be empty")

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")

        self.model.fit(X)
        self._fitted = True

        return self

    def partial_fit(self, X):
        """
        Incrementally update the factorization.

        Currently supported by Incremental SVD.

        Parameters
        ----------
        X : array-like
            New observations.

        Returns
        -------
        MatrixFactorizationEngine
            The updated engine.
        """
        if self.method != "incremental_svd":
            raise NotImplementedError(
                "partial_fit is currently supported only "
                "for 'incremental_svd'"
            )

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional matrix")

        if X.size == 0:
            raise ValueError("X cannot be empty")

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")

        self.model.partial_fit(X)
        self._fitted = True

        return self

    def transform(self, X):
        """
        Transform data using the fitted factorization.

        Currently supported by SVD, Incremental SVD, and
        decomposition classes that expose a transform method.
        """
        self._check_fitted()

        if not hasattr(self.model, "transform"):
            raise NotImplementedError(
                f"Transform is not supported for '{self.method}'"
            )

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional matrix")

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")

        return self.model.transform(X)

    def reconstruct(self):
        """
        Reconstruct the matrix from the fitted factorization.

        Returns
        -------
        ndarray
            Reconstructed matrix.
        """
        self._check_fitted()

        if self.method == "robust_pca":
            return self.model.reconstruct()

        if self.method == "lu":
            return self.model.P.T @ self.model.L @ self.model.U

        if hasattr(self.model, "reconstruct"):
            return self.model.reconstruct()

        raise NotImplementedError(
            f"Reconstruction is not supported for '{self.method}'"
        )

    def get_model(self):
        """
        Return the underlying decomposition object.

        Returns
        -------
        object
            Fitted decomposition model.
        """
        return self.model

    def _check_fitted(self):
        """Raise an error if the engine has not been fitted."""
        if not self._fitted:
            raise RuntimeError(
                "MatrixFactorizationEngine must be fitted before "
                "this operation"
            )

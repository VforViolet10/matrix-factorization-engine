import numpy as np


class SVD:
    """
    Singular Value Decomposition.

    Decomposes a matrix A as:

        A = U @ Sigma @ Vt

    where:
        U  = left singular vectors
        Sigma = singular values
        Vt = transpose of right singular vectors
    """

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.U = None
        self.singular_values = None
        self.Vt = None

    def fit(self, A):
        """
        Fit SVD to matrix A.

        Parameters
        ----------
        A : numpy.ndarray
            Input 2D matrix.

        Returns
        -------
        self
        """

        A = np.asarray(A, dtype=float)

        if A.ndim != 2:
            raise ValueError("Input must be a 2D matrix.")

        m, n = A.shape
        max_components = min(m, n)

        if self.n_components is None:
            k = max_components
        else:
            if self.n_components <= 0:
                raise ValueError(
                    "n_components must be a positive integer."
                )

            if self.n_components > max_components:
                raise ValueError(
                    "n_components cannot exceed min(rows, columns)."
                )

            k = self.n_components

        U, s, Vt = np.linalg.svd(A, full_matrices=False)

        self.U = U[:, :k]
        self.singular_values = s[:k]
        self.Vt = Vt[:k, :]

        return self

    def reconstruct(self):
        """
        Reconstruct the matrix from the factorized components.

        Returns
        -------
        numpy.ndarray
            Reconstructed matrix.
        """

        if self.U is None:
            raise ValueError("Model has not been fitted yet.")

        return self.U @ np.diag(self.singular_values) @ self.Vt

    def transform(self, A):
        """
        Project A onto the learned right singular vectors.

        Parameters
        ----------
        A : numpy.ndarray
            Input matrix.

        Returns
        -------
        numpy.ndarray
            Transformed representation.
        """

        if self.Vt is None:
            raise ValueError("Model has not been fitted yet.")

        A = np.asarray(A, dtype=float)

        return A @ self.Vt.T

    def explained_variance_ratio(self):
        """
        Calculate the proportion of variance explained
        by each singular value.
        """

        if self.singular_values is None:
            raise ValueError("Model has not been fitted yet.")

        squared = self.singular_values ** 2

        return squared / np.sum(squared)

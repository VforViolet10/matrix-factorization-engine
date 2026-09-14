import numpy as np


class EigenvalueDecomposition:
    """
    Eigenvalue Decomposition of a square matrix.

    For a diagonalizable matrix A:

        A = V @ Lambda @ V^(-1)

    where:
    - V contains the eigenvectors
    - Lambda is a diagonal matrix of eigenvalues
    """

    def __init__(self, tol=1e-10):
        self.tol = tol
        self.eigenvalues = None
        self.eigenvectors = None
        self.eigenvalue_matrix = None

    def fit(self, A):
        """
        Compute the eigenvalue decomposition of matrix A.

        Parameters
        ----------
        A : array-like
            Square input matrix.

        Returns
        -------
        EigenvalueDecomposition
            Fitted decomposition object.
        """

        A = np.asarray(A, dtype=float)

        # Check that A is a matrix
        if A.ndim != 2:
            raise ValueError("Input matrix must be 2-dimensional.")

        # Check that A is square
        rows, cols = A.shape

        if rows != cols:
            raise ValueError("Eigenvalue decomposition requires a square matrix.")

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(A)

        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.eigenvalue_matrix = np.diag(eigenvalues)

        return self

    def reconstruct(self):
        """
        Reconstruct the original matrix from the eigenvalue decomposition.

        A = V @ Lambda @ V^(-1)
        """

        if self.eigenvalues is None or self.eigenvectors is None:
            raise ValueError(
                "Eigenvalue decomposition has not been fitted."
            )

        V = self.eigenvectors
        Lambda = self.eigenvalue_matrix

        return V @ Lambda @ np.linalg.inv(V)

    def eigenvector_residuals(self):
        """
        Calculate the residual for each eigenpair.

        For every eigenvalue/eigenvector pair:

            A @ v ≈ lambda @ v

        Returns
        -------
        numpy.ndarray
            Residual for each eigenpair.
        """

        if self.eigenvalues is None or self.eigenvectors is None:
            raise ValueError(
                "Eigenvalue decomposition has not been fitted."
            )

        A = self.reconstruct()

        residuals = []

        for i in range(len(self.eigenvalues)):
            v = self.eigenvectors[:, i]
            eigenvalue = self.eigenvalues[i]

            residual = np.linalg.norm(
                A @ v - eigenvalue * v
            )

            residuals.append(residual)

        return np.array(residuals)

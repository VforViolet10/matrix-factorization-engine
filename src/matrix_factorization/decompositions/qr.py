import numpy as np


class QR:
    """
    QR Factorization using the Modified Gram-Schmidt algorithm.

    For a matrix A, computes:

        A = QR

    where Q has orthonormal columns and R is upper triangular.
    """

    def __init__(self, tol=1e-10):
        self.tol = tol
        self.Q = None
        self.R = None

    def fit(self, A):
        """
        Compute the QR factorization of matrix A.
        """

        A = np.asarray(A, dtype=float)

        if A.ndim != 2:
            raise ValueError("Input matrix must be 2-dimensional.")

        m, n = A.shape

        if m < n:
            raise ValueError(
                "QR decomposition requires rows >= columns."
            )

        Q = np.zeros((m, n))
        R = np.zeros((n, n))

        V = A.copy()

        for i in range(n):

            R[i, i] = np.linalg.norm(V[:, i])

            if R[i, i] < self.tol:
                raise ValueError(
                    "Matrix is rank-deficient or numerically singular."
                )

            Q[:, i] = V[:, i] / R[i, i]

            for j in range(i + 1, n):
                R[i, j] = np.dot(Q[:, i], V[:, j])
                V[:, j] = V[:, j] - R[i, j] * Q[:, i]

        self.Q = Q
        self.R = R

        return self

    def reconstruct(self):
        """
        Reconstruct the original matrix from Q and R.
        """

        if self.Q is None or self.R is None:
            raise ValueError(
                "QR decomposition has not been fitted."
            )

        return self.Q @ self.R

    def orthogonality_error(self):
        """
        Calculate the orthogonality error of Q.
        """

        if self.Q is None:
            raise ValueError(
                "QR decomposition has not been fitted."
            )

        identity = np.eye(self.Q.shape[1])

        return np.linalg.norm(
            self.Q.T @ self.Q - identity
        )

import numpy as np


class LU:
    """
    LU Factorization with partial pivoting.

    For a square matrix A, computes:

        P @ A = L @ U

    where:
    - P is a permutation matrix
    - L is a lower triangular matrix
    - U is an upper triangular matrix
    """

    def __init__(self, tol=1e-10):
        self.tol = tol
        self.P = None
        self.L = None
        self.U = None

    def fit(self, A):
        """
        Compute the LU factorization of matrix A.

        Parameters
        ----------
        A : array-like
            Square input matrix.

        Returns
        -------
        LU
            Fitted LU decomposition object.
        """

        A = np.asarray(A, dtype=float)

        if A.ndim != 2:
            raise ValueError("Input matrix must be 2-dimensional.")

        rows, cols = A.shape

        if rows != cols:
            raise ValueError(
                "LU decomposition requires a square matrix."
            )

        n = rows

        U = A.copy()
        L = np.eye(n)
        P = np.eye(n)

        for i in range(n):

            # Find the row with the largest pivot
            pivot_row = i + np.argmax(
                np.abs(U[i:, i])
            )

            # Check for a numerically singular matrix
            if abs(U[pivot_row, i]) < self.tol:
                raise ValueError(
                    "Matrix is singular or numerically singular."
                )

            # Swap rows in U and P
            if pivot_row != i:
                U[[i, pivot_row], :] = U[[pivot_row, i], :]
                P[[i, pivot_row], :] = P[[pivot_row, i], :]

                # Swap previously calculated values in L
                if i > 0:
                    L[[i, pivot_row], :i] = (
                        L[[pivot_row, i], :i]
                    )

            # Eliminate entries below the pivot
            for j in range(i + 1, n):

                L[j, i] = U[j, i] / U[i, i]

                U[j, i:] = (
                    U[j, i:]
                    - L[j, i] * U[i, i:]
                )

                U[j, i] = 0.0

        self.P = P
        self.L = L
        self.U = U

        return self

    def reconstruct(self):
        """
        Reconstruct P @ A from L and U.

        Since:

            P @ A = L @ U

        this method returns L @ U.
        """

        if self.P is None or self.L is None or self.U is None:
            raise ValueError(
                "LU decomposition has not been fitted."
            )

        return self.L @ self.U

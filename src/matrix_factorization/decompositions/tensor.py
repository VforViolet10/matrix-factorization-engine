"""
Tensor Factorization using CP/PARAFAC decomposition.

This module implements a lightweight CP decomposition using
Alternating Least Squares (ALS).

For a 3-dimensional tensor X, CP decomposition approximates:

    X ≈ Σ_r λ_r * a_r ⊗ b_r ⊗ c_r

where:
    - R is the selected rank
    - λ_r are component weights
    - a_r, b_r, c_r are factor vectors
    - ⊗ denotes the outer product

The implementation is designed for educational and
research-oriented experimentation.
"""

from __future__ import annotations

import numpy as np


class TensorFactorization:
    """
    CP/PARAFAC tensor factorization using Alternating Least Squares.

    Parameters
    ----------
    rank : int
        Number of components in the CP decomposition.

    max_iter : int, default=100
        Maximum number of ALS iterations.

    tol : float, default=1e-6
        Convergence tolerance based on relative reconstruction error.

    random_state : int or None, default=None
        Random seed used for reproducible initialization.

    Attributes
    ----------
    factors_ : list of numpy.ndarray
        Factor matrices for each tensor mode.

    weights_ : numpy.ndarray
        Component weights.

    errors_ : list of float
        Reconstruction error recorded at each iteration.

    n_iter_ : int
        Number of iterations performed.

    fitted_ : bool
        Whether the model has been fitted.
    """

    def __init__(
        self,
        rank: int,
        max_iter: int = 100,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        if not isinstance(rank, int) or rank <= 0:
            raise ValueError("rank must be a positive integer.")

        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")

        if tol <= 0:
            raise ValueError("tol must be greater than zero.")

        self.rank = rank
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.factors_: list[np.ndarray] | None = None
        self.weights_: np.ndarray | None = None
        self.errors_: list[float] = []
        self.n_iter_: int = 0
        self.fitted_: bool = False

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_tensor(tensor: np.ndarray) -> np.ndarray:
        """
        Validate and convert input into a floating-point tensor.
        """

        tensor = np.asarray(tensor, dtype=float)

        if tensor.ndim < 2:
            raise ValueError(
                "Tensor must have at least two dimensions."
            )

        if any(size <= 0 for size in tensor.shape):
            raise ValueError(
                "All tensor dimensions must be greater than zero."
            )

        if not np.all(np.isfinite(tensor)):
            raise ValueError(
                "Tensor must contain only finite values."
            )

        return tensor

    # ------------------------------------------------------------------
    # Khatri-Rao product
    # ------------------------------------------------------------------

    @staticmethod
    def _khatri_rao(
        matrices: list[np.ndarray],
        skip_matrix: int | None = None,
    ) -> np.ndarray:
        """
        Compute the Khatri-Rao product.

        The Khatri-Rao product is a column-wise Kronecker product.

        Parameters
        ----------
        matrices : list of ndarray
            Factor matrices.

        skip_matrix : int or None
            Optional index of a matrix to exclude.

        Returns
        -------
        ndarray
            Khatri-Rao product.
        """

        selected = [
            matrix
            for index, matrix in enumerate(matrices)
            if index != skip_matrix
        ]

        if not selected:
            raise ValueError(
                "At least one matrix is required for the "
                "Khatri-Rao product."
            )

        result = selected[0]

        for matrix in selected[1:]:
            result = np.einsum(
                "ir,jr->ijr",
                result,
                matrix,
            ).reshape(
                result.shape[0] * matrix.shape[0],
                result.shape[1],
            )

        return result

    # ------------------------------------------------------------------
    # Mode-n unfolding
    # ------------------------------------------------------------------

    @staticmethod
    def _unfold(
        tensor: np.ndarray,
        mode: int,
    ) -> np.ndarray:
        """
        Unfold a tensor along a specified mode.

        Parameters
        ----------
        tensor : ndarray
            Input tensor.

        mode : int
            Mode along which the tensor is unfolded.

        Returns
        -------
        ndarray
            Mode-n unfolding.
        """

        if mode < 0 or mode >= tensor.ndim:
            raise ValueError(
                f"mode must be between 0 and {tensor.ndim - 1}."
            )

        axes = [mode] + [
            axis for axis in range(tensor.ndim)
            if axis != mode
        ]

        unfolded = np.transpose(tensor, axes)

        return unfolded.reshape(
            tensor.shape[mode],
            -1,
        )

    # ------------------------------------------------------------------
    # Tensor reconstruction
    # ------------------------------------------------------------------

    @staticmethod
    def reconstruct(
        factors: list[np.ndarray],
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Reconstruct a tensor from CP factor matrices.

        Parameters
        ----------
        factors : list of ndarray
            Factor matrices.

        weights : ndarray or None
            Optional component weights.

        Returns
        -------
        ndarray
            Reconstructed tensor.
        """

        if not factors:
            raise ValueError("At least one factor matrix is required.")

        rank = factors[0].shape[1]

        for factor in factors:
            if factor.ndim != 2:
                raise ValueError(
                    "Each factor matrix must be two-dimensional."
                )

            if factor.shape[1] != rank:
                raise ValueError(
                    "All factor matrices must have the same number "
                    "of components."
                )

        if weights is None:
            weights = np.ones(rank)

        weights = np.asarray(weights, dtype=float)

        if weights.shape != (rank,):
            raise ValueError(
                "weights must have one value for each component."
            )

        shape = tuple(factor.shape[0] for factor in factors)

        tensor = np.zeros(shape, dtype=float)

        for component in range(rank):
            component_tensor = weights[component]

            for factor in factors:
                component_tensor = np.multiply.outer(
                    component_tensor,
                    factor[:, component],
                )

            tensor += component_tensor

        return tensor

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, tensor: np.ndarray) -> "TensorFactorization":
        """
        Fit the CP decomposition using ALS.

        Parameters
        ----------
        tensor : ndarray
            Input tensor.

        Returns
        -------
        TensorFactorization
            Fitted model.
        """

        tensor = self._validate_tensor(tensor)

        if self.rank > min(tensor.shape):
            raise ValueError(
                "rank cannot be greater than the smallest "
                "tensor dimension."
            )

        rng = np.random.default_rng(self.random_state)

        # Random initialization of factor matrices.
        factors = [
            rng.random((dimension, self.rank))
            for dimension in tensor.shape
        ]

        # Normalize initial factors.
        for factor in factors:
            norms = np.linalg.norm(factor, axis=0)

            norms[norms == 0] = 1.0

            factor /= norms

        weights = np.ones(self.rank)

        self.errors_ = []

        previous_error = None

        for iteration in range(self.max_iter):

            for mode in range(tensor.ndim):

                # Mode-n unfolding.
                unfolded = self._unfold(tensor, mode)

                # Khatri-Rao product of all other factors.
                kr_product = self._khatri_rao(
                    factors,
                    skip_matrix=mode,
                )

                # Gram matrix.
                gram = np.ones(
                    (self.rank, self.rank),
                    dtype=float,
                )

                for factor_index, factor in enumerate(factors):
                    if factor_index == mode:
                        continue

                    gram *= factor.T @ factor

                # Numerical regularization.
                regularization = 1e-10 * np.eye(self.rank)

                # ALS update.
                updated_factor = (
                    unfolded @ kr_product
                    @ np.linalg.pinv(
                        gram + regularization
                    )
                )

                # Normalize factor columns.
                norms = np.linalg.norm(
                    updated_factor,
                    axis=0,
                )

                norms[norms == 0] = 1.0

                updated_factor /= norms

                factors[mode] = updated_factor

            # Calculate component weights.
            weights = np.ones(self.rank)

            for factor in factors:
                norms = np.linalg.norm(
                    factor,
                    axis=0,
                )

                norms[norms == 0] = 1.0

                weights *= norms

                factor /= norms

            # Reconstruction.
            reconstructed = self.reconstruct(
                factors,
                weights,
            )

            # Relative Frobenius reconstruction error.
            tensor_norm = np.linalg.norm(tensor)

            if tensor_norm == 0:
                error = np.linalg.norm(
                    tensor - reconstructed
                )
            else:
                error = (
                    np.linalg.norm(
                        tensor - reconstructed
                    )
                    / tensor_norm
                )

            self.errors_.append(float(error))

            self.n_iter_ = iteration + 1

            # Convergence check.
            if previous_error is not None:

                change = abs(
                    previous_error - error
                )

                if change < self.tol:
                    break

            previous_error = error

        self.factors_ = factors
        self.weights_ = weights
        self.fitted_ = True

        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def reconstruct_tensor(self) -> np.ndarray:
        """
        Reconstruct the fitted tensor.

        Returns
        -------
        ndarray
            Reconstructed tensor.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """

        if not self.fitted_:
            raise RuntimeError(
                "The model must be fitted before reconstruction."
            )

        if self.factors_ is None or self.weights_ is None:
            raise RuntimeError(
                "Factorization results are unavailable."
            )

        return self.reconstruct(
            self.factors_,
            self.weights_,
        )

    # ------------------------------------------------------------------
    # Reconstruction error
    # ------------------------------------------------------------------

    def reconstruction_error(
        self,
        tensor: np.ndarray,
    ) -> float:
        """
        Calculate relative Frobenius reconstruction error.

        Parameters
        ----------
        tensor : ndarray
            Original tensor.

        Returns
        -------
        float
            Relative reconstruction error.
        """

        if not self.fitted_:
            raise RuntimeError(
                "The model must be fitted before calculating "
                "reconstruction error."
            )

        tensor = self._validate_tensor(tensor)

        reconstructed = self.reconstruct_tensor()

        if tensor.shape != reconstructed.shape:
            raise ValueError(
                "Input tensor shape does not match the fitted tensor."
            )

        denominator = np.linalg.norm(tensor)

        if denominator == 0:
            return float(
                np.linalg.norm(
                    tensor - reconstructed
                )
            )

        return float(
            np.linalg.norm(
                tensor - reconstructed
            ) / denominator
        )

    # ------------------------------------------------------------------
    # Fit + reconstruct convenience method
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        tensor: np.ndarray,
    ) -> np.ndarray:
        """
        Fit the model and return the reconstructed tensor.

        Parameters
        ----------
        tensor : ndarray
            Input tensor.

        Returns
        -------
        ndarray
            Reconstructed tensor.
        """

        self.fit(tensor)

        return self.reconstruct_tensor()

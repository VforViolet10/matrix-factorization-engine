import numpy as np

from matrix_factorization.decompositions import (
    EigenvalueDecomposition
)


def main():
    # Create a square matrix
    A = np.array([
        [4.0, 1.0],
        [2.0, 3.0]
    ])

    print("Original Matrix A:")
    print(A)

    # Compute eigenvalue decomposition
    eigen = EigenvalueDecomposition().fit(A)

    print("\nEigenvalues:")
    print(eigen.eigenvalues)

    print("\nEigenvectors:")
    print(eigen.eigenvectors)

    print("\nEigenvalue Matrix (Lambda):")
    print(eigen.eigenvalue_matrix)

    # Reconstruct A
    reconstructed = eigen.reconstruct()

    print("\nReconstructed Matrix:")
    print(reconstructed)

    # Calculate reconstruction error
    reconstruction_error = np.linalg.norm(
        A - reconstructed
    )

    print("\nReconstruction Error:")
    print(reconstruction_error)

    # Calculate eigenvector residuals
    residuals = eigen.eigenvector_residuals()

    print("\nEigenvector Residuals:")
    print(residuals)


if __name__ == "__main__":
    main()

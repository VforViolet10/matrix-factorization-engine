import numpy as np

from matrix_factorization.decompositions import LU


def main():
    # Example matrix
    A = np.array([
        [4, 3],
        [6, 3]
    ], dtype=float)

    print("Original Matrix A:")
    print(A)

    # Perform LU decomposition
    lu = LU()
    lu.fit(A)

    print("\nPermutation Matrix P:")
    print(lu.P)

    print("\nLower Triangular Matrix L:")
    print(lu.L)

    print("\nUpper Triangular Matrix U:")
    print(lu.U)

    # Reconstruction
    PA = lu.P @ A
    LU_reconstructed = lu.reconstruct()

    print("\nP @ A:")
    print(PA)

    print("\nL @ U:")
    print(LU_reconstructed)

    # Reconstruction error
    error = np.linalg.norm(
        PA - LU_reconstructed
    )

    print("\nReconstruction Error:")
    print(error)


if __name__ == "__main__":
    main()

import numpy as np

from matrix_factorization.decompositions import QR


def main():
    # Create a sample matrix
    A = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 10.0],
        [2.0, 3.0, 4.0]
    ])

    print("Original Matrix A:")
    print(A)

    # Compute QR factorization
    qr = QR().fit(A)

    print("\nMatrix Q:")
    print(qr.Q)

    print("\nMatrix R:")
    print(qr.R)

    # Reconstruct A
    reconstructed = qr.reconstruct()

    print("\nReconstructed Matrix Q @ R:")
    print(reconstructed)

    # Calculate reconstruction error
    reconstruction_error = np.linalg.norm(
        A - reconstructed
    )

    print("\nReconstruction Error:")
    print(reconstruction_error)

    # Calculate orthogonality error
    orthogonality_error = qr.orthogonality_error()

    print("\nQ Orthogonality Error:")
    print(orthogonality_error)


if __name__ == "__main__":
    main()

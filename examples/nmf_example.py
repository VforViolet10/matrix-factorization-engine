import numpy as np

from matrix_factorization.decompositions import NMF


def main():
    # Create a non-negative matrix
    A = np.array(
        [
            [5, 3, 0, 1],
            [4, 0, 0, 1],
            [1, 1, 0, 5],
            [0, 0, 4, 4],
        ],
        dtype=float,
    )

    # Create and fit NMF
    model = NMF(
        n_components=2,
        max_iter=1000,
        tol=1e-5,
        random_state=42,
    )

    model.fit(A)

    # Reconstruct the matrix
    reconstructed = model.reconstruct()

    print("Original Matrix:")
    print(A)

    print("\nW Matrix:")
    print(model.W)

    print("\nH Matrix:")
    print(model.H)

    print("\nReconstructed Matrix:")
    print(reconstructed)

    print("\nIterations:", model.n_iter_)
    print("Reconstruction Error:", model.reconstruction_error_)


if __name__ == "__main__":
    main()

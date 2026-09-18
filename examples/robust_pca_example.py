import numpy as np

from matrix_factorization.decompositions import RobustPCA


def main():
    rng = np.random.default_rng(42)

    # Create a low-rank matrix.
    low_rank = np.outer(
        rng.normal(size=8),
        rng.normal(size=6),
    )

    # Add sparse anomalies/outliers.
    sparse = np.zeros((8, 6))
    sparse[1, 2] = 8.0
    sparse[5, 4] = -6.0
    sparse[6, 1] = 5.0

    X = low_rank + sparse

    model = RobustPCA(
        max_iter=1000,
        tol=1e-7,
    )

    print("=== Robust PCA ===")

    model.fit(X)

    print(f"Iterations: {model.n_iter_}")
    print(f"Estimated rank: {model.rank()}")
    print(f"Sparsity: {model.sparsity():.4f}")
    print(
        f"Reconstruction error: "
        f"{model.reconstruction_error(X):.8f}"
    )

    print("\n=== Low-Rank Component ===")
    print(np.round(model.L_, 3))

    print("\n=== Sparse Component ===")
    print(np.round(model.S_, 3))

    print("\n=== Reconstruction ===")
    print(np.round(model.reconstruct(), 3))

    print("\n=== Convergence ===")
    print(
        f"Initial error: "
        f"{model.convergence_history_[0]:.8e}"
    )
    print(
        f"Final error: "
        f"{model.convergence_history_[-1]:.8e}"
    )


if __name__ == "__main__":
    main()

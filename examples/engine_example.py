import numpy as np

from matrix_factorization import MatrixFactorizationEngine


def main():
    X = np.array([
        [5.0, 3.0, 1.0, 4.0],
        [4.0, 2.0, 1.0, 5.0],
        [1.0, 5.0, 4.0, 2.0],
        [5.0, 4.0, 2.0, 3.0],
    ])

    print("=== Available Methods ===")
    print(MatrixFactorizationEngine.available_methods())

    print("\n=== SVD ===")
    svd_engine = MatrixFactorizationEngine(
        method="svd",
        n_components=2,
    )
    svd_engine.fit(X)

    print(f"Method: {svd_engine.method}")
    print(f"Singular values: {svd_engine.model.singular_values}")
    print("Reconstruction:")
    print(np.round(svd_engine.reconstruct(), 3))

    print("\n=== NMF ===")
    nmf_engine = MatrixFactorizationEngine(
        method="nmf",
        n_components=2,
    )
    nmf_engine.fit(X)

    print(f"Method: {nmf_engine.method}")
    print("Reconstruction:")
    print(np.round(nmf_engine.reconstruct(), 3))

    print("\n=== Incremental SVD ===")
    incremental_engine = MatrixFactorizationEngine(
        method="incremental_svd",
        n_components=2,
    )

    incremental_engine.fit(X[:2])

    new_data = X[2:]
    incremental_engine.partial_fit(new_data)

    print(f"Samples seen: {incremental_engine.model.n_samples_seen_}")
    print("Reconstruction:")
    print(np.round(incremental_engine.reconstruct(), 3))

    print("\n=== Robust PCA ===")
    robust_engine = MatrixFactorizationEngine(
        method="robust_pca",
        max_iter=1000,
        tol=1e-7,
    )

    robust_engine.fit(X)

    print(f"Estimated rank: {robust_engine.model.rank()}")
    print(f"Sparsity: {robust_engine.model.sparsity():.4f}")
    print(
        f"Reconstruction error: "
        f"{robust_engine.model.reconstruction_error(X):.8f}"
    )


if __name__ == "__main__":
    main()

import numpy as np

from matrix_factorization.decompositions import IncrementalSVD


def main():
    # Initial batch of observations
    X_initial = np.array([
        [5.0, 3.0, 1.0, 4.0],
        [4.0, 2.0, 1.0, 5.0],
        [1.0, 5.0, 4.0, 2.0],
    ])

    # New observations arriving later
    X_new = np.array([
        [5.0, 4.0, 2.0, 3.0],
        [2.0, 5.0, 5.0, 1.0],
    ])

    model = IncrementalSVD(n_components=2)

    print("=== Initial Fit ===")
    model.fit(X_initial)

    print(f"Components: {model.n_components}")
    print(f"Samples seen: {model.n_samples_seen_}")
    print(f"Singular values: {model.s_}")
    print(f"Explained variance ratio: {model.explained_variance_ratio()}")

    print("\n=== Updating with New Data ===")
    model.partial_fit(X_new)

    print(f"Samples seen: {model.n_samples_seen_}")
    print(f"Singular values: {model.s_}")
    print(f"Explained variance ratio: {model.explained_variance_ratio()}")

    print("\n=== Reconstruction ===")
    reconstructed = model.reconstruct()

    print(reconstructed)

    print("\n=== Latent Representation ===")
    transformed = model.transform(X_new)

    print(transformed)


if __name__ == "__main__":
    main()

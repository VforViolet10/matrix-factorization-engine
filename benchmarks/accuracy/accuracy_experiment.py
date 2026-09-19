"""
Accuracy / Rank Benchmark

Measures how reconstruction quality changes with the number
of retained components for SVD and NMF.
"""

from pathlib import Path
import time
import tracemalloc

import numpy as np
import pandas as pd

from matrix_factorization.decompositions.svd import SVD
from matrix_factorization.decompositions.nmf import NMF


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

OUTPUT_FILE = RESULTS_DIR / "accuracy_results.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


MATRIX_SIZES = [50, 100, 200]
COMPONENTS = [5, 10, 20, 40]
REPEATS = 3


def generate_matrix(size, seed=42):
    """Generate a reproducible non-negative matrix."""

    rng = np.random.default_rng(seed)

    return rng.random((size, size))


def reconstruction_metrics(original, reconstructed):
    """Calculate reconstruction metrics."""

    difference = original - reconstructed

    error = np.linalg.norm(difference)

    relative_error = (
        error / np.linalg.norm(original)
    )

    rmse = np.sqrt(
        np.mean(difference ** 2)
    )

    return error, relative_error, rmse


def run_svd(matrix, components):
    """Fit SVD and reconstruct the matrix."""

    model = SVD(
        n_components=components
    )

    model.fit(matrix)

    return model.reconstruct()


def run_nmf(matrix, components):
    """Fit NMF and reconstruct the matrix."""

    model = NMF(
        n_components=components,
        random_state=42,
    )

    model.fit(matrix)

    return model.reconstruct()


def benchmark_algorithm(
    algorithm,
    matrix,
    components,
):
    """Benchmark one algorithm configuration."""

    runtimes = []
    memories = []
    metrics = []

    for _ in range(REPEATS):

        tracemalloc.start()

        start = time.perf_counter()

        if algorithm == "SVD":
            reconstructed = run_svd(
                matrix,
                components,
            )

        elif algorithm == "NMF":
            reconstructed = run_nmf(
                matrix,
                components,
            )

        else:
            tracemalloc.stop()

            raise ValueError(
                f"Unknown algorithm: {algorithm}"
            )

        runtime = time.perf_counter() - start

        _, peak_memory = (
            tracemalloc.get_traced_memory()
        )

        tracemalloc.stop()

        error, relative_error, rmse = (
            reconstruction_metrics(
                matrix,
                reconstructed,
            )
        )

        runtimes.append(runtime)

        memories.append(
            peak_memory / 1024
        )

        metrics.append(
            (
                error,
                relative_error,
                rmse,
            )
        )

    errors = [
        metric[0]
        for metric in metrics
    ]

    relative_errors = [
        metric[1]
        for metric in metrics
    ]

    rmses = [
        metric[2]
        for metric in metrics
    ]

    return {
        "runtime_seconds": np.mean(
            runtimes
        ),
        "runtime_std_seconds": np.std(
            runtimes
        ),
        "peak_memory_kb": np.mean(
            memories
        ),
        "peak_memory_std_kb": np.std(
            memories
        ),
        "reconstruction_error": np.mean(
            errors
        ),
        "relative_reconstruction_error": np.mean(
            relative_errors
        ),
        "rmse": np.mean(
            rmses
        ),
    }


def main():

    print("=" * 70)

    print(
        "MATRIX FACTORIZATION ENGINE - "
        "ACCURACY BENCHMARK"
    )

    print("=" * 70)

    rows = []

    for size in MATRIX_SIZES:

        print(
            f"\nMatrix Size: "
            f"{size} x {size}"
        )

        matrix = generate_matrix(size)

        for algorithm in [
            "SVD",
            "NMF",
        ]:

            for components in COMPONENTS:

                if components > size:
                    continue

                print(
                    f"  {algorithm:<5} "
                    f"k={components:<3}",
                    end=" ",
                    flush=True,
                )

                result = benchmark_algorithm(
                    algorithm,
                    matrix,
                    components,
                )

                row = {
                    "algorithm": algorithm,
                    "matrix_size": size,
                    "matrix_elements": (
                        size * size
                    ),
                    "components": components,
                    "repeats": REPEATS,
                    **result,
                }

                rows.append(row)

                print(
                    f"error="
                    f"{result['relative_reconstruction_error']:.6f} "
                    f"time="
                    f"{result['runtime_seconds']:.6f}s"
                )

    df = pd.DataFrame(rows)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    print(
        f"\nResults saved to:"
        f"\n{OUTPUT_FILE}"
    )

    print(
        f"\nTotal configurations: "
        f"{len(df)}"
    )


if __name__ == "__main__":
    main()

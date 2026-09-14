import csv
import os
import time
import tracemalloc

import matplotlib.pyplot as plt
import numpy as np

from matrix_factorization.decompositions import (
    SVD,
    NMF,
    QR,
    LU,
)

from matrix_factorization.metrics import (
    reconstruction_error,
    relative_reconstruction_error,
    rmse,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MATRIX_SIZES = [50, 100, 200]
N_COMPONENTS = 10
NMF_MAX_ITER = 200
N_REPEATS = 3

RESULTS_DIR = os.path.join(
    "benchmarks",
    "results"
)

CSV_PATH = os.path.join(
    RESULTS_DIR,
    "benchmark_results.csv"
)


# ---------------------------------------------------------
# Matrix generation
# ---------------------------------------------------------

def generate_benchmark_matrix(size, seed=42):
    """
    Generate a positive matrix suitable for all algorithms.

    The matrix is strictly positive so that NMF can be
    applied directly.
    """

    rng = np.random.default_rng(seed)

    return rng.random(
        (size, size)
    ) + 0.1


# ---------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------

def measure_algorithm(algorithm, A):
    """
    Measure runtime and peak memory usage.

    Parameters
    ----------
    algorithm : object
        Decomposition object with fit() and reconstruct().
    A : ndarray
        Input matrix.

    Returns
    -------
    tuple
        Runtime, peak memory, reconstructed matrix.
    """

    tracemalloc.start()

    start_time = time.perf_counter()

    algorithm.fit(A)

    reconstructed = algorithm.reconstruct()

    end_time = time.perf_counter()

    _, peak_memory = tracemalloc.get_traced_memory()

    tracemalloc.stop()

    runtime = end_time - start_time

    return (
        runtime,
        peak_memory / 1024,
        reconstructed,
    )


# ---------------------------------------------------------
# Algorithm factory
# ---------------------------------------------------------

def create_algorithms(size):
    """
    Create decomposition objects for a given matrix size.
    """

    components = min(
        N_COMPONENTS,
        size
    )

    return {
        "SVD": SVD(
            n_components=components
        ),

        "NMF": NMF(
            n_components=components,
            max_iter=NMF_MAX_ITER,
            random_state=42
        ),

        "QR": QR(),

        "LU": LU(),
    }


# ---------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------

def run_benchmarks():
    """
    Run benchmarks for all configured matrix sizes
    and decomposition algorithms.
    """

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    results = []

    print("=" * 70)
    print("MATRIX FACTORIZATION ENGINE - BENCHMARK")
    print("=" * 70)

    print(
        f"Matrix sizes: {MATRIX_SIZES}"
    )

    print(
        f"Repeats per configuration: {N_REPEATS}"
    )

    print()

    for size in MATRIX_SIZES:

        print("-" * 70)
        print(f"Matrix Size: {size} x {size}")
        print("-" * 70)

        A = generate_benchmark_matrix(
            size
        )

        algorithms = create_algorithms(
            size
        )

        for name, algorithm_factory in algorithms.items():

            runtimes = []
            memories = []
            errors = []
            relative_errors = []
            rmses = []

            print(
                f"\nRunning {name}..."
            )

            for repeat in range(N_REPEATS):

                # Create a fresh algorithm object
                # for every repetition.
                algorithm = create_algorithms(
                    size
                )[name]

                (
                    runtime,
                    memory,
                    A_hat,
                ) = measure_algorithm(
                    algorithm,
                    A
                )

                error = reconstruction_error(
                    A,
                    A_hat
                )

                relative_error = (
                    relative_reconstruction_error(
                        A,
                        A_hat
                    )
                )

                error_rmse = rmse(
                    A,
                    A_hat
                )

                runtimes.append(
                    runtime
                )

                memories.append(
                    memory
                )

                errors.append(
                    error
                )

                relative_errors.append(
                    relative_error
                )

                rmses.append(
                    error_rmse
                )

            average_runtime = np.mean(
                runtimes
            )

            average_memory = np.mean(
                memories
            )

            average_error = np.mean(
                errors
            )

            average_relative_error = np.mean(
                relative_errors
            )

            average_rmse = np.mean(
                rmses
            )

            if name in ["SVD", "NMF"]:
                components = min(
                    N_COMPONENTS,
                    size
                )
            else:
                components = size

            result = {
                "algorithm": name,
                "matrix_size": size,
                "components": components,
                "runtime_seconds": average_runtime,
                "peak_memory_kb": average_memory,
                "reconstruction_error": average_error,
                "relative_reconstruction_error": (
                    average_relative_error
                ),
                "rmse": average_rmse,
                "repeats": N_REPEATS,
            }

            results.append(
                result
            )

            print(
                f"  Runtime: "
                f"{average_runtime:.6f} s"
            )

            print(
                f"  Peak Memory: "
                f"{average_memory:.2f} KB"
            )

            print(
                f"  Reconstruction Error: "
                f"{average_error:.6e}"
            )

            print(
                f"  Relative Error: "
                f"{average_relative_error:.6e}"
            )

            print(
                f"  RMSE: "
                f"{average_rmse:.6e}"
            )

    return results


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

def save_results(results):
    """
    Save benchmark results to CSV.
    """

    fieldnames = [
        "algorithm",
        "matrix_size",
        "components",
        "runtime_seconds",
        "peak_memory_kb",
        "reconstruction_error",
        "relative_reconstruction_error",
        "rmse",
        "repeats",
    ]

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print()
    print(
        f"Results saved to: {CSV_PATH}"
    )


# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------

def plot_metric(
    results,
    metric,
    ylabel,
    filename,
    title
):
    """
    Plot a benchmark metric against matrix size.
    """

    algorithms = sorted(
        set(
            result["algorithm"]
            for result in results
        )
    )

    plt.figure(
        figsize=(9, 6)
    )

    for algorithm in algorithms:

        algorithm_results = [
            result
            for result in results
            if result["algorithm"] == algorithm
        ]

        sizes = [
            result["matrix_size"]
            for result in algorithm_results
        ]

        values = [
            result[metric]
            for result in algorithm_results
        ]

        plt.plot(
            sizes,
            values,
            marker="o",
            label=algorithm
        )

    plt.xlabel(
        "Matrix Size"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(
        RESULTS_DIR,
        filename
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    print(
        f"Plot saved to: {path}"
    )


def generate_plots(results):
    """
    Generate all benchmark plots.
    """

    plot_metric(
        results,
        "runtime_seconds",
        "Runtime (seconds)",
        "runtime_vs_size.png",
        "Decomposition Runtime vs Matrix Size"
    )

    plot_metric(
        results,
        "peak_memory_kb",
        "Peak Memory (KB)",
        "memory_vs_size.png",
        "Peak Memory Usage vs Matrix Size"
    )

    plot_metric(
        results,
        "relative_reconstruction_error",
        "Relative Reconstruction Error",
        "reconstruction_error.png",
        "Reconstruction Error Comparison"
    )

    plot_metric(
        results,
        "rmse",
        "RMSE",
        "rmse_comparison.png",
        "RMSE Comparison"
    )


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

def print_summary(results):
    """
    Print the best-performing algorithm for each metric.
    """

    print()
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    for size in MATRIX_SIZES:

        size_results = [
            result
            for result in results
            if result["matrix_size"] == size
        ]

        fastest = min(
            size_results,
            key=lambda x: x["runtime_seconds"]
        )

        lowest_memory = min(
            size_results,
            key=lambda x: x["peak_memory_kb"]
        )

        lowest_error = min(
            size_results,
            key=lambda x: x[
                "relative_reconstruction_error"
            ]
        )

        print(
            f"\nMatrix Size: {size} x {size}"
        )

        print(
            f"  Fastest: "
            f"{fastest['algorithm']} "
            f"({fastest['runtime_seconds']:.6f} s)"
        )

        print(
            f"  Lowest Memory: "
            f"{lowest_memory['algorithm']} "
            f"({lowest_memory['peak_memory_kb']:.2f} KB)"
        )

        print(
            f"  Lowest Relative Error: "
            f"{lowest_error['algorithm']} "
            f"({lowest_error['relative_reconstruction_error']:.6e})"
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    results = run_benchmarks()

    save_results(
        results
    )

    generate_plots(
        results
    )

    print_summary(
        results
    )

    print()
    print("=" * 70)
    print("BENCHMARKING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

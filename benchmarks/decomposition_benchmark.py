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
    EigenvalueDecomposition,
    IncrementalSVD,
    RobustPCA,
)

from matrix_factorization.metrics import (
    reconstruction_error,
    relative_reconstruction_error,
    rmse,
)


# =========================================================
# Configuration
# =========================================================

MATRIX_SIZES = [50, 100, 200]

N_COMPONENTS = 10
NMF_MAX_ITER = 200

INCREMENTAL_INITIAL_FRACTION = 0.5

ROBUST_PCA_MAX_ITER = 300

N_REPEATS = 3

RESULTS_DIR = os.path.join(
    "benchmarks",
    "results",
)

CSV_PATH = os.path.join(
    RESULTS_DIR,
    "benchmark_results.csv",
)


# =========================================================
# Matrix generation
# =========================================================

def generate_benchmark_matrix(size, seed=42):
    """
    Generate a positive matrix suitable for SVD, NMF,
    QR, LU, Eigenvalue Decomposition, and Robust PCA.
    """

    rng = np.random.default_rng(seed)

    return rng.random((size, size)) + 0.1


# =========================================================
# Timing and memory measurement
# =========================================================

def measure_fit_and_reconstruction(algorithm, A):
    """
    Measure fit + reconstruction runtime and peak memory.

    Returns
    -------
    tuple
        runtime_seconds, peak_memory_kb, reconstructed_matrix
    """

    tracemalloc.start()

    start_time = time.perf_counter()

    algorithm.fit(A)
    reconstructed = algorithm.reconstruct()

    end_time = time.perf_counter()

    _, peak_memory = tracemalloc.get_traced_memory()

    tracemalloc.stop()

    return (
        end_time - start_time,
        peak_memory / 1024.0,
        reconstructed,
    )


def calculate_metrics(A, reconstructed):
    """
    Calculate common reconstruction metrics.
    """

    return {
        "reconstruction_error": reconstruction_error(
            A,
            reconstructed,
        ),
        "relative_reconstruction_error": (
            relative_reconstruction_error(
                A,
                reconstructed,
            )
        ),
        "rmse": rmse(
            A,
            reconstructed,
        ),
    }


# =========================================================
# Standard decomposition benchmarks
# =========================================================

def benchmark_standard_algorithm(name, factory, A):
    """
    Benchmark a standard decomposition algorithm.

    Used for:
        SVD
        NMF
        QR
        LU
        Eigenvalue Decomposition
    """

    runtimes = []
    memories = []
    errors = []
    relative_errors = []
    rmses = []

    for _ in range(N_REPEATS):

        algorithm = factory()

        (
            runtime,
            memory,
            reconstructed,
        ) = measure_fit_and_reconstruction(
            algorithm,
            A,
        )

        metrics = calculate_metrics(
            A,
            reconstructed,
        )

        runtimes.append(runtime)
        memories.append(memory)
        errors.append(
            metrics["reconstruction_error"]
        )
        relative_errors.append(
            metrics["relative_reconstruction_error"]
        )
        rmses.append(
            metrics["rmse"]
        )

    return {
        "algorithm": name,
        "matrix_size": A.shape[0],
        "components": (
            min(N_COMPONENTS, A.shape[0])
            if name in ("SVD", "NMF")
            else A.shape[0]
        ),
        "runtime_seconds": float(
            np.mean(runtimes)
        ),
        "peak_memory_kb": float(
            np.mean(memories)
        ),
        "reconstruction_error": float(
            np.mean(errors)
        ),
        "relative_reconstruction_error": float(
            np.mean(relative_errors)
        ),
        "rmse": float(
            np.mean(rmses)
        ),
        "repeats": N_REPEATS,
        "initial_fit_seconds": "",
        "update_seconds": "",
        "iterations": "",
        "convergence_error": "",
        "rank": "",
        "sparsity": "",
    }


# =========================================================
# Incremental SVD benchmark
# =========================================================

def benchmark_incremental_svd(A):
    """
    Benchmark Incremental SVD using two stages:

    1. Initial fit on the first portion of the matrix.
    2. partial_fit() using the remaining rows.

    The final reconstruction is evaluated against the
    complete original matrix.
    """

    initial_rows = max(
        1,
        int(A.shape[0] * INCREMENTAL_INITIAL_FRACTION),
    )

    A_initial = A[:initial_rows]
    A_update = A[initial_rows:]

    initial_times = []
    update_times = []
    memories = []
    errors = []
    relative_errors = []
    rmses = []

    for _ in range(N_REPEATS):

        algorithm = IncrementalSVD(
            n_components=min(
                N_COMPONENTS,
                A.shape[0],
                A.shape[1],
            )
        )

        tracemalloc.start()

        start_initial = time.perf_counter()

        algorithm.fit(A_initial)

        end_initial = time.perf_counter()

        start_update = time.perf_counter()

        algorithm.partial_fit(A_update)

        reconstructed = algorithm.reconstruct()

        end_update = time.perf_counter()

        _, peak_memory = tracemalloc.get_traced_memory()

        tracemalloc.stop()

        metrics = calculate_metrics(
            A,
            reconstructed,
        )

        initial_times.append(
            end_initial - start_initial
        )

        update_times.append(
            end_update - start_update
        )

        memories.append(
            peak_memory / 1024.0
        )

        errors.append(
            metrics["reconstruction_error"]
        )

        relative_errors.append(
            metrics["relative_reconstruction_error"]
        )

        rmses.append(
            metrics["rmse"]
        )

    return {
        "algorithm": "IncrementalSVD",
        "matrix_size": A.shape[0],
        "components": min(
            N_COMPONENTS,
            A.shape[0],
            A.shape[1],
        ),
        "runtime_seconds": float(
            np.mean(initial_times)
            + np.mean(update_times)
        ),
        "peak_memory_kb": float(
            np.mean(memories)
        ),
        "reconstruction_error": float(
            np.mean(errors)
        ),
        "relative_reconstruction_error": float(
            np.mean(relative_errors)
        ),
        "rmse": float(
            np.mean(rmses)
        ),
        "repeats": N_REPEATS,
        "initial_fit_seconds": float(
            np.mean(initial_times)
        ),
        "update_seconds": float(
            np.mean(update_times)
        ),
        "iterations": "",
        "convergence_error": "",
        "rank": "",
        "sparsity": "",
    }


# =========================================================
# Robust PCA benchmark
# =========================================================

def benchmark_robust_pca(A):
    """
    Benchmark Robust PCA.

    Records:
        - runtime
        - memory
        - reconstruction metrics
        - ADMM iteration count
        - final convergence error
        - estimated low-rank rank
        - sparse component density
    """

    runtimes = []
    memories = []
    errors = []
    relative_errors = []
    rmses = []
    iterations = []
    convergence_errors = []
    ranks = []
    sparsities = []

    for _ in range(N_REPEATS):

        algorithm = RobustPCA(
            max_iter=ROBUST_PCA_MAX_ITER,
            tol=1e-7,
        )

        (
            runtime,
            memory,
            reconstructed,
        ) = measure_fit_and_reconstruction(
            algorithm,
            A,
        )

        metrics = calculate_metrics(
            A,
            reconstructed,
        )

        runtimes.append(runtime)
        memories.append(memory)
        errors.append(
            metrics["reconstruction_error"]
        )
        relative_errors.append(
            metrics["relative_reconstruction_error"]
        )
        rmses.append(
            metrics["rmse"]
        )

        iterations.append(
            algorithm.n_iter_
        )

        if algorithm.convergence_history_:
            convergence_errors.append(
                algorithm.convergence_history_[-1]
            )
        else:
            convergence_errors.append(
                np.nan
            )

        ranks.append(
            algorithm.rank()
        )

        sparsities.append(
            algorithm.sparsity()
        )

    return {
        "algorithm": "RobustPCA",
        "matrix_size": A.shape[0],
        "components": "",
        "runtime_seconds": float(
            np.mean(runtimes)
        ),
        "peak_memory_kb": float(
            np.mean(memories)
        ),
        "reconstruction_error": float(
            np.mean(errors)
        ),
        "relative_reconstruction_error": float(
            np.mean(relative_errors)
        ),
        "rmse": float(
            np.mean(rmses)
        ),
        "repeats": N_REPEATS,
        "initial_fit_seconds": "",
        "update_seconds": "",
        "iterations": float(
            np.mean(iterations)
        ),
        "convergence_error": float(
            np.mean(convergence_errors)
        ),
        "rank": float(
            np.mean(ranks)
        ),
        "sparsity": float(
            np.mean(sparsities)
        ),
    }


# =========================================================
# Benchmark execution
# =========================================================

def run_benchmarks():
    """
    Run all decomposition benchmarks.
    """

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
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

    standard_algorithms = {
        "SVD": lambda size: SVD(
            n_components=min(
                N_COMPONENTS,
                size,
            )
        ),
        "NMF": lambda size: NMF(
            n_components=min(
                N_COMPONENTS,
                size,
            ),
            max_iter=NMF_MAX_ITER,
            random_state=42,
        ),
        "QR": lambda size: QR(),
        "LU": lambda size: LU(),
        "Eigen": lambda size: EigenvalueDecomposition(),
    }

    for size in MATRIX_SIZES:

        print("-" * 70)
        print(
            f"Matrix Size: {size} x {size}"
        )
        print("-" * 70)

        A = generate_benchmark_matrix(
            size
        )

        for name, factory in standard_algorithms.items():

            print(
                f"\nRunning {name}..."
            )

            result = benchmark_standard_algorithm(
                name,
                lambda: factory(size),
                A,
            )

            results.append(result)

            print(
                f"  Runtime: "
                f"{result['runtime_seconds']:.6f} s"
            )

            print(
                f"  Peak Memory: "
                f"{result['peak_memory_kb']:.2f} KB"
            )

            print(
                f"  Relative Error: "
                f"{result['relative_reconstruction_error']:.6e}"
            )

            print(
                f"  RMSE: "
                f"{result['rmse']:.6e}"
            )

        print("\nRunning IncrementalSVD...")

        incremental_result = benchmark_incremental_svd(
            A
        )

        results.append(
            incremental_result
        )

        print(
            f"  Initial Fit: "
            f"{incremental_result['initial_fit_seconds']:.6f} s"
        )

        print(
            f"  Incremental Update: "
            f"{incremental_result['update_seconds']:.6f} s"
        )

        print(
            f"  Total Runtime: "
            f"{incremental_result['runtime_seconds']:.6f} s"
        )

        print(
            f"  Relative Error: "
            f"{incremental_result['relative_reconstruction_error']:.6e}"
        )

        print("\nRunning RobustPCA...")

        robust_result = benchmark_robust_pca(
            A
        )

        results.append(
            robust_result
        )

        print(
            f"  Runtime: "
            f"{robust_result['runtime_seconds']:.6f} s"
        )

        print(
            f"  Iterations: "
            f"{robust_result['iterations']:.1f}"
        )

        print(
            f"  Final Convergence Error: "
            f"{robust_result['convergence_error']:.6e}"
        )

        print(
            f"  Estimated Rank: "
            f"{robust_result['rank']:.1f}"
        )

        print(
            f"  Sparsity: "
            f"{robust_result['sparsity']:.6f}"
        )

    return results


# =========================================================
# Save CSV
# =========================================================

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
        "initial_fit_seconds",
        "update_seconds",
        "iterations",
        "convergence_error",
        "rank",
        "sparsity",
    ]

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print(
        f"Results saved to: {CSV_PATH}"
    )


# =========================================================
# Plotting
# =========================================================

def plot_metric(
    results,
    metric,
    ylabel,
    filename,
    title,
):
    """
    Plot a benchmark metric against matrix size.

    Only algorithms with a numeric value for the
    requested metric are plotted.
    """

    algorithms = sorted(
        set(
            result["algorithm"]
            for result in results
            if isinstance(
                result.get(metric),
                (int, float),
            )
            and np.isfinite(
                result[metric]
            )
        )
    )

    plt.figure(
        figsize=(9, 6)
    )

    for algorithm in algorithms:

        algorithm_results = [
            result
            for result in results
            if (
                result["algorithm"] == algorithm
                and isinstance(
                    result.get(metric),
                    (int, float),
                )
                and np.isfinite(
                    result[metric]
                )
            )
        ]

        algorithm_results.sort(
            key=lambda x: x["matrix_size"]
        )

        sizes = [
            result["matrix_size"]
            for result in algorithm_results
        ]

        values = [
            result[metric]
            for result in algorithm_results
        ]

        if sizes:
            plt.plot(
                sizes,
                values,
                marker="o",
                label=algorithm,
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
        filename,
    )

    plt.savefig(
        path,
        dpi=300,
    )

    plt.close()

    print(
        f"Plot saved to: {path}"
    )


def generate_plots(results):
    """
    Generate benchmark plots.
    """

    plot_metric(
        results,
        "runtime_seconds",
        "Runtime (seconds)",
        "runtime_vs_size.png",
        "Decomposition Runtime vs Matrix Size",
    )

    plot_metric(
        results,
        "peak_memory_kb",
        "Peak Memory (KB)",
        "memory_vs_size.png",
        "Peak Memory Usage vs Matrix Size",
    )

    plot_metric(
        results,
        "relative_reconstruction_error",
        "Relative Reconstruction Error",
        "reconstruction_error.png",
        "Reconstruction Error vs Matrix Size",
    )

    plot_metric(
        results,
        "rmse",
        "RMSE",
        "rmse_comparison.png",
        "RMSE vs Matrix Size",
    )

    plot_metric(
        results,
        "iterations",
        "Iterations",
        "convergence_iterations.png",
        "Robust PCA Iterations vs Matrix Size",
    )

    plot_metric(
        results,
        "convergence_error",
        "Final Convergence Error",
        "convergence_error.png",
        "Robust PCA Convergence Error",
    )


# =========================================================
# Summary
# =========================================================

def print_summary(results):
    """
    Print measured benchmark values grouped by matrix size.

    This intentionally reports measurements rather than
    declaring an overall winner.
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

        print(
            f"\nMatrix Size: {size} x {size}"
        )

        for result in size_results:

            print(
                f"  {result['algorithm']:<15}"
                f" runtime="
                f"{result['runtime_seconds']:.6f}s"
                f" | memory="
                f"{result['peak_memory_kb']:.2f}KB"
                f" | rel_error="
                f"{result['relative_reconstruction_error']:.3e}"
            )

            if result["algorithm"] == "IncrementalSVD":

                print(
                    f"    initial_fit="
                    f"{result['initial_fit_seconds']:.6f}s"
                    f" | update="
                    f"{result['update_seconds']:.6f}s"
                )

            if result["algorithm"] == "RobustPCA":

                print(
                    f"    iterations="
                    f"{result['iterations']:.1f}"
                    f" | convergence="
                    f"{result['convergence_error']:.3e}"
                    f" | rank="
                    f"{result['rank']:.1f}"
                    f" | sparsity="
                    f"{result['sparsity']:.6f}"
                )


# =========================================================
# Main
# =========================================================

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

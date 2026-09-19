"""
Scalability Experiment.

Measures runtime, peak memory, and reconstruction quality
as matrix dimensions increase.

This experiment is intentionally separate from the regular
decomposition benchmark so scalability results can be analyzed
independently.
"""

import csv
import os
import time
import tracemalloc

import numpy as np

from matrix_factorization.decompositions import (
    SVD,
    NMF,
    QR,
    LU,
    EigenvalueDecomposition,
)

from matrix_factorization.metrics import (
    reconstruction_error,
    relative_reconstruction_error,
    rmse,
)


# =========================================================
# Configuration
# =========================================================

MATRIX_SIZES = [
    50,
    100,
    200,
    300,
    400,
]

N_COMPONENTS = 10
NMF_MAX_ITER = 200

N_REPEATS = 3

RANDOM_SEED = 42

RESULTS_DIR = os.path.join(
    "benchmarks",
    "scalability",
    "results",
)

CSV_PATH = os.path.join(
    RESULTS_DIR,
    "scalability_results.csv",
)


# =========================================================
# Matrix generation
# =========================================================

def generate_matrix(size):
    """
    Generate a reproducible symmetric positive matrix.

    Symmetry ensures that Eigenvalue Decomposition produces
    real-valued eigenvalues and eigenvectors.
    """

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    random_matrix = rng.random(
        (size, size)
    )

    matrix = (
        random_matrix
        + random_matrix.T
    ) / 2.0

    matrix += 0.1 * np.eye(size)

    return matrix


# =========================================================
# Algorithm factory
# =========================================================

def create_algorithms(size):
    """
    Create decomposition objects for a matrix size.
    """

    components = min(
        N_COMPONENTS,
        size,
    )

    return {
        "SVD": SVD(
            n_components=components,
        ),

        "NMF": NMF(
            n_components=components,
            max_iter=NMF_MAX_ITER,
            random_state=RANDOM_SEED,
        ),

        "QR": QR(),

        "LU": LU(),

        "Eigen": EigenvalueDecomposition(),
    }


# =========================================================
# Measurement
# =========================================================

def measure_algorithm(algorithm, matrix):
    """
    Measure runtime and peak memory.

    Returns
    -------
    dict
        Runtime, memory, reconstruction and error metrics.
    """

    tracemalloc.start()

    start_time = time.perf_counter()

    algorithm.fit(matrix)

    if isinstance(algorithm, LU):
        reconstructed = (
            algorithm.P.T
            @ algorithm.L
            @ algorithm.U
        )
    else:
        reconstructed = algorithm.reconstruct()

    end_time = time.perf_counter()

    _, peak_memory = (
        tracemalloc.get_traced_memory()
    )

    tracemalloc.stop()

    runtime = (
        end_time - start_time
    )

    memory_kb = (
        peak_memory / 1024.0
    )

    return {
        "runtime_seconds": runtime,
        "peak_memory_kb": memory_kb,
        "reconstruction_error": (
            reconstruction_error(
                matrix,
                reconstructed,
            )
        ),
        "relative_reconstruction_error": (
            relative_reconstruction_error(
                matrix,
                reconstructed,
            )
        ),
        "rmse": rmse(
            matrix,
            reconstructed,
        ),
    }


# =========================================================
# Single configuration
# =========================================================

def benchmark_configuration(
    algorithm_name,
    matrix,
    size,
):
    """
    Benchmark one algorithm on one matrix size.
    """

    runtimes = []
    memories = []
    errors = []
    relative_errors = []
    rmses = []

    for repeat in range(N_REPEATS):

        algorithm = create_algorithms(
            size
        )[algorithm_name]

        metrics = measure_algorithm(
            algorithm,
            matrix,
        )

        runtimes.append(
            metrics["runtime_seconds"]
        )

        memories.append(
            metrics["peak_memory_kb"]
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

        print(
            f"    Repeat {repeat + 1}/{N_REPEATS}: "
            f"{metrics['runtime_seconds']:.4f}s"
        )

    return {
        "algorithm": algorithm_name,
        "matrix_size": size,
        "matrix_elements": size * size,
        "components": (
            min(N_COMPONENTS, size)
            if algorithm_name in ("SVD", "NMF")
            else size
        ),
        "runtime_seconds": float(
            np.mean(runtimes)
        ),
        "runtime_std_seconds": float(
            np.std(runtimes)
        ),
        "peak_memory_kb": float(
            np.mean(memories)
        ),
        "peak_memory_std_kb": float(
            np.std(memories)
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
    }


# =========================================================
# Experiment
# =========================================================

def run_experiment():
    """
    Run scalability experiments for all matrix sizes.
    """

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    results = []

    print("=" * 72)
    print(
        "MATRIX FACTORIZATION ENGINE - SCALABILITY EXPERIMENT"
    )
    print("=" * 72)

    print(
        f"Matrix sizes: {MATRIX_SIZES}"
    )

    print(
        f"Repeats: {N_REPEATS}"
    )

    print()

    for size in MATRIX_SIZES:

        print("-" * 72)
        print(
            f"Matrix Size: {size} x {size}"
        )
        print("-" * 72)

        matrix = generate_matrix(
            size
        )

        algorithms = create_algorithms(
            size
        )

        for algorithm_name in algorithms:

            print(
                f"\nRunning {algorithm_name}..."
            )

            result = benchmark_configuration(
                algorithm_name,
                matrix,
                size,
            )

            results.append(
                result
            )

            print(
                f"  Mean Runtime: "
                f"{result['runtime_seconds']:.6f}s"
            )

            print(
                f"  Runtime Std: "
                f"{result['runtime_std_seconds']:.6f}s"
            )

            print(
                f"  Mean Peak Memory: "
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

    return results


# =========================================================
# CSV
# =========================================================

def save_results(results):
    """
    Save scalability measurements to CSV.
    """

    fieldnames = [
        "algorithm",
        "matrix_size",
        "matrix_elements",
        "components",
        "runtime_seconds",
        "runtime_std_seconds",
        "peak_memory_kb",
        "peak_memory_std_kb",
        "reconstruction_error",
        "relative_reconstruction_error",
        "rmse",
        "repeats",
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

        writer.writerows(
            results
        )

    print()
    print(
        f"Results saved to: {CSV_PATH}"
    )


# =========================================================
# Main
# =========================================================

def main():

    results = run_experiment()

    save_results(
        results
    )

    print()
    print("=" * 72)
    print(
        "SCALABILITY EXPERIMENT COMPLETE"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()

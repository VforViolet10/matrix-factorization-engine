"""
Scalability Benchmark Analysis

Analyzes scalability_results.csv and generates:
1. Summary statistics
2. Runtime plots
3. Memory plots
4. Reconstruction-error plots
5. Scaling-factor analysis
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR.parent / "plots"

INPUT_FILE = RESULTS_DIR / "scalability_results.csv"
SUMMARY_FILE = RESULTS_DIR / "scalability_summary.csv"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Benchmark results not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "algorithm",
        "matrix_size",
        "runtime_seconds",
        "peak_memory_kb",
        "reconstruction_error",
        "relative_reconstruction_error",
        "rmse",
    ]

    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


# ---------------------------------------------------------------------
# Generate summary
# ---------------------------------------------------------------------

def generate_summary(df):
    summary = (
        df.groupby("algorithm")
        .agg(
            avg_runtime_seconds=("runtime_seconds", "mean"),
            max_runtime_seconds=("runtime_seconds", "max"),
            avg_memory_kb=("peak_memory_kb", "mean"),
            max_memory_kb=("peak_memory_kb", "max"),
            avg_relative_error=("relative_reconstruction_error", "mean"),
            max_relative_error=("relative_reconstruction_error", "max"),
            avg_rmse=("rmse", "mean"),
            max_rmse=("rmse", "max"),
        )
        .reset_index()
    )

    summary.to_csv(SUMMARY_FILE, index=False)

    return summary


# ---------------------------------------------------------------------
# Runtime plot
# ---------------------------------------------------------------------

def plot_runtime(df):
    plt.figure(figsize=(10, 6))

    for algorithm in df["algorithm"].unique():
        data = df[df["algorithm"] == algorithm]

        plt.plot(
            data["matrix_size"],
            data["runtime_seconds"],
            marker="o",
            label=algorithm,
        )

    plt.xlabel("Matrix Size (n × n)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime vs Matrix Size")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output = PLOTS_DIR / "runtime_vs_matrix_size.png"
    plt.savefig(output, dpi=300)
    plt.close()

    print(f"Created: {output}")


# ---------------------------------------------------------------------
# Memory plot
# ---------------------------------------------------------------------

def plot_memory(df):
    plt.figure(figsize=(10, 6))

    for algorithm in df["algorithm"].unique():
        data = df[df["algorithm"] == algorithm]

        plt.plot(
            data["matrix_size"],
            data["peak_memory_kb"],
            marker="o",
            label=algorithm,
        )

    plt.xlabel("Matrix Size (n × n)")
    plt.ylabel("Peak Memory (KB)")
    plt.title("Peak Memory vs Matrix Size")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output = PLOTS_DIR / "memory_vs_matrix_size.png"
    plt.savefig(output, dpi=300)
    plt.close()

    print(f"Created: {output}")


# ---------------------------------------------------------------------
# Reconstruction error plot
# ---------------------------------------------------------------------

def plot_error(df):
    plt.figure(figsize=(10, 6))

    for algorithm in df["algorithm"].unique():
        data = df[df["algorithm"] == algorithm]

        plt.plot(
            data["matrix_size"],
            data["relative_reconstruction_error"],
            marker="o",
            label=algorithm,
        )

    plt.xlabel("Matrix Size (n × n)")
    plt.ylabel("Relative Reconstruction Error")
    plt.title("Reconstruction Error vs Matrix Size")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output = PLOTS_DIR / "error_vs_matrix_size.png"
    plt.savefig(output, dpi=300)
    plt.close()

    print(f"Created: {output}")


# ---------------------------------------------------------------------
# Scaling analysis
# ---------------------------------------------------------------------

def calculate_scaling(df):
    rows = []

    for algorithm in df["algorithm"].unique():
        data = (
            df[df["algorithm"] == algorithm]
            .sort_values("matrix_size")
        )

        first = data.iloc[0]
        last = data.iloc[-1]

        size_ratio = last["matrix_size"] / first["matrix_size"]

        runtime_ratio = (
            last["runtime_seconds"] /
            first["runtime_seconds"]
        )

        memory_ratio = (
            last["peak_memory_kb"] /
            first["peak_memory_kb"]
        )

        rows.append(
            {
                "algorithm": algorithm,
                "initial_matrix_size": first["matrix_size"],
                "final_matrix_size": last["matrix_size"],
                "matrix_size_ratio": size_ratio,
                "runtime_ratio": runtime_ratio,
                "memory_ratio": memory_ratio,
            }
        )

    scaling = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("SCALING ANALYSIS")
    print("=" * 70)

    print(scaling.to_string(index=False))

    return scaling


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("=" * 70)
    print("MATRIX FACTORIZATION ENGINE - SCALABILITY ANALYSIS")
    print("=" * 70)

    df = load_data()

    print(f"\nLoaded {len(df)} benchmark records.")

    summary = generate_summary(df)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(summary.to_string(index=False))

    print("\nGenerating plots...")

    plot_runtime(df)
    plot_memory(df)
    plot_error(df)

    calculate_scaling(df)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(f"\nSummary saved to:")
    print(SUMMARY_FILE)

    print("\nPlots saved to:")
    print(PLOTS_DIR)


if __name__ == "__main__":
    main()

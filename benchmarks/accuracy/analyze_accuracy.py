"""
Accuracy Benchmark Analysis

Generates summary tables and plots for reconstruction
quality, runtime, and memory as the number of
components increases.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent

RESULTS_FILE = (
    BASE_DIR
    / "results"
    / "accuracy_results.csv"
)

PLOTS_DIR = (
    BASE_DIR.parent
    / "plots"
)

SUMMARY_FILE = (
    BASE_DIR
    / "results"
    / "accuracy_summary.csv"
)

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def main():

    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Results file not found: {RESULTS_FILE}"
        )

    df = pd.read_csv(
        RESULTS_FILE
    )

    required_columns = {
        "algorithm",
        "matrix_size",
        "components",
        "runtime_seconds",
        "peak_memory_kb",
        "reconstruction_error",
        "relative_reconstruction_error",
        "rmse",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    summary = (
        df.groupby(
            [
                "algorithm",
                "matrix_size",
                "components",
            ],
            as_index=False,
        )
        .agg(
            {
                "runtime_seconds": "mean",
                "peak_memory_kb": "mean",
                "reconstruction_error": "mean",
                "relative_reconstruction_error": "mean",
                "rmse": "mean",
            }
        )
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    # --------------------------------------------------
    # Reconstruction Error vs Components
    # --------------------------------------------------

    plt.figure()

    for algorithm in sorted(
        df["algorithm"].unique()
    ):

        for size in sorted(
            df["matrix_size"].unique()
        ):

            subset = df[
                (df["algorithm"] == algorithm)
                & (df["matrix_size"] == size)
            ]

            plt.plot(
                subset["components"],
                subset[
                    "relative_reconstruction_error"
                ],
                marker="o",
                label=f"{algorithm} - {size}x{size}",
            )

    plt.xlabel(
        "Number of Components"
    )

    plt.ylabel(
        "Relative Reconstruction Error"
    )

    plt.title(
        "Reconstruction Error vs Components"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "accuracy_error_vs_components.png"
    )

    plt.close()

    # --------------------------------------------------
    # Runtime vs Components
    # --------------------------------------------------

    plt.figure()

    for algorithm in sorted(
        df["algorithm"].unique()
    ):

        for size in sorted(
            df["matrix_size"].unique()
        ):

            subset = df[
                (df["algorithm"] == algorithm)
                & (df["matrix_size"] == size)
            ]

            plt.plot(
                subset["components"],
                subset["runtime_seconds"],
                marker="o",
                label=f"{algorithm} - {size}x{size}",
            )

    plt.xlabel(
        "Number of Components"
    )

    plt.ylabel(
        "Runtime (seconds)"
    )

    plt.title(
        "Runtime vs Components"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "accuracy_runtime_vs_components.png"
    )

    plt.close()

    # --------------------------------------------------
    # Memory vs Components
    # --------------------------------------------------

    plt.figure()

    for algorithm in sorted(
        df["algorithm"].unique()
    ):

        for size in sorted(
            df["matrix_size"].unique()
        ):

            subset = df[
                (df["algorithm"] == algorithm)
                & (df["matrix_size"] == size)
            ]

            plt.plot(
                subset["components"],
                subset["peak_memory_kb"],
                marker="o",
                label=f"{algorithm} - {size}x{size}",
            )

    plt.xlabel(
        "Number of Components"
    )

    plt.ylabel(
        "Peak Memory (KB)"
    )

    plt.title(
        "Memory Usage vs Components"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "accuracy_memory_vs_components.png"
    )

    plt.close()

    print("=" * 70)
    print(
        "ACCURACY ANALYSIS COMPLETE"
    )
    print("=" * 70)

    print(
        f"\nSummary saved to:\n"
        f"{SUMMARY_FILE}"
    )

    print(
        "\nPlots generated:"
    )

    print(
        PLOTS_DIR
        / "accuracy_error_vs_components.png"
    )

    print(
        PLOTS_DIR
        / "accuracy_runtime_vs_components.png"
    )

    print(
        PLOTS_DIR
        / "accuracy_memory_vs_components.png"
    )


if __name__ == "__main__":
    main()

"""
Scaling Report

Analyzes how runtime and memory scale with matrix size.
Uses log-log regression to estimate empirical scaling exponents.
"""

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

INPUT_FILE = RESULTS_DIR / "scalability_results.csv"
OUTPUT_FILE = RESULTS_DIR / "scaling_report.csv"


def estimate_exponent(x, y):
    """
    Estimate empirical exponent y ~ x^p
    using linear regression in log-log space.
    """
    log_x = np.log(x)
    log_y = np.log(y)

    slope, _ = np.polyfit(log_x, log_y, 1)

    return slope


def main():
    print("=" * 70)
    print("MATRIX FACTORIZATION ENGINE - SCALING REPORT")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find benchmark file: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    rows = []

    for algorithm in sorted(df["algorithm"].unique()):
        data = (
            df[df["algorithm"] == algorithm]
            .sort_values("matrix_size")
        )

        runtime_exponent = estimate_exponent(
            data["matrix_size"].values,
            data["runtime_seconds"].values,
        )

        memory_exponent = estimate_exponent(
            data["matrix_size"].values,
            data["peak_memory_kb"].values,
        )

        first_runtime = data.iloc[0]["runtime_seconds"]
        last_runtime = data.iloc[-1]["runtime_seconds"]

        first_memory = data.iloc[0]["peak_memory_kb"]
        last_memory = data.iloc[-1]["peak_memory_kb"]

        rows.append(
            {
                "algorithm": algorithm,
                "runtime_scaling_exponent": runtime_exponent,
                "memory_scaling_exponent": memory_exponent,
                "runtime_growth_factor": last_runtime / first_runtime,
                "memory_growth_factor": last_memory / first_memory,
            }
        )

    report = pd.DataFrame(rows)

    report.to_csv(OUTPUT_FILE, index=False)

    print("\nScaling results:")
    print()

    print(
        report.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    for _, row in report.iterrows():
        print(
            f"\n{row['algorithm']}:"
            f"\n  Runtime exponent : "
            f"{row['runtime_scaling_exponent']:.2f}"
            f"\n  Memory exponent  : "
            f"{row['memory_scaling_exponent']:.2f}"
            f"\n  Runtime growth   : "
            f"{row['runtime_growth_factor']:.2f}x"
            f"\n  Memory growth    : "
            f"{row['memory_growth_factor']:.2f}x"
        )

    print("\n" + "=" * 70)
    print("REPORT COMPLETE")
    print("=" * 70)

    print(f"\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()

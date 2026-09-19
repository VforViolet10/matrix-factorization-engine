"""
Unified Benchmark Runner

Runs the available benchmark analysis modules for the
Matrix Factorization Engine.
"""

import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent


def run_command(command):
    """Run a command and stop if it fails."""

    print("\n" + "=" * 70)
    print(f"RUNNING: {' '.join(command)}")
    print("=" * 70)

    result = subprocess.run(
        command,
        cwd=ROOT_DIR,
    )

    if result.returncode != 0:
        print(
            f"\nERROR: Command failed with exit code "
            f"{result.returncode}"
        )
        sys.exit(result.returncode)


def main():
    print("=" * 70)
    print("MATRIX FACTORIZATION ENGINE - BENCHMARK SUITE")
    print("=" * 70)

    python = sys.executable

    commands = [
        [
            python,
            "benchmarks/scalability/analyze_scalability.py",
        ],
        [
            python,
            "benchmarks/scalability/scaling_report.py",
        ],
        [
            python,
            "benchmarks/accuracy/analyze_accuracy.py",
        ],
    ]

    for command in commands:
        run_command(command)

    print("\n" + "=" * 70)
    print("ALL BENCHMARK ANALYSES COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()

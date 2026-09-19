import pandas as pd
from pathlib import Path


RESULTS_FILE = Path(
    "benchmarks/scalability/results/scalability_results.csv"
)


def test_scalability_results_exist():
    assert RESULTS_FILE.exists()


def test_scalability_results_not_empty():
    df = pd.read_csv(RESULTS_FILE)

    assert not df.empty


def test_expected_algorithms_present():
    df = pd.read_csv(RESULTS_FILE)

    expected = {
        "SVD",
        "NMF",
        "QR",
        "LU",
        "Eigen",
    }

    assert expected.issubset(set(df["algorithm"]))


def test_expected_matrix_sizes_present():
    df = pd.read_csv(RESULTS_FILE)

    expected_sizes = {
        50,
        100,
        200,
        300,
        400,
    }

    assert expected_sizes.issubset(
        set(df["matrix_size"])
    )


def test_runtime_values_positive():
    df = pd.read_csv(RESULTS_FILE)

    assert (df["runtime_seconds"] > 0).all()


def test_memory_values_positive():
    df = pd.read_csv(RESULTS_FILE)

    assert (df["peak_memory_kb"] > 0).all()


def test_reconstruction_error_non_negative():
    df = pd.read_csv(RESULTS_FILE)

    assert (df["reconstruction_error"] >= 0).all()


def test_relative_error_non_negative():
    df = pd.read_csv(RESULTS_FILE)

    assert (df["relative_reconstruction_error"] >= 0).all()


def test_rmse_non_negative():
    df = pd.read_csv(RESULTS_FILE)

    assert (df["rmse"] >= 0).all()


def test_three_repeats_per_configuration():
    df = pd.read_csv(RESULTS_FILE)

    counts = (
        df.groupby(
            ["algorithm", "matrix_size"]
        )["repeats"]
        .first()
    )

    assert (counts == 3).all()

from pathlib import Path

import pandas as pd


RESULTS_FILE = Path(
    "benchmarks/accuracy/results/accuracy_results.csv"
)


def test_accuracy_results_exist():
    assert RESULTS_FILE.exists()


def test_accuracy_results_not_empty():
    df = pd.read_csv(RESULTS_FILE)

    assert not df.empty


def test_expected_algorithms_present():
    df = pd.read_csv(RESULTS_FILE)

    expected = {
        "SVD",
        "NMF",
    }

    assert expected.issubset(
        set(df["algorithm"])
    )


def test_expected_matrix_sizes_present():
    df = pd.read_csv(RESULTS_FILE)

    expected = {
        50,
        100,
        200,
    }

    assert expected.issubset(
        set(df["matrix_size"])
    )


def test_expected_components_present():
    df = pd.read_csv(RESULTS_FILE)

    expected = {
        5,
        10,
        20,
        40,
    }

    assert expected.issubset(
        set(df["components"])
    )


def test_runtime_positive():
    df = pd.read_csv(RESULTS_FILE)

    assert (
        df["runtime_seconds"] > 0
    ).all()


def test_memory_positive():
    df = pd.read_csv(RESULTS_FILE)

    assert (
        df["peak_memory_kb"] > 0
    ).all()


def test_errors_non_negative():
    df = pd.read_csv(RESULTS_FILE)

    assert (
        df["reconstruction_error"] >= 0
    ).all()

    assert (
        df["relative_reconstruction_error"] >= 0
    ).all()

    assert (
        df["rmse"] >= 0
    ).all()


def test_three_repeats():
    df = pd.read_csv(RESULTS_FILE)

    assert (
        df["repeats"] == 3
    ).all()


def test_expected_configuration_count():
    df = pd.read_csv(RESULTS_FILE)

    assert len(df) == 24

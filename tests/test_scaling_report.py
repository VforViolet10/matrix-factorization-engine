from pathlib import Path

import pandas as pd


REPORT_FILE = Path(
    "benchmarks/scalability/results/scaling_report.csv"
)


def test_scaling_report_exists():
    assert REPORT_FILE.exists()


def test_scaling_report_not_empty():
    df = pd.read_csv(REPORT_FILE)

    assert not df.empty


def test_scaling_report_algorithms():
    df = pd.read_csv(REPORT_FILE)

    expected = {
        "SVD",
        "NMF",
        "QR",
        "LU",
        "Eigen",
    }

    assert expected.issubset(set(df["algorithm"]))


def test_runtime_exponents_positive():
    df = pd.read_csv(REPORT_FILE)

    assert (
        df["runtime_scaling_exponent"] > 0
    ).all()


def test_memory_exponents_positive():
    df = pd.read_csv(REPORT_FILE)

    assert (
        df["memory_scaling_exponent"] > 0
    ).all()


def test_runtime_growth_positive():
    df = pd.read_csv(REPORT_FILE)

    assert (
        df["runtime_growth_factor"] > 1
    ).all()


def test_memory_growth_positive():
    df = pd.read_csv(REPORT_FILE)

    assert (
        df["memory_growth_factor"] > 1
    ).all()

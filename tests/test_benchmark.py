import os

import numpy as np

from benchmarks.decomposition_benchmark import (
    generate_benchmark_matrix,
    create_algorithms,
)


def test_generate_benchmark_matrix():

    A = generate_benchmark_matrix(
        10,
        seed=42
    )

    assert A.shape == (10, 10)

    assert np.all(
        A > 0
    )


def test_benchmark_matrix_reproducibility():

    A1 = generate_benchmark_matrix(
        10,
        seed=42
    )

    A2 = generate_benchmark_matrix(
        10,
        seed=42
    )

    assert np.allclose(
        A1,
        A2
    )


def test_create_algorithms():

    algorithms = create_algorithms(
        20
    )

    expected = {
        "SVD",
        "NMF",
        "QR",
        "LU",
    }

    assert set(
        algorithms.keys()
    ) == expected


def test_benchmark_output_directory():

    directory = os.path.join(
        "benchmarks",
        "results"
    )

    assert os.path.isdir(
        directory
    )

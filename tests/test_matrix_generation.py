import numpy as np

from src.matrix_factorization.utils.matrix_generation import (
    generate_random_matrix,
    generate_low_rank_matrix,
)


def test_random_matrix_shape():
    A = generate_random_matrix(10, 5, seed=42)

    assert A.shape == (10, 5)


def test_random_matrix_reproducibility():
    A = generate_random_matrix(10, 5, seed=42)
    B = generate_random_matrix(10, 5, seed=42)

    assert np.array_equal(A, B)


def test_low_rank_matrix_shape():
    A = generate_low_rank_matrix(20, 10, 3, seed=42)

    assert A.shape == (20, 10)


def test_invalid_rank():
    try:
        generate_low_rank_matrix(10, 5, 6)
        assert False
    except ValueError:
        assert True

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.matrix_factorization.utils.matrix_generation import (
    generate_random_matrix,
    generate_low_rank_matrix,
)

from src.matrix_factorization.utils.visualization import plot_matrix


# Generate random matrix
A = generate_random_matrix(20, 10, seed=42)

print("Random Matrix:")
print(A)

print("\nShape:", A.shape)


# Generate low-rank matrix
B = generate_low_rank_matrix(20, 10, rank=3, seed=42)

print("\nLow-Rank Matrix Shape:", B.shape)

plot_matrix(B, "Low-Rank Matrix")

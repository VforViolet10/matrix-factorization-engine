# Matrix Factorization Engine

A research-oriented Python engine for implementing, comparing,
and benchmarking matrix and tensor factorization techniques.

The project combines classical linear algebra decompositions,
advanced matrix factorization methods, tensor decomposition,
real-world applications, and quantitative performance analysis.

---

## Project Overview

The Matrix Factorization Engine provides a unified framework for
experimenting with different factorization techniques and studying
their:

- Reconstruction accuracy
- Approximation quality
- Computational runtime
- Memory consumption
- Scalability
- Convergence behavior
- Low-rank representations

The project is designed for both educational experimentation and
research-oriented benchmarking.

---

## Algorithms

### Classical Matrix Decompositions

| Algorithm | Description |
|---|---|
| SVD | Singular Value Decomposition |
| QR | QR Factorization |
| Eigen | Eigenvalue Decomposition |
| LU | LU Decomposition |

### Advanced Factorization Methods

| Algorithm | Description |
|---|---|
| NMF | Non-negative Matrix Factorization |
| Incremental SVD | Incremental / streaming low-rank factorization |
| Robust PCA | Low-rank + sparse decomposition |
| Tensor Factorization | CP/PARAFAC decomposition using ALS |

---

## Unified API

The project provides a common interface for supported
matrix factorization methods.

```python
import numpy as np

from matrix_factorization import MatrixFactorizationEngine

X = np.array([
    [5.0, 3.0, 1.0],
    [4.0, 2.0, 5.0],
    [1.0, 5.0, 4.0],
])

engine = MatrixFactorizationEngine(
    method="svd",
    n_components=2,
)

engine.fit(X)

reconstructed = engine.reconstruct()

print(reconstructed)

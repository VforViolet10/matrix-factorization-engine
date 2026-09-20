import time

import numpy as np
from fastapi import APIRouter, HTTPException

from matrix_factorization import MatrixFactorizationEngine

from api.schemas import (
    FactorizationRequest,
    FactorizationResponse,
)


router = APIRouter()


@router.get("/methods")
def methods():
    return {
        "methods": MatrixFactorizationEngine.available_methods()
    }


@router.post("/factorize", response_model=FactorizationResponse)
def factorize(request: FactorizationRequest):
    try:
        matrix = np.asarray(request.matrix, dtype=float)

        if matrix.ndim != 2:
            raise ValueError("Matrix must be two-dimensional.")

        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("Matrix cannot be empty.")

        kwargs = {}

        if request.n_components is not None:
            kwargs["n_components"] = request.n_components

        engine = MatrixFactorizationEngine(
            method=request.method,
            **kwargs,
        )

        start_time = time.perf_counter()

        engine.fit(matrix)

        reconstructed = engine.reconstruct()

        runtime_ms = (time.perf_counter() - start_time) * 1000

        reconstruction_error = float(
            np.linalg.norm(matrix - reconstructed)
        )

        return FactorizationResponse(
            method=engine.method,
            matrix_shape=list(matrix.shape),
            n_components=request.n_components,
            input_matrix=matrix.tolist(),
            reconstructed_matrix=np.asarray(reconstructed).tolist(),
            reconstruction_error=reconstruction_error,
            runtime_ms=runtime_ms,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

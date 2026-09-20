from typing import List

from pydantic import BaseModel, Field


class FactorizationRequest(BaseModel):
    matrix: List[List[float]] = Field(
        ...,
        description="Input matrix",
    )
    method: str = Field(
        default="svd",
        description="Factorization method",
    )
    n_components: int | None = Field(
        default=None,
        ge=1,
        description="Number of components for applicable methods",
    )


class FactorizationResponse(BaseModel):
    method: str
    matrix_shape: List[int]
    n_components: int | None = None
    input_matrix: List[List[float]]
    reconstructed_matrix: List[List[float]]
    reconstruction_error: float | None = None
    runtime_ms: float

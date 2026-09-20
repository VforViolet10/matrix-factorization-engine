from typing import List

from pydantic import BaseModel, Field


class ImageCompressionResponse(BaseModel):
    filename: str
    components: int
    original_shape: List[int]
    compressed_shape: List[int]
    compression_ratio: float
    reconstruction_error: float
    relative_reconstruction_error: float
    rmse: float
    compressed_image: str


class RecommendationRequest(BaseModel):
    ratings: List[List[float]] = Field(
        ...,
        description="User-item rating matrix. Use 0 for unrated items.",
    )
    method: str = Field(
        default="svd",
        description="Recommendation factorization method: svd or nmf.",
    )
    n_components: int = Field(
        default=2,
        ge=1,
        description="Number of latent components.",
    )
    user_index: int = Field(
        default=0,
        ge=0,
        description="Zero-based user index.",
    )
    n: int = Field(
        default=3,
        ge=1,
        description="Number of recommendations.",
    )


class RecommendationItem(BaseModel):
    item: str
    predicted_rating: float


class RecommendationResponse(BaseModel):
    method: str
    n_components: int
    user_index: int
    recommendations: List[RecommendationItem]

class RecommendationEvaluationRequest(BaseModel):
    predictions: List[List[float]] = Field(
        ...,
        description="Predicted user-item rating matrix.",
    )
    train_ratings: List[List[float]] = Field(
        ...,
        description="Training rating matrix. Use 0 for unrated items.",
    )
    test_ratings: List[List[float]] = Field(
        ...,
        description="Test rating matrix. Use 0 for unobserved ratings.",
    )
    k: int = Field(
        default=5,
        ge=1,
        description="Number of top recommendations to evaluate.",
    )
    relevance_threshold: float = Field(
        default=4.0,
        description="Minimum rating considered relevant.",
    )


class RecommendationEvaluationResponse(BaseModel):
    rmse: float
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    k: int
    relevance_threshold: float

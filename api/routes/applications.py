import base64
import io

import numpy as np
from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from PIL import Image

from applications.image_compression import (
    SVDImageCompressor,
)
from applications.recommendation_system import (
    MatrixFactorizationRecommender,
    get_item_names,
)

from applications.recommendation_system.evaluation import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse_on_observed_ratings,
)

from api.application_schemas import (
    RecommendationEvaluationRequest,
    RecommendationEvaluationResponse,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
)


router = APIRouter(
    prefix="/applications",
    tags=["Applications"],
)


@router.post(
    "/recommendations",
    response_model=RecommendationResponse,
)
def recommendations(
    request: RecommendationRequest,
):
    try:
        ratings = np.asarray(
            request.ratings,
            dtype=float,
        )

        if ratings.ndim != 2:
            raise ValueError(
                "ratings must be a two-dimensional matrix."
            )

        if (
            ratings.shape[0] == 0
            or ratings.shape[1] == 0
        ):
            raise ValueError(
                "ratings matrix cannot be empty."
            )

        if np.any(ratings < 0):
            raise ValueError(
                "ratings must contain non-negative values."
            )

        if request.user_index >= ratings.shape[0]:
            raise ValueError(
                "user_index is out of range."
            )

        item_names = get_item_names()

        if len(item_names) != ratings.shape[1]:
            raise ValueError(
                "The rating matrix must have the same "
                "number of columns as the available "
                "item names."
            )

        recommender = MatrixFactorizationRecommender(
            method=request.method,
            n_components=request.n_components,
            random_state=42,
        )

        recommender.fit(ratings)

        recommendations = recommender.recommend(
            user_index=request.user_index,
            item_names=item_names,
            ratings=ratings,
            n=request.n,
        )

        return RecommendationResponse(
            method=recommender.method,
            n_components=request.n_components,
            user_index=request.user_index,
            recommendations=[
                RecommendationItem(
                    item=item,
                    predicted_rating=float(
                        predicted_rating
                    ),
                )
                for item, predicted_rating
                in recommendations
            ],
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@router.post("/image-compression")
async def image_compression(
    file: UploadFile = File(...),
    components: int = Form(20),
):
    try:
        if components <= 0:
            raise ValueError(
                "components must be greater than zero."
            )

        image_bytes = await file.read()

        if not image_bytes:
            raise ValueError(
                "Uploaded image is empty."
            )

        original = Image.open(
            io.BytesIO(image_bytes)
        )

        if original.mode not in {
            "L",
            "RGB",
            "RGBA",
        }:
            original = original.convert("RGB")

        compressor = SVDImageCompressor(
            n_components=components
        )

        compressed = compressor.compress(
            original
        )

        metrics = compressor.evaluate(
            original,
            compressed,
        )

        ratio = compressor.compression_ratio(
            original
        )

        output_buffer = io.BytesIO()

        compressed.save(
            output_buffer,
            format="PNG",
        )

        encoded_image = base64.b64encode(
            output_buffer.getvalue()
        ).decode("utf-8")

        return {
            "filename": file.filename,
            "components": components,
            "original_shape": list(
                np.asarray(original).shape
            ),
            "compressed_shape": list(
                np.asarray(compressed).shape
            ),
            "compression_ratio": float(
                ratio
            ),
            "reconstruction_error": float(
                metrics["reconstruction_error"]
            ),
            "relative_reconstruction_error": float(
                metrics[
                    "relative_reconstruction_error"
                ]
            ),
            "rmse": float(
                metrics["rmse"]
            ),
            "compressed_image": (
                "data:image/png;base64,"
                + encoded_image
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@router.post(
    "/recommendations/evaluate",
    response_model=RecommendationEvaluationResponse,
)
def evaluate_recommendations(
    request: RecommendationEvaluationRequest,
):
    try:
        predictions = np.asarray(
            request.predictions,
            dtype=float,
        )

        train_ratings = np.asarray(
            request.train_ratings,
            dtype=float,
        )

        test_ratings = np.asarray(
            request.test_ratings,
            dtype=float,
        )

        if predictions.ndim != 2:
            raise ValueError(
                "predictions must be a two-dimensional matrix."
            )

        if train_ratings.ndim != 2:
            raise ValueError(
                "train_ratings must be a two-dimensional matrix."
            )

        if test_ratings.ndim != 2:
            raise ValueError(
                "test_ratings must be a two-dimensional matrix."
            )

        if (
            predictions.shape
            != train_ratings.shape
            or predictions.shape
            != test_ratings.shape
        ):
            raise ValueError(
                "predictions, train_ratings, and test_ratings "
                "must have the same shape."
            )

        if np.any(train_ratings < 0):
            raise ValueError(
                "train_ratings must contain non-negative values."
            )

        if np.any(test_ratings < 0):
            raise ValueError(
                "test_ratings must contain non-negative values."
            )

        rmse = rmse_on_observed_ratings(
            predictions,
            test_ratings,
        )

        precision = precision_at_k(
            predictions,
            train_ratings,
            test_ratings,
            k=request.k,
            relevance_threshold=request.relevance_threshold,
        )

        recall = recall_at_k(
            predictions,
            train_ratings,
            test_ratings,
            k=request.k,
            relevance_threshold=request.relevance_threshold,
        )

        ndcg = ndcg_at_k(
            predictions,
            train_ratings,
            test_ratings,
            k=request.k,
            relevance_threshold=request.relevance_threshold,
        )

        return RecommendationEvaluationResponse(
            rmse=rmse,
            precision_at_k=precision,
            recall_at_k=recall,
            ndcg_at_k=ndcg,
            k=request.k,
            relevance_threshold=request.relevance_threshold,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

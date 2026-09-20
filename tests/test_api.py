from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["name"] == "Matrix Factorization Engine API"
    assert data["status"] == "running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"


def test_methods():
    response = client.get("/methods")

    assert response.status_code == 200

    data = response.json()

    assert "methods" in data
    assert isinstance(data["methods"], list)
    assert len(data["methods"]) > 0


def test_svd_factorization():
    payload = {
        "matrix": [
            [5, 3, 1, 4],
            [4, 2, 1, 5],
            [1, 5, 4, 2],
            [5, 4, 2, 3],
        ],
        "method": "svd",
        "n_components": 2,
    }

    response = client.post("/factorize", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["method"] == "svd"
    assert data["matrix_shape"] == [4, 4]
    assert data["n_components"] == 2
    assert data["input_matrix"] == payload["matrix"]
    assert len(data["reconstructed_matrix"]) == 4
    assert len(data["reconstructed_matrix"][0]) == 4
    assert data["reconstruction_error"] >= 0
    assert data["runtime_ms"] >= 0


def test_invalid_matrix():
    payload = {
        "matrix": [1, 2, 3],
        "method": "svd",
        "n_components": 2,
    }

    response = client.post("/factorize", json=payload)

    assert response.status_code == 422


def test_invalid_method():
    payload = {
        "matrix": [
            [1, 2],
            [3, 4],
        ],
        "method": "does_not_exist",
        "n_components": 1,
    }

    response = client.post("/factorize", json=payload)

    assert response.status_code == 400



def test_recommendations_svd():
    payload = {
        "ratings": [
            [5, 4, 0, 0, 2, 1, 0, 0],
            [4, 5, 0, 0, 1, 2, 0, 0],
            [1, 2, 5, 4, 0, 0, 0, 1],
            [0, 1, 4, 5, 0, 0, 2, 1],
            [5, 4, 0, 0, 5, 4, 0, 0],
            [4, 5, 0, 0, 4, 5, 0, 0],
            [0, 0, 5, 4, 1, 0, 5, 4],
            [0, 0, 4, 5, 2, 1, 4, 5],
        ],
        "method": "svd",
        "n_components": 2,
        "user_index": 0,
        "n": 3,
    }

    response = client.post(
        "/applications/recommendations",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["method"] == "svd"
    assert data["n_components"] == 2
    assert data["user_index"] == 0
    assert len(data["recommendations"]) == 3

    for recommendation in data["recommendations"]:
        assert "item" in recommendation
        assert "predicted_rating" in recommendation
        assert 1.0 <= recommendation["predicted_rating"] <= 5.0


def test_recommendations_nmf():
    payload = {
        "ratings": [
            [5, 4, 0, 0, 2, 1, 0, 0],
            [4, 5, 0, 0, 1, 2, 0, 0],
            [1, 2, 5, 4, 0, 0, 0, 1],
            [0, 1, 4, 5, 0, 0, 2, 1],
            [5, 4, 0, 0, 5, 4, 0, 0],
            [4, 5, 0, 0, 4, 5, 0, 0],
            [0, 0, 5, 4, 1, 0, 5, 4],
            [0, 0, 4, 5, 2, 1, 4, 5],
        ],
        "method": "nmf",
        "n_components": 2,
        "user_index": 0,
        "n": 3,
    }

    response = client.post(
        "/applications/recommendations",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["method"] == "nmf"
    assert data["n_components"] == 2
    assert data["user_index"] == 0
    assert len(data["recommendations"]) == 3


def test_recommendations_invalid_user():
    payload = {
        "ratings": [
            [5, 4, 0, 0, 2, 1, 0, 0],
            [4, 5, 0, 0, 1, 2, 0, 0],
        ],
        "method": "svd",
        "n_components": 2,
        "user_index": 10,
        "n": 3,
    }

    response = client.post(
        "/applications/recommendations",
        json=payload,
    )

    assert response.status_code == 400


def test_image_compression():
    from io import BytesIO

    from PIL import Image

    image = Image.new(
        "RGB",
        (20, 30),
        color=(120, 80, 200),
    )

    buffer = BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    buffer.seek(0)

    response = client.post(
        "/applications/image-compression",
        files={
            "file": (
                "test.png",
                buffer,
                "image/png",
            )
        },
        data={
            "components": "5",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["filename"] == "test.png"
    assert data["components"] == 5
    assert data["original_shape"] == [30, 20, 3]
    assert data["compressed_shape"] == [30, 20, 3]
    assert data["compression_ratio"] > 1
    assert data["reconstruction_error"] >= 0
    assert data["relative_reconstruction_error"] >= 0
    assert data["rmse"] >= 0
    assert data["compressed_image"].startswith(
        "data:image/png;base64,"
    )


def test_image_compression_invalid_components():
    from io import BytesIO

    from PIL import Image

    image = Image.new(
        "RGB",
        (20, 30),
        color=(120, 80, 200),
    )

    buffer = BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    buffer.seek(0)

    response = client.post(
        "/applications/image-compression",
        files={
            "file": (
                "test.png",
                buffer,
                "image/png",
            )
        },
        data={
            "components": "0",
        },
    )

    assert response.status_code == 400

def test_recommendations_evaluation():
    payload = {
        "predictions": [
            [5, 4, 3, 2],
            [4, 5, 2, 1],
        ],
        "train_ratings": [
            [5, 0, 0, 0],
            [0, 5, 0, 0],
        ],
        "test_ratings": [
            [0, 4, 3, 0],
            [4, 0, 2, 0],
        ],
        "k": 2,
        "relevance_threshold": 4.0,
    }

    response = client.post(
        "/applications/recommendations/evaluate",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["rmse"] == 0.0
    assert data["precision_at_k"] == 0.5
    assert data["recall_at_k"] == 1.0
    assert data["ndcg_at_k"] == 1.0
    assert data["k"] == 2
    assert data["relevance_threshold"] == 4.0


def test_recommendations_evaluation_invalid_shape():
    payload = {
        "predictions": [
            [5, 4, 3],
            [4, 5, 2],
        ],
        "train_ratings": [
            [5, 0, 0, 0],
            [0, 5, 0, 0],
        ],
        "test_ratings": [
            [0, 4, 3, 0],
            [4, 0, 2, 0],
        ],
        "k": 2,
        "relevance_threshold": 4.0,
    }

    response = client.post(
        "/applications/recommendations/evaluate",
        json=payload,
    )

    assert response.status_code == 400

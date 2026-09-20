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
    assert data["input_matrix"] == payload["matrix"]
    assert len(data["reconstructed_matrix"]) == 4
    assert len(data["reconstructed_matrix"][0]) == 4
    assert data["reconstruction_error"] >= 0


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

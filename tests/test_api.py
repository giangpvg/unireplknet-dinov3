import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "version" in data


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "weights_status" in data
    assert "device" in data


def test_info_endpoint(client):
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "UniRepLKNet with DINOv3 Backbone"
    assert data["num_classes"] == 3
    assert len(data["classes"]) == 3


def test_predict_synthetic_endpoint(client):
    payload = {"use_synthetic": True}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["input_shape"] == [1, 1, 128, 1024]
    assert "prediction" in data
    assert "predicted_label" in data["prediction"]
    assert 0.0 <= data["prediction"]["confidence"] <= 1.0


def test_predict_custom_spectrogram(client):
    # Tạo mảng spectrogram mẫu 128 x 1024
    sample_spectrogram = [[0.5] * 1024 for _ in range(128)]
    payload = {"spectrogram": sample_spectrogram, "use_synthetic": False}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["prediction"]["class_id"] in range(3)


def test_predict_invalid_dimensions(client):
    # Truyền mảng 1D không hợp lệ
    payload = {"spectrogram": [1.0, 2.0, 3.0], "use_synthetic": False}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

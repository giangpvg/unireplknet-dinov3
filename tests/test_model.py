import torch
import pytest
from app.model import AudioModelService


@pytest.fixture(scope="module")
def model_service():
    return AudioModelService()


def test_model_initialization(model_service):
    assert model_service.model is not None
    assert model_service.num_classes == 3
    assert len(model_service.classes) == 3
    assert "untrained_weights" in model_service.weights_status or "loaded" in model_service.weights_status


def test_model_predict_shape_and_probability(model_service):
    dummy_input = torch.rand(1, 1, 128, 1024)
    result, shape, latency = model_service.predict(dummy_input)

    assert shape == [1, 1, 128, 1024]
    assert latency > 0
    assert result["class_id"] in range(3)
    assert result["predicted_label"] in model_service.classes
    assert 0.0 <= result["confidence"] <= 1.0

    # Kiểm tra tổng xác suất xấp xỉ 1.0
    prob_sum = sum(result["probabilities"].values())
    assert pytest.approx(prob_sum, rel=1e-2) == 1.0


def test_model_preprocess_2d_input(model_service):
    # Đầu vào 2D [128, 512]
    dummy_2d = torch.rand(128, 512)
    result, shape, latency = model_service.predict(dummy_2d)

    assert shape == [1, 1, 128, 512]
    assert result["predicted_label"] is not None


def test_model_predict_from_audio_bytes(model_service):
    import io
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, torch.randn(16000).numpy(), 16000, format="WAV")
    buf.seek(0)
    result, shape, latency = model_service.predict_audio(buf.read())

    assert shape == [1, 1, 128, 512]
    assert result["predicted_label"] in model_service.classes
    assert latency > 0

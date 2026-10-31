from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["healthy"])
    device: str = Field(..., examples=["cpu"])
    weights_status: str = Field(..., examples=["untrained_weights (fallback mode)"])
    version: str = Field(..., examples=["1.0.0"])


class ModelInfoResponse(BaseModel):
    model_name: str = "UniRepLKNet with DINOv3 Backbone"
    backbone_arch: str = "dinov3_vits16"
    in_channels: int = 1
    expected_input_shape: List[int] = [1, 1, 128, 1024]
    num_classes: int = 10
    classes: List[str] = [
        "silence", "unknown", "yes", "no", "up", 
        "down", "left", "right", "on", "off"
    ]
    weights_status: str
    version: str = "1.0.0"


class PredictJsonRequest(BaseModel):
    """
    Dữ liệu đầu vào dạng JSON:
    - spectrogram: danh sách 2D số thực (128 x 1024) hoặc None để test với tensor ngẫu nhiên
    - use_synthetic: nếu True, hệ thống tự sinh spectrogram mẫu 128x1024 để kiểm thử nhanh
    """
    spectrogram: Optional[List[List[float]]] = None
    use_synthetic: Optional[bool] = False


class PredictionDetail(BaseModel):
    predicted_label: str
    class_id: int
    confidence: float
    probabilities: Dict[str, float]


class PredictResponse(BaseModel):
    status: str
    prediction: PredictionDetail
    input_shape: List[int]
    latency_ms: float

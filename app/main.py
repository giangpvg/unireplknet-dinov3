import logging
from typing import Dict
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.model import model_service
from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictJsonRequest,
    PredictResponse,
    PredictionDetail,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ai_service.api")

app = FastAPI(
    title="UniRepLKNet-DINOv3 Audio Classification Service",
    description="REST API phục vụ suy luận mô hình UniRepLKNet với DINOv3 backbone cho bài toán phân loại âm thanh.",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", summary="Root Endpoint")
def root_endpoint() -> Dict[str, str]:
    return {
        "service": "UniRepLKNet-DINOv3 AI Serving",
        "version": __version__,
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health",
        "model_info": "/info",
    }


@app.get("/health", response_model=HealthResponse, summary="Health Check")
def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        device=str(model_service.device),
        weights_status=model_service.weights_status,
        version=__version__,
    )


@app.get("/info", response_model=ModelInfoResponse, summary="Model Metadata")
def get_model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        model_name="UniRepLKNet with DINOv3 Backbone",
        backbone_arch=model_service.backbone_arch,
        in_channels=1,
        expected_input_shape=[1, 1, 128, 1024],
        num_classes=model_service.num_classes,
        classes=model_service.classes,
        weights_status=model_service.weights_status,
        version=__version__,
    )


@app.post("/predict", response_model=PredictResponse, summary="Predict Audio Spectrogram")
def predict_audio(request: PredictJsonRequest) -> PredictResponse:
    try:
        if request.use_synthetic or request.spectrogram is None:
            # Sinh tensor spectrogram mô phỏng kích thước chuẩn (1, 1, 128, 1024)
            input_tensor = torch.rand(1, 1, 128, 1024)
        else:
            arr = np.array(request.spectrogram, dtype=np.float32)
            if arr.ndim not in (2, 3, 4):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Spectrogram phải có 2, 3 hoặc 4 chiều, nhận được: {arr.ndim} chiều",
                )
            input_tensor = torch.from_numpy(arr)

        pred_detail, input_shape, latency = model_service.predict(input_tensor)

        return PredictResponse(
            status="success",
            prediction=PredictionDetail(**pred_detail),
            input_shape=input_shape,
            latency_ms=latency,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi trong quá trình inference: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi hệ thống khi dự đoán: {str(e)}",
        )

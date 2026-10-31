import os
import time
import logging
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F

from app.core.unireplknet_dinov3 import UniRepLKNet

logger = logging.getLogger("ai_service.model")

DEFAULT_CLASSES = [
    "angry", "neutral", "happy"
]


class AudioModelService:
    def __init__(self, weights_dir: str = "./weights"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_dir = weights_dir
        self.classes = DEFAULT_CLASSES
        self.num_classes = len(self.classes)
        self.backbone_arch = "dinov3_vits16"
        
        # Khởi tạo kiến trúc UniRepLKNet với DINOv3 ViT-S/16 backbone
        self.model = UniRepLKNet(
            in_chans=1,
            num_classes=self.num_classes,
            backbone_arch=self.backbone_arch
        ).to(self.device)
        
        # Quét và nạp trọng số nếu có
        self.weights_status = self._load_weights()
        self.model.eval()
        logger.info(f"Khởi tạo AudioModelService thành công trên {self.device}. Trạng thái weights: {self.weights_status}")

    def _load_weights(self) -> str:
        custom_path = os.environ.get("MODEL_WEIGHTS_PATH")
        target_file = None

        if custom_path and os.path.isfile(custom_path):
            target_file = custom_path
        elif os.path.isdir(self.weights_dir):
            candidates = [
                f for f in os.listdir(self.weights_dir)
                if f.endswith((".pth", ".pt", ".bin"))
            ]
            if candidates:
                target_file = os.path.join(self.weights_dir, candidates[0])

        if target_file:
            try:
                ckpt = torch.load(target_file, map_location=self.device)
                state_dict = ckpt.get("state_dict", ckpt)
                self.model.load_state_dict(state_dict, strict=False)
                return f"loaded: {os.path.basename(target_file)}"
            except Exception as e:
                logger.warning(f"Lỗi khi nạp trọng số từ {target_file}: {e}. Fallback về untrained weights.")
                return f"fallback_untrained (error loading {os.path.basename(target_file)})"
        
        return "untrained_weights (CI/test mode - ready for checkpoint)"

    def preprocess_tensor(self, raw_input: torch.Tensor) -> torch.Tensor:
        """
        Chuẩn hóa tensor đầu vào về kích thước chuẩn (1, 1, 128, 1024).
        """
        tensor = raw_input.to(self.device).float()
        
        # Nếu là 2D [128, 1024] -> [1, 1, 128, 1024]
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        # Đảm bảo đúng kích thước (H, W) = (128, 1024)
        if tensor.shape[-2:] != (128, 1024):
            tensor = F.interpolate(tensor, size=(128, 1024), mode="bilinear", align_corners=False)
            
        return tensor

    def predict(self, input_tensor: torch.Tensor) -> Tuple[Dict, List[int], float]:
        """
        Thực thi inference và đo độ trễ.
        """
        start_time = time.perf_counter()
        processed_input = self.preprocess_tensor(input_tensor)
        
        with torch.no_grad():
            logits = self.model(processed_input)
            probabilities = F.softmax(logits, dim=-1).squeeze(0)
            
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        conf, pred_id = torch.max(probabilities, dim=-1)
        pred_idx = int(pred_id.item())
        pred_conf = float(conf.item())
        
        prob_dict = {
            self.classes[i]: round(float(probabilities[i].item()), 4)
            for i in range(self.num_classes)
        }
        
        result = {
            "predicted_label": self.classes[pred_idx],
            "class_id": pred_idx,
            "confidence": round(pred_conf, 4),
            "probabilities": prob_dict
        }
        
        return result, list(processed_input.shape), latency_ms


# Khởi tạo singleton instance cho API
model_service = AudioModelService()

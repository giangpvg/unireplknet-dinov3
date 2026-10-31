import os
import io
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.unireplknet_dinov3 import UniRepLKNet
from app.audio_utils import audio_to_fbank_tensor

logger = logging.getLogger("ai_service.model")

DEFAULT_CLASSES = [
    "angry", "neutral", "happy"
]


class AudioUniRepLKNet(nn.Module):
    """
    UniRepLKNet với DINOv3 ViT backbone kết hợp MLP Head phân loại
    theo đúng kiến trúc chuẩn trong train.py.
    """
    def __init__(self, num_classes=3, arch='dinov3_vits16', ckpt=None, drop_path_rate=0.1, head_dropout=0.1):
        super().__init__()
        self.unireplknet = UniRepLKNet(
            in_chans=1,
            num_classes=None,  # Trả về pooled embedding (N, embed_dim)
            drop_path_rate=drop_path_rate,
            backbone_arch=arch,
            backbone_ckpt=ckpt
        )
        embed_dim = self.unireplknet.backbone.embed_dim

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim, eps=1e-6),
            nn.Dropout(p=head_dropout) if head_dropout > 0 else nn.Identity(),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        if x.dim() == 3:  # (B, T, F) -> (B, 1, F, T)
            x = x.unsqueeze(1).transpose(2, 3)
        feats = self.unireplknet(x)
        logits = self.mlp_head(feats)
        return logits


class AudioModelService:
    def __init__(self, weights_dir: str = "./weights"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_dir = weights_dir
        self.classes = DEFAULT_CLASSES
        self.num_classes = len(self.classes)
        self.backbone_arch = "dinov3_vits16"
        self.target_length = 512
        self.num_mel_bins = 128
        
        # Khởi tạo kiến trúc AudioUniRepLKNet chuẩn hóa
        self.model = AudioUniRepLKNet(
            num_classes=self.num_classes,
            arch=self.backbone_arch
        ).to(self.device)
        
        # Quét và nạp trọng số nếu có
        self.weights_status = self._load_weights()
        self.model.eval()
        logger.info(f"Khởi tạo AudioModelService thành công trên {self.device}. Trạng thái weights: {self.weights_status}")

    def _download_model_weights(self) -> Optional[str]:
        """
        Tự động tải model weights từ Remote Storage (GitHub Release / Hugging Face)
        khi container khởi động mà chưa có file weights cục bộ.
        """
        download_url = os.environ.get(
            "MODEL_DOWNLOAD_URL",
            "https://github.com/giangpvg/unireplknet-dinov3/releases/download/v1.0.0/dinov3_vits16_pretrain_lvd1689m.pth"
        )
        auto_download = os.environ.get("AUTO_DOWNLOAD_WEIGHTS", "true").lower() in ("1", "true", "yes")
        if not auto_download or not download_url:
            return None

        try:
            os.makedirs(self.weights_dir, exist_ok=True)
            filename = os.path.basename(download_url.split("?")[0]) or "dinov3_vits16_pretrain_lvd1689m.pth"
            dest_path = os.path.join(self.weights_dir, filename)

            # Nếu file đã tồn tại và đủ dung lượng (>1MB) thì dùng luôn
            if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 1000000:
                return dest_path

            logger.info(f"Đang tự động tải model weights từ {download_url} về {dest_path}...")
            import urllib.request
            import shutil

            req = urllib.request.Request(
                download_url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; AudioServingService/1.0)"}
            )
            with urllib.request.urlopen(req, timeout=180) as response, open(dest_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)

            if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 1000000:
                logger.info(f"Tải thành công {dest_path} ({os.path.getsize(dest_path)} bytes).")
                return dest_path
            else:
                logger.warning("File tải về không hợp lệ hoặc quá nhỏ.")
                return None
        except Exception as e:
            logger.warning(f"Không thể tự động tải weights từ {download_url}: {e}. Fallback về untrained weights.")
            return None

    def _load_weights(self) -> str:
        custom_path = os.environ.get("MODEL_WEIGHTS_PATH")
        target_file = None

        if custom_path and os.path.isfile(custom_path):
            target_file = custom_path
        elif os.path.isdir(self.weights_dir):
            candidates = sorted([
                f for f in os.listdir(self.weights_dir)
                if f.endswith((".pth", ".pt", ".bin"))
            ])
            # Prioritize full fine-tuned model checkpoint over raw backbone if both exist
            classifier_ckpts = [f for f in candidates if "best" in f.lower() or "classifier" in f.lower()]
            if classifier_ckpts:
                target_file = os.path.join(self.weights_dir, classifier_ckpts[0])
            elif candidates:
                target_file = os.path.join(self.weights_dir, candidates[0])

        # Nếu chưa có file cục bộ, tự động tải từ Remote Storage (GitHub Release)
        if target_file is None:
            target_file = self._download_model_weights()

        if target_file:
            try:
                ckpt = torch.load(target_file, map_location=self.device, weights_only=False)
                state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
                if isinstance(ckpt, dict) and "classes" in ckpt and isinstance(ckpt["classes"], list):
                    self.classes = ckpt["classes"]
                    self.num_classes = len(self.classes)
                    # Tái khởi tạo head nếu số lớp thay đổi
                    embed_dim = self.model.unireplknet.backbone.embed_dim
                    self.model.mlp_head = nn.Sequential(
                        nn.LayerNorm(embed_dim, eps=1e-6),
                        nn.Linear(embed_dim, self.num_classes)
                    ).to(self.device)

                # Kiểm tra loại checkpoint: Full classifier hay Raw DINOv3 backbone
                if any(k.startswith("unireplknet.") or k.startswith("mlp_head.") for k in state_dict.keys()):
                    self.model.load_state_dict(state_dict, strict=False)
                    return f"loaded_classifier: {os.path.basename(target_file)}"
                elif "patch_embed.proj.weight" in state_dict or "cls_token" in state_dict:
                    # Raw DINOv3 ViT backbone weights
                    if "patch_embed.proj.weight" in state_dict and state_dict["patch_embed.proj.weight"].shape[1] == 3:
                        state_dict["patch_embed.proj.weight"] = (
                            state_dict["patch_embed.proj.weight"].sum(dim=1, keepdim=True) / 3.0
                        )
                    self.model.unireplknet.backbone.vit.load_state_dict(state_dict, strict=False)
                    return f"loaded_backbone: {os.path.basename(target_file)}"
                else:
                    self.model.load_state_dict(state_dict, strict=False)
                    return f"loaded: {os.path.basename(target_file)}"
            except Exception as e:
                logger.warning(f"Lỗi khi nạp trọng số từ {target_file}: {e}. Fallback về untrained weights.")
                return f"fallback_untrained (error loading {os.path.basename(target_file)})"
        
        return "untrained_weights (CI/test mode - ready for checkpoint)"

    def preprocess_tensor(self, raw_input: torch.Tensor) -> torch.Tensor:
        """
        Chuẩn hóa tensor đầu vào về kích thước chuẩn (1, 1, 128, 512).
        """
        tensor = raw_input.to(self.device).float()
        
        # Nếu là 2D [128, length] -> [1, 1, 128, length]
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        # Đảm bảo H và W chia hết cho 16 (DINOv3 patch_size = 16)
        h, w = tensor.shape[-2], tensor.shape[-1]
        target_h = ((h + 15) // 16) * 16 if h % 16 != 0 else h
        target_w = ((w + 15) // 16) * 16 if w % 16 != 0 else w
        
        if (h, w) != (target_h, target_w):
            tensor = F.interpolate(tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)
            
        return tensor

    def predict(self, input_tensor: torch.Tensor) -> Tuple[Dict, List[int], float]:
        """
        Thực thi inference từ tensor spectrogram và đo độ trễ.
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

    def predict_audio(self, audio_source: Union[str, Path, bytes, io.BytesIO]) -> Tuple[Dict, List[int], float]:
        """
        Nhận file âm thanh thô (WAV, MP3,...), tự động trích xuất fbank
        theo pipeline của train.py và thực thi dự đoán.
        """
        fbank_tensor = audio_to_fbank_tensor(
            audio_source,
            target_sr=16000,
            num_mel_bins=self.num_mel_bins,
            target_length=self.target_length
        )
        return self.predict(fbank_tensor)


# Khởi tạo singleton instance cho API
model_service = AudioModelService()

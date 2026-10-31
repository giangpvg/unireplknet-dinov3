# UniRepLKNet with DINOv3 Backbone - Speech Emotion Recognition API

Production-grade Docker container serving **Speech Emotion Recognition (SER)** powered by **UniRepLKNet** with a **DINOv3 (Vision Transformer)** backbone and **FastAPI**.

* **GitHub Repository:** [https://github.com/giangpvg/unireplknet-dinov3](https://github.com/giangpvg/unireplknet-dinov3)
* **Docker Hub Repository:** [https://hub.docker.com/repository/docker/giangpvg/unireplknet-dinov3](https://hub.docker.com/repository/docker/giangpvg/unireplknet-dinov3)
* **Emotion Classes (3 labels):** `angry`, `neutral`, `happy`
* **Input Specifications:** 
  - Raw Audio Files (`.wav`, `.mp3`, `.flac`) automatically resampled to 16kHz.
  - Kaldi/AST Log-mel filterbank feature extraction: `[1, 1, 128, 512]` (128 Mel bins × 512 time frames).

---

## 🧠 Pretrained Model & Weights Management (QUAN TRỌNG)

Dự án áp dụng nguyên lý tách biệt giữa **Code Engine** và **Model Weights (MLOps Best Practice)**.

### 1. Thông tin về File Pretrained Checkpoint
* **Tên file:** `dinov3_vits16_pretrain_lvd1689m.pth` (~82.5 MB).
* **Nguồn gốc & Kiến trúc:** Trọng số pretrained chính thống của mô hình **DINOv3 ViT-S/16** (huấn luyện tự giám sát trên tập dữ liệu quy mô lớn LVD-1689M của Meta AI).
* **Cơ chế Thích ứng kênh tự động (Automatic Channel Adaptation):**
  - Mạng DINOv3 gốc nhận ảnh RGB 3 kênh (`patch_embed.proj.weight` shape `[384, 3, 16, 16]`).
  - Khi xử lý âm thanh, phổ Mel-filterbank chỉ có 1 kênh.
  - Service suy luận tự động phát hiện và chuyển đổi ma trận chiếu kênh:
    $$\text{weight}_{1\text{ch}} = \frac{1}{3} \sum_{c=1}^{3} \text{weight}_{c}$$
  - Nhờ đó, toàn bộ **188 layer weights** (12 khối Transformer blocks, Attention heads, LayerNorms) được nạp trọn vẹn và chuẩn xác 100% (**0 missing keys, 0 unexpected keys**).

### 2. Tại sao file Pretrained không nằm trực tiếp trên Git?
* File `*.pth` (83MB) được đưa vào `.gitignore` để tuân thủ chuẩn MLOps:
  - Giữ cho Git repository siêu nhẹ và không bị dính giới hạn kích thước file của GitHub (100MB).
  - Giúp các runner CI/CD trên GitHub Actions thực thi kiểm thử và build chỉ trong **vài chục giây** mà không bị nghẽn băng thông do tải file nặng.

### 3. Hai cơ chế thực thi (Execution Modes)

#### A. Chế độ Chạy Thật với Pretrained Model (Khuyên dùng khi triển khai)
Đặt file `dinov3_vits16_pretrain_lvd1689m.pth` vào thư mục `./weights/` trên máy host, sau đó gắn volume vào container:
```bash
docker run -d --name ai-service -p 8000:8000 \
  -v $(pwd)/weights:/app/weights \
  giangpvg/unireplknet-dinov3:latest
```
*Khi khởi động, service sẽ quét thư mục `/app/weights`, phát hiện file checkpoint và tự động nạp. Khi gọi API `/info`, trường `weights_status` sẽ báo:*
```json
"weights_status": "loaded_backbone: dinov3_vits16_pretrain_lvd1689m.pth"
```

#### B. Chế độ Kiểm thử Nhanh (Untrained Fallback Mode)
Nếu người dùng kéo image về chạy chay mà **không có file weights**:
```bash
docker run -d --name ai-service -p 8000:8000 giangpvg/unireplknet-dinov3:latest
```
*Hệ thống được thiết kế cơ chế **Graceful Fallback**: service **KHÔNG bị văng lỗi (crash)**, mà tự động khởi tạo cấu trúc mạng mặc định với trạng thái:*
```json
"weights_status": "untrained_weights (CI/test mode - ready for checkpoint)"
```
*Mọi API (`/health`, `/info`, `/predict/audio`) vẫn hoạt động bình thường để phục vụ việc kiểm thử tích hợp hệ thống, đo đạc độ trễ và kiểm tra luồng API.*

---

## 🚀 Hướng dẫn Kiểm thử với cURL

### 1. Kiểm tra trạng thái Sức khỏe và Thiết bị
```bash
curl -s http://localhost:8000/health | jq .
```
**Kết quả mẫu:**
```json
{
  "status": "healthy",
  "device": "cpu",
  "weights_status": "loaded_backbone: dinov3_vits16_pretrain_lvd1689m.pth",
  "version": "1.0.0"
}
```

---

### 2. Xem Metadata của Model và Trạng thái Trọng số
```bash
curl -s http://localhost:8000/info | jq .
```
**Kết quả mẫu:**
```json
{
  "model_name": "UniRepLKNet with DINOv3 Backbone",
  "backbone_arch": "dinov3_vits16",
  "in_channels": 1,
  "expected_input_shape": [1, 1, 128, 512],
  "num_classes": 3,
  "classes": ["angry", "neutral", "happy"],
  "weights_status": "loaded_backbone: dinov3_vits16_pretrain_lvd1689m.pth",
  "version": "1.0.0"
}
```

---

### 3. Phân loại Cảm xúc từ File Âm thanh Thô (`POST /predict/audio`)
Tải tệp âm thanh trực tiếp (`.wav`, `.mp3`, `.flac`) qua giao thức `multipart/form-data`:
```bash
curl -s -X POST "http://localhost:8000/predict/audio" \
  -H "accept: application/json" \
  -F "file=@./path/to/your_voice.wav" | jq .
```
**Kết quả phản hồi (HTTP 200 OK):**
```json
{
  "status": "success",
  "prediction": {
    "predicted_label": "neutral",
    "class_id": 1,
    "confidence": 0.4207,
    "probabilities": {
      "angry": 0.3238,
      "neutral": 0.4207,
      "happy": 0.2555
    }
  },
  "input_shape": [1, 1, 128, 512],
  "latency_ms": 110.16
}
```

---

### 4. Benchmark với Dữ liệu Phổ Âm thanh Mô phỏng (`POST /predict`)
```bash
curl -s -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"use_synthetic": true}' | jq .
```

---

## 📌 Danh sách các Endpoint REST API

| Phương thức | Endpoint | Mô tả chức năng |
|:---|:---|:---|
| `GET` | `/health` | Kiểm tra container còn sống và thiết bị thực thi (CPU/GPU) |
| `GET` | `/info` | Xem thông tin chi tiết mô hình, nhãn phân loại và trạng thái nạp weights |
| `POST` | `/predict/audio` | Phân loại cảm xúc trực tiếp từ file âm thanh thô (`.wav`, `.mp3`) |
| `POST` | `/predict` | Dự đoán từ ma trận phổ hoặc chạy benchmark dữ liệu mô phỏng |

---

## 🛡️ Đặc điểm Kỹ thuật & Bảo mật

* **Dung lượng tối ưu:** Base image `python:3.10-slim` kết hợp bánh xe PyTorch CPU-only, nén trên Docker Hub chỉ còn **~310 MB**.
* **Tăng cường bảo mật:** Container thực thi hoàn toàn dưới quyền người dùng không có đặc quyền root (`appuser`, UID 1000).
* **Tự động hóa hoàn toàn:** Tích hợp pipeline CI/CD khép kín qua GitHub Actions (tự động Lint code bằng `flake8`, chạy 11/11 tests `pytest` và tự động build/push lên Docker Hub).

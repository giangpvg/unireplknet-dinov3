# -------------------------------------------------------------------
# Production Dockerfile for UniRepLKNet-DINOv3 AI Serving
# -------------------------------------------------------------------
FROM python:3.10-slim

# Thiết lập biến môi trường tối ưu cho Python trong container
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# Cài đặt curl để hỗ trợ HEALTHCHECK và các tiện ích hệ thống
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Tạo user non-root để đảm bảo an toàn bảo mật container
RUN useradd -m -u 1000 appuser

# Tận dụng Docker layer caching: copy requirements trước
COPY requirements.txt .

# Cài đặt PyTorch CPU wheel và các dependencies (giúp image nhẹ hơn ~3GB so với GPU CUDA wheel)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy toàn bộ mã nguồn ứng dụng
COPY app/ ./app/
COPY weights/ ./weights/

# Phân quyền cho appuser
RUN chown -R appuser:appuser /app
USER appuser

# Mở cổng phục vụ API
EXPOSE 8000

# Thiết lập kiểm tra trạng thái container tự động
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Khởi chạy ứng dụng FastAPI bằng Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

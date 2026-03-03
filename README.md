# Chart QA Pipeline

Hỏi đáp thông minh về biểu đồ sử dụng:
**YOLO** (phân loại) → **PaddleOCR-VL** (extract data) → **Vintern** (QA)

---

## Cấu trúc project

```
chart_qa/
├── main.py               # FastAPI app
├── pipeline.py           # Pipeline orchestrator
├── config.py             # Cấu hình (paths, device, params)
├── requirements.txt
├── .env                  # (tuỳ chọn) override config
├── models/
│   ├── chart_classifier.py   # YOLO classifier
│   ├── data_extractor.py     # PaddleOCR-VL extractor
│   └── chart_qa.py           # Vintern QA
├── static/
│   └── index.html        # Web UI
└── uploads/              # Ảnh được upload (tự tạo)
```

---

## Cài đặt

### 1. Tạo virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
# hoặc
venv\Scripts\activate      # Windows
```

### 2. Cài PyTorch (theo CUDA version)

```bash
# CUDA 12.1 (Kaggle / phần lớn RTX)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchvision
```

### 3. Cài transformers đúng version

> ⚠️ **Xung đột version**: PaddleOCR-VL cần `transformers>=4.45`, Vintern cần `transformers==4.44.x`.
> Giải pháp đơn giản nhất: cài `transformers>=4.45` và thêm `low_cpu_mem_usage=True` cho Vintern (đã xử lý trong code).

```bash
pip install -r requirements.txt
```

### 4. (Tuỳ chọn) Cài PaddlePaddle nếu dùng PaddleOCR native

```bash
pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
pip install -U "paddleocr[doc-parser]"
```

---

## Cấu hình

Chỉnh `config.py` hoặc tạo file `.env`:

```env
# Đường dẫn YOLO model weights của bạn
YOLO_MODEL_PATH=yolo_chart.pt

# HuggingFace ID hoặc local path
PADDLE_MODEL_PATH=PaddlePaddle/PaddleOCR-VL
VINTERN_MODEL_PATH=5CD-AI/Vintern-1B-v3_5

# cuda hoặc cpu
DEVICE=cuda

# Tham số generation
VINTERN_MAX_NEW_TOKENS=1024
PADDLE_MAX_NEW_TOKENS=512
```

### YOLO model

- Nếu **đã có** file `.pt`: set `YOLO_MODEL_PATH=path/to/your_model.pt`
- Nếu **chưa có**: pipeline vẫn chạy, chart_type sẽ trả về `"unknown"` và Vintern vẫn QA bình thường

---

## Chạy

```bash
cd chart_qa
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#Venv paddle
cd D:\Pipeline_CHartqa
venv_paddle\Scripts\activate
python -m pip install "transformers>=4.45" accelerate pillow fastapi uvicorn python-multipart httpx
python -m uvicorn paddle_server:app --port 8001
Sau đó mở browser: **http://localhost:8000**
#main cụ
cd D:\Pipeline_CHartqa\files
venv\Scripts\activate
python -m pip install "transformers==4.44.2" --force-reinstall
python -m pip install httpx
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
---

## API

### `POST /api/ask`

**Form data:**
| Field    | Type | Mô tả |
|----------|------|--------|
| `image`  | file | Ảnh biểu đồ (PNG/JPG/WEBP, max 10MB) |
| `question` | string | Câu hỏi về biểu đồ |

**Response:**
```json
{
  "question": "Công ty nào có doanh thu cao nhất?",
  "answer": "FPT có doanh thu cao nhất với 80 triệu đồng.",
  "chart_type": "bar_chart",
  "extracted_data": "| Công ty | Doanh thu |\n|---------|----------|\n| FPT | 80 |...",
  "latency": {
    "classify": 0.12,
    "extract": 3.45,
    "qa": 2.87,
    "total": 6.44
  }
}
```

### `GET /health`

Kiểm tra trạng thái models.

### `GET /`

Web UI.

---

## Notes

- **GPU T4 (Kaggle)**: set `DEVICE=cuda`, dùng `float16` cho Paddle, `bfloat16` cho Vintern
- **RAM**: Cả 3 model cần ~10-12GB VRAM tổng
- **Vintern transformers conflict**: code đã dùng `low_cpu_mem_usage=True` để tương thích `transformers>=4.45`

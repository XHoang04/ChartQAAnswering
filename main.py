"""
FastAPI App: Chart Question Answering
Endpoints:
  POST /api/ask  - Upload ảnh + câu hỏi → nhận câu trả lời
  GET  /         - Web UI
  GET  /health   - Health check
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import io
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from config import settings
from pipeline import ChartQAPipeline

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global pipeline instance ─────────────────────────────────────────────────
pipeline: ChartQAPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load pipeline on startup, cleanup on shutdown."""
    global pipeline
    logger.info("⏳ Loading models, please wait...")
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    pipeline = ChartQAPipeline()
    logger.info("🟢 All models loaded. API ready.")
    yield
    logger.info("🔴 Shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Chart QA API",
    description="Chart Question Answering: YOLO → PaddleOCR-VL → Vintern",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": settings.DEVICE,
        "models": {
            "classifier": "loaded" if pipeline and pipeline.classifier.model else "fallback (unknown)",
            "extractor": "loaded" if pipeline and pipeline.extractor.model else "error",
            "qa": "loaded" if pipeline and pipeline.qa.model else "error",
        },
    }


@app.post("/api/ask")
async def ask(
    image: UploadFile = File(..., description="Ảnh biểu đồ (PNG, JPG, JPEG)"),
    question: str = Form(..., description="Câu hỏi về biểu đồ"),
):
    """
    Main endpoint: nhận ảnh biểu đồ + câu hỏi → trả về câu trả lời.

    - **image**: file ảnh biểu đồ
    - **question**: câu hỏi bất kỳ về biểu đồ (tiếng Việt hoặc tiếng Anh)
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline chưa sẵn sàng, thử lại sau.")

    # Validate file type
    allowed_types = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Chỉ chấp nhận ảnh JPEG/PNG/WEBP, nhận được: {image.content_type}",
        )

    # Validate file size
    contents = await image.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.MAX_IMAGE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"Ảnh quá lớn ({size_mb:.1f}MB), tối đa {settings.MAX_IMAGE_SIZE_MB}MB",
        )

    # Load image
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không thể đọc ảnh: {e}")

    # Save upload (optional, for logging)
    upload_name = f"{uuid.uuid4().hex}_{image.filename}"
    upload_path = Path(settings.UPLOAD_DIR) / upload_name
    upload_path.write_bytes(contents)

    # Run pipeline
    try:
        result = pipeline.run(image=pil_image, question=question)
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    return JSONResponse(
        content={
            "question": question,
            "answer": result.answer,
            "chart_type": result.chart_type,
            "extracted_data": result.extracted_data,
            "latency": result.latency,
        }
    )


@app.get("/", response_class=HTMLResponse)
async def ui():
    """Serve the web UI."""
    # Thử nhiều path khác nhau
    possible_paths = [
        Path(__file__).parent / "static" / "index.html",
        Path(__file__).parent / "index.html",
    ]
    for html_path in possible_paths:
        if html_path.exists():
            return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Chart QA API</h1><p>Go to <a href='/docs'>/docs</a></p>")
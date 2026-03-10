"""
Configuration settings for Chart QA project.
Chỉnh sửa các giá trị này theo môi trường của bạn.
"""

import torch
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Model paths ──────────────────────────────────────────────────────────
    # YOLO: trỏ tới file .pt của bạn, hoặc để mặc định (sẽ fallback "unknown")
    RESNET_MODEL_PATH: str = "./models_local/resnet18/resnet18_chart_classifier.pt"

    # PaddleOCR-VL: HuggingFace model ID hoặc local path
    PADDLE_MODEL_PATH: str = "./models_local/paddleocr_vl"

    # Vintern: HuggingFace model ID hoặc local path
    VINTERN_MODEL_PATH: str = "./models_local/vintern_finetuned" 

    # ── Hardware ─────────────────────────────────────────────────────────────
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Generation params ────────────────────────────────────────────────────
    VINTERN_MAX_NEW_TOKENS: int = 1024
    PADDLE_MAX_NEW_TOKENS: int = 512

    # ── API settings ─────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    MAX_IMAGE_SIZE_MB: int = 10

    # ── Upload dir ───────────────────────────────────────────────────────────
    UPLOAD_DIR: str = "uploads"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

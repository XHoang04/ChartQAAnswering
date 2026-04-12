"""
Pipeline: Chart Question Answering
Orchestrates: ResNet18 → PaddleOCR-VL → Vintern
Hỗ trợ multi-turn chat với session
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image

from chart_classifier import ChartClassifier
from data_extractor import ChartDataExtractor
from chart_qa import ChartQA
from config import settings

logger = logging.getLogger(__name__)

SUPPORTED_CHART_TYPES = {"bar", "pie", "line"}


@dataclass
class PipelineResult:
    chart_type: str
    extracted_data: str
    answer: str
    latency: dict
    supported: bool = True
    session_id: str = ""


@dataclass
class Session:
    session_id: str
    image: Image.Image
    chart_type: str
    extracted_data: str
    history: list = field(default_factory=list)  # Vintern chat history
    created_at: float = field(default_factory=time.time)


class ChartQAPipeline:
    """
    Full pipeline với session support:
      1. upload_image() → classify + extract → tạo session
      2. chat() → Vintern multi-turn với history
    """

    def __init__(self):
        logger.info("Initializing Chart QA Pipeline...")
        t0 = time.time()

        self.classifier = ChartClassifier(settings.RESNET_MODEL_PATH, settings.DEVICE)
        self.extractor  = ChartDataExtractor(
            model_path=settings.PADDLE_MODEL_PATH,
            device=settings.DEVICE,
        )
        self.qa = ChartQA(
            model_path=settings.VINTERN_MODEL_PATH,
            device=settings.DEVICE,
        )

        # Session storage (in-memory)
        self._sessions: dict[str, Session] = {}

        logger.info(f"Pipeline ready in {time.time() - t0:.1f}s")

    # ── Session management ────────────────────────────────────────────────

    def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str):
        self._sessions.pop(session_id, None)

    def list_sessions(self) -> list:
        return list(self._sessions.keys())

    # ── Upload: classify + extract → tạo session ─────────────────────────

    def upload_image(self, image: Image.Image) -> PipelineResult:
        total_start = time.time()
        latency = {}

        # Step 1: Classify
        t = time.time()
        chart_type = self.classifier.classify(image)
        latency["classify"] = round(time.time() - t, 2)
        logger.info(f"[1/2] Chart type: {chart_type} ({latency['classify']}s)")

        # Kiểm tra hỗ trợ
        if chart_type not in SUPPORTED_CHART_TYPES:
            latency["total"] = round(time.time() - total_start, 2)
            return PipelineResult(
                chart_type=chart_type,
                extracted_data="",
                answer=f"Hệ thống chỉ hỗ trợ biểu đồ: cột (bar), tròn (pie), đường (line). "
                       f"Biểu đồ của bạn là '{chart_type}', không được hỗ trợ.",
                latency=latency,
                supported=False,
                session_id="",
            )

       # Step 2: Extract
        t = time.time()
        extracted_data = self.extractor.extract(image)
        latency["extract"] = round(time.time() - t, 2)
        logger.info(f"[2/2] Extracted {len(extracted_data)} chars ({latency['extract']}s)")

        # Giải phóng VRAM Paddle ngay sau extract
        try:
            import httpx, gc, torch
            httpx.post("http://localhost:8001/free", timeout=10)
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("Paddle VRAM freed → Vintern ready")
        except Exception as e:
            logger.warning(f"Could not free Paddle VRAM: {e}")

        # Tạo session
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = Session(
            session_id=session_id,
            image=image,
            chart_type=chart_type,
            extracted_data=extracted_data,
            history=[],
        )

        latency["total"] = round(time.time() - total_start, 2)
        logger.info(f"Session created: {session_id}")

        return PipelineResult(
            chart_type=chart_type,
            extracted_data=extracted_data,
            answer="",
            latency=latency,
            supported=True,
            session_id=session_id,
        )

    # ── Chat: multi-turn với session ──────────────────────────────────────

    def chat(self, session_id: str, question: str) -> PipelineResult:
        session = self.get_session(session_id)
        if session is None:
            return PipelineResult(
                chart_type="unknown",
                extracted_data="",
                answer="Session không tồn tại. Vui lòng upload ảnh lại.",
                latency={"total": 0},
                supported=False,
                session_id=session_id,
            )

        total_start = time.time()

        # QA với history
 
        answer, _ = self.qa.answer_with_history(
            image=session.image,
            question=question,
            chart_type=session.chart_type,
            extracted_data=session.extracted_data,
            history=[], 
            max_new_tokens=settings.VINTERN_MAX_NEW_TOKENS,
        )

        # XÓA HOẶC COMMENT DÒNG DƯỚI ĐÂY LẠI
        # session.history = new_history 

        latency = {"qa": round(time.time() - total_start, 2),
                   "total": round(time.time() - total_start, 2)}
        return PipelineResult(
            chart_type=session.chart_type,
            extracted_data=session.extracted_data,
            answer=answer,
            latency=latency,
            supported=True,
            session_id=session_id,
        )
    # ── Backward compat: run() vẫn hoạt động ─────────────────────────────

    def run(self, image: Image.Image, question: str) -> PipelineResult:
        upload_result = self.upload_image(image)
        if not upload_result.supported:
            return upload_result
        return self.chat(upload_result.session_id, question)

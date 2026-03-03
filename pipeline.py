"""
Pipeline: Chart Question Answering
Orchestrates: YOLO → PaddleOCR-VL → Vintern
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import time
from dataclasses import dataclass
from PIL import Image

from chart_classifier import ChartClassifier
from data_extractor import ChartDataExtractor
from chart_qa import ChartQA
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    chart_type: str
    extracted_data: str
    answer: str
    latency: dict  # {"classify": float, "extract": float, "qa": float, "total": float}


class ChartQAPipeline:
    """
    Full pipeline:
      1. ChartClassifier (YOLO)   → chart_type
      2. ChartDataExtractor (Paddle) → extracted_data
      3. ChartQA (Vintern)        → answer
    """

    def __init__(self):
        logger.info("🚀 Initializing Chart QA Pipeline...")
        t0 = time.time()

        self.classifier = ChartClassifier(model_path=settings.YOLO_MODEL_PATH)
        self.extractor = ChartDataExtractor(
            model_path=settings.PADDLE_MODEL_PATH,
            device=settings.DEVICE,
        )
        self.qa = ChartQA(
            model_path=settings.VINTERN_MODEL_PATH,
            device=settings.DEVICE,
        )

        logger.info(f"✅ Pipeline ready in {time.time() - t0:.1f}s")

    def run(self, image: Image.Image, question: str) -> PipelineResult:
        """
        Run full pipeline on a chart image + user question.
        """
        total_start = time.time()
        latency = {}

        # Step 1: Classify chart type
        t = time.time()
        chart_type = self.classifier.classify(image)
        latency["classify"] = round(time.time() - t, 2)
        logger.info(f"[1/3] Chart type: {chart_type} ({latency['classify']}s)")

        # Step 2: Extract data from chart
        t = time.time()
        extracted_data = self.extractor.extract(image)
        latency["extract"] = round(time.time() - t, 2)
        logger.info(f"[2/3] Extracted {len(extracted_data)} chars ({latency['extract']}s)")

        # Step 3: QA with Vintern
        t = time.time()
        answer = self.qa.answer(
            image=image,
            question=question,
            chart_type=chart_type,
            extracted_data=extracted_data,
            max_new_tokens=settings.VINTERN_MAX_NEW_TOKENS,
        )
        latency["qa"] = round(time.time() - t, 2)
        logger.info(f"[3/3] QA done ({latency['qa']}s)")

        latency["total"] = round(time.time() - total_start, 2)

        return PipelineResult(
            chart_type=chart_type,
            extracted_data=extracted_data,
            answer=answer,
            latency=latency,
        )
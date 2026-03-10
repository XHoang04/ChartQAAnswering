"""
Pipeline: Chart Question Answering
Orchestrates: ResNet18 → PaddleOCR-VL → Vintern
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

SUPPORTED_CHART_TYPES = {"bar", "pie", "line"}


@dataclass
class PipelineResult:
    chart_type: str
    extracted_data: str
    answer: str
    latency: dict  # {"classify": float, "extract": float, "qa": float, "total": float}
    supported: bool = True  # False nếu chart type không hỗ trợ


class ChartQAPipeline:
    """
    Full pipeline:
      1. ChartClassifier (ResNet18) → chart_type
         → Nếu không phải bar/pie/line: dừng, báo không hỗ trợ
      2. ChartDataExtractor (Paddle) → extracted_data
      3. ChartQA (Vintern)           → answer
    """

    def __init__(self):
        logger.info(" Initializing Chart QA Pipeline...")
        t0 = time.time()

        self.classifier = ChartClassifier(settings.RESNET_MODEL_PATH, settings.DEVICE)
        self.extractor = ChartDataExtractor(
            model_path=settings.PADDLE_MODEL_PATH,
            device=settings.DEVICE,
        )
        self.qa = ChartQA(
            model_path=settings.VINTERN_MODEL_PATH,
            device=settings.DEVICE,
        )

        logger.info(f" Pipeline ready in {time.time() - t0:.1f}s")

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

        # Kiểm tra hỗ trợ — chỉ tiếp tục nếu là bar / pie / line
        if chart_type not in SUPPORTED_CHART_TYPES:
            latency["total"] = round(time.time() - total_start, 2)
            logger.info(f"[!] Chart type '{chart_type}' không hỗ trợ → dừng pipeline")
            return PipelineResult(
                chart_type=chart_type,
                extracted_data="",
                answer=(
                    f"Hệ thống hiện chỉ hỗ trợ biểu đồ dạng: cột (bar), tròn (pie), đường (line). "
                    f"Biểu đồ của bạn được nhận dạng là '{chart_type}', không nằm trong danh sách hỗ trợ."
                ),
                latency=latency,
                supported=False,
            )

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
            supported=True,
        )
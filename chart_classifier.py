"""
Module: Chart Classifier using YOLO
Nhận ảnh biểu đồ → trả về loại chart (bar, line, pie, etc.)
"""

import logging
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

# Mapping label index → tên chart type
CHART_LABELS = {
        0: 'v_bar',
        1: 'h_bar',
        2: 'line',
        3: 'other',
        4: 'pie',
        # 5: 'plot_area',
        # 6: 'x_axis',
        # 7: 'y_axis',
        # 8: 'title',
        # 9: 'legend'
}


class ChartClassifier:
    """
    YOLO-based chart type classifier.
    Nếu không có YOLO model weights, fallback về 'unknown'.
    """

    def __init__(self, model_path: str = "yolo_chart.pt"):
        self.model = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            if Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
                logger.info(f" YOLO model loaded from {self.model_path}")
            else:
                logger.warning(
                    f"⚠️  YOLO weights not found at '{self.model_path}'. "
                    "ChartClassifier will return 'unknown' for all inputs. "
                    "Provide a valid YOLO model path in config.py to enable classification."
                )
        except ImportError:
            logger.warning("⚠️  ultralytics not installed. pip install ultralytics")
        except Exception as e:
            logger.warning(f"⚠️  Cannot load YOLO model: {e}")

    def classify(self, image: Image.Image) -> str:
        """
        Classify chart type from PIL Image.
        Returns chart type string.
        """
        if self.model is None:
            return "unknown"

        try:
            results = self.model(image, verbose=False)
            if results and len(results) > 0:
                result = results[0]
                # Classification task: top1 class
                if hasattr(result, "probs") and result.probs is not None:
                    top1 = int(result.probs.top1)
                    chart_type = CHART_LABELS.get(top1, "unknown")
                    conf = float(result.probs.top1conf)
                    logger.info(f"Chart classified as: {chart_type} (conf={conf:.2f})")
                    return chart_type
                # Detection task: most confident box class
                elif hasattr(result, "boxes") and result.boxes is not None and len(result.boxes):
                    cls_id = int(result.boxes.cls[0])
                    chart_type = CHART_LABELS.get(cls_id, "unknown")
                    return chart_type
        except Exception as e:
            logger.error(f"YOLO inference error: {e}")

        return "unknown"

import os
import io
import base64
import logging
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WeldDetector")


class WeldDetector:
    def __init__(self, model_path=None):
        """Initialize the YOLO Weld Detector with GPU/CPU auto-selection."""
        if model_path is None:
            model_path = config.MODEL_PATH

        if not os.path.exists(model_path) and os.path.exists(config.FALLBACK_MODEL_PATH):
            logger.info(f"Primary model not found at {model_path}. Using fallback: {config.FALLBACK_MODEL_PATH}")
            model_path = config.FALLBACK_MODEL_PATH

        self.model_path = model_path
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.class_names = {}
        self.loaded = False

        self._load_model()

    def _load_model(self):
        """Loads the YOLO model from specified path."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at: {self.model_path}")

            logger.info(f"Loading YOLO model from: {self.model_path} on device: {self.device}")
            self.model = YOLO(self.model_path)
            self.class_names = self.model.names if hasattr(self.model, "names") else {}
            self.loaded = True
            logger.info(f"Model successfully loaded. Classes detected: {self.class_names}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
            self.loaded = False
            raise e

    def classify_weld_quality(self, class_name):
        """Categorize class name into GOOD or BAD weld based on config rules."""
        name = str(class_name).strip().lower()

        # Check Good matches
        for good_pattern in config.GOOD_CLASS_NAMES:
            if good_pattern in name:
                return "GOOD"

        # Check Bad matches
        for bad_pattern in config.BAD_CLASS_NAMES:
            if bad_pattern in name:
                return "BAD"

        # Fallback heuristic: if it contains 'good' -> GOOD, 'bad' or 'defect' -> BAD
        if "good" in name:
            return "GOOD"
        elif "bad" in name or "defect" in name:
            return "BAD"

        # Default fallback for unmapped classes
        return "UNKNOWN"

    def predict(self, image_input, conf_threshold=None):
        """
        Run YOLO inference on an image input (file path, PIL Image, bytes, or numpy BGR array).

        Returns structured dict with quality inspection, detections, and base64 rendered images.
        """
        if not self.loaded or self.model is None:
            raise RuntimeError("YOLO model is not loaded.")

        if conf_threshold is None:
            conf_threshold = config.DEFAULT_CONFIDENCE_THRESHOLD

        # 1. Parse Image Input into BGR numpy array
        img_bgr, original_b64 = self._parse_image_input(image_input)

        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Invalid image input provided.")

        # 2. Run YOLO Inference
        results = self.model.predict(
            source=img_bgr,
            conf=conf_threshold,
            imgsz=config.IMAGE_SIZE,
            device=self.device,
            verbose=False
        )

        detections = []
        annotated_img = img_bgr.copy()

        good_count = 0
        bad_count = 0
        max_conf_percent = 0.0

        # Colors in BGR format
        COLOR_GOOD = (0, 230, 118)    # Green
        COLOR_BAD = (54, 67, 244)     # Bright Red
        COLOR_UNKNOWN = (7, 193, 255) # Amber

        # 3. Process Detections
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0].item())
                cls_id = int(box.cls[0].item())
                cls_name = self.class_names.get(cls_id, f"Class {cls_id}")
                conf_pct = round(conf * 100, 2)

                if conf_pct > max_conf_percent:
                    max_conf_percent = conf_pct

                quality = self.classify_weld_quality(cls_name)

                if quality == "GOOD":
                    good_count += 1
                    color = COLOR_GOOD
                    badge_label = f"GOOD: {cls_name} {conf_pct}%"
                elif quality == "BAD":
                    bad_count += 1
                    color = COLOR_BAD
                    badge_label = f"BAD: {cls_name} {conf_pct}%"
                else:
                    # Treat unknown as bad for safety, but log as unknown
                    bad_count += 1
                    color = COLOR_UNKNOWN
                    badge_label = f"{cls_name} {conf_pct}%"

                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 4),
                    "confidence_percent": conf_pct,
                    "bbox": [x1, y1, x2, y2],
                    "quality": quality
                })

                # Draw Bounding Box
                thickness = max(2, int(round(min(img_bgr.shape[:2]) / 300)))
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, thickness)

                # Label Text Box
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = max(0.5, min(img_bgr.shape[:2]) / 800)
                font_thickness = max(1, int(font_scale * 2))

                (text_width, text_height), baseline = cv2.getTextSize(
                    badge_label, font, font_scale, font_thickness
                )

                # Ensure text stays within frame
                label_y1 = max(0, y1 - text_height - 10)
                label_y2 = y1 if y1 - text_height - 10 >= 0 else y1 + text_height + 10

                # Background fill for text badge
                cv2.rectangle(
                    annotated_img,
                    (x1, label_y1),
                    (x1 + text_width + 10, label_y2),
                    color,
                    cv2.FILLED
                )

                # Draw text in white or dark depending on background
                text_color = (255, 255, 255)
                cv2.putText(
                    annotated_img,
                    badge_label,
                    (x1 + 5, label_y2 - 5 if y1 - text_height - 10 >= 0 else label_y2 - 5),
                    font,
                    font_scale,
                    text_color,
                    font_thickness,
                    cv2.LINE_AA
                )

        # 4. Overall Decision System
        total_detections = len(detections)

        if total_detections == 0:
            overall_result = "NO WELD DETECTED"
        elif bad_count > 0:
            overall_result = "BAD WELD"
        else:
            overall_result = "GOOD WELD"

        # 5. Base64 Encode Annotated Image
        processed_b64 = self._encode_bgr_to_b64(annotated_img)

        return {
            "success": True,
            "overall_result": overall_result,
            "detections": detections,
            "detection_count": total_detections,
            "good_count": good_count,
            "bad_count": bad_count,
            "highest_confidence_percent": max_conf_percent,
            "processed_image": f"data:image/jpeg;base64,{processed_b64}",
            "original_image": f"data:image/jpeg;base64,{original_b64}" if original_b64 else None,
            "device": self.device.upper(),
            "confidence_threshold_used": conf_threshold
        }

    def _parse_image_input(self, image_input):
        """Converts various image inputs into BGR numpy array and original Base64 string."""
        img_bgr = None
        original_b64 = None

        try:
            if isinstance(image_input, str):
                # File path
                if os.path.exists(image_input):
                    img_bgr = cv2.imread(image_input)
                    with open(image_input, "rb") as f:
                        original_b64 = base64.b64encode(f.read()).decode("utf-8")
            elif isinstance(image_input, bytes):
                # Raw image bytes
                nparr = np.frombuffer(image_input, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                original_b64 = base64.b64encode(image_input).decode("utf-8")
            elif isinstance(image_input, Image.Image):
                # PIL Image
                rgb = np.array(image_input.convert("RGB"))
                img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                buffered = io.BytesIO()
                image_input.save(buffered, format="JPEG")
                original_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            elif isinstance(image_input, np.ndarray):
                # Numpy array
                img_bgr = image_input
                original_b64 = self._encode_bgr_to_b64(img_bgr)
            else:
                logger.error(f"Unsupported image input type: {type(image_input)}")
        except Exception as e:
            logger.error(f"Error parsing image input: {e}")

        return img_bgr, original_b64

    def _encode_bgr_to_b64(self, bgr_array):
        """Helper to convert BGR numpy array to JPEG Base64 string."""
        _, buffer = cv2.imencode(".jpg", bgr_array, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return base64.b64encode(buffer).decode("utf-8")

import os

# Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model configuration
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
FALLBACK_MODEL_PATH = os.path.join(
    BASE_DIR, "runs", "runs", "detect", "YOLO weld detection project", "weld_yolo_training", "weights", "best.pt"
)

# Inference Defaults
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 640

# Upload configuration
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max limit
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}

# Class Mappings (Normalized lowercase for comparison)
GOOD_CLASS_NAMES = [
    "good weld",
    "good_weld",
    "good",
    "goodweld"
]

BAD_CLASS_NAMES = [
    "bad weld",
    "bad_weld",
    "bad",
    "badweld",
    "defect",
    "crack",
    "porosity",
    "undercut",
    "spatter",
    "incomplete weld",
    "incomplete_weld"
]

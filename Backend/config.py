from pathlib import Path
import torch


# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# DATA DIRECTORIES
# ============================================================

DATA_DIR = BASE_DIR / "data"

SAMPLES_DIR = DATA_DIR / "samples"
TEST_DIR = DATA_DIR / "test"


# ============================================================
# MODEL DIRECTORIES
# ============================================================

MODEL_DIR = BASE_DIR / "models_weights"

VQA_MODEL_DIR = MODEL_DIR / "vqa"
CAPTIONING_MODEL_DIR = MODEL_DIR / "captioning"
CHANGE_MODEL_DIR = MODEL_DIR / "change_detection"
OPTICAL_SAR_MODEL_DIR = MODEL_DIR / "optical_sar"


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR = BASE_DIR / "outputs"

CHANGE_MAP_DIR = OUTPUT_DIR / "change_maps"
EVIDENCE_DIR = OUTPUT_DIR / "evidence"
REPORT_DIR = OUTPUT_DIR / "reports"


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# SUPPORTED FILE FORMATS
# ============================================================

SUPPORTED_GEOSPATIAL_FORMATS = {
    ".tif",
    ".tiff",
}

SUPPORTED_IMAGE_FORMATS = {
    ".png",
    ".jpg",
    ".jpeg",
}


# ============================================================
# APPLICATION SETTINGS
# ============================================================

APP_NAME = "SatQuery AI"

MAX_UPLOAD_SIZE_MB = 200
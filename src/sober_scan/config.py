"""Configuration settings for sober_scan."""

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict

# Base paths
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODEL_DIR = ROOT_DIR / "models"

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# Blood Alcohol Concentration (BAC) thresholds
class BACLevel(Enum):
    """Blood Alcohol Concentration levels."""

    SOBER = 0
    MILD = 1  # 0.01 - 0.05 BAC
    MODERATE = 2  # 0.06 - 0.10 BAC
    SEVERE = 3  # 0.11+ BAC


# Model types
class ModelType(str, Enum):
    """Types of models that can be downloaded.

    The four drowsiness models (svm/nb/knn/rf) were removed in this
    refactor: their training labels were computed by thresholding the
    same EAR feature the models were trained on, so the perfect
    accuracy was a tautology. Drowsiness detection now uses a direct
    EAR rule (see ``commands/detect.py``).
    """

    DLIB_SHAPE_PREDICTOR = "dlib-shape-predictor"
    CNN = "cnn"
    ALL = "all"


# Model URLs
MODEL_URLS = {
    ModelType.DLIB_SHAPE_PREDICTOR: "https://github.com/Sang-Buster/Sober-Scan/releases/download/0.0.1/shape_predictor_68_face_landmarks.dat.bz2",
    ModelType.CNN: "https://github.com/Sang-Buster/Sober-Scan/releases/download/0.0.1/intoxication_cnn.pt",
}

# Model file names
MODEL_FILENAMES = {
    ModelType.DLIB_SHAPE_PREDICTOR: "shape_predictor_68_face_landmarks.dat",
    ModelType.CNN: "cnn.pt",
}

# Model descriptions
MODEL_DESCRIPTIONS = {
    ModelType.DLIB_SHAPE_PREDICTOR: "68-point facial landmark predictor model from dlib",
    ModelType.CNN: (
        "Pre-trained CNN model for intoxication detection. "
        "NOTE: published 81% accuracy was a leaky-split artifact; "
        "honest cross-subject AUC is ~0.48 (below random). See "
        "models/MODEL_EVALUATION_REPORT.md."
    ),
}


# Feature extraction parameters
FEATURE_PARAMS = {
    "face_detector": "dlib",  # Alternative: "mediapipe", "opencv"
    "landmark_model": "dlib",  # 68-point facial landmark detector
    "eye_aspect_ratio_threshold": 0.2,  # Below this is considered eye closure
    "skin_regions": ["forehead", "cheeks"],  # Regions to analyze for redness
    "landmark_distances": [  # Pairs of landmark indices to measure
        (36, 45),  # Left eye to right eye
        (48, 54),  # Mouth width
        (51, 57),  # Mouth height
        (21, 22),  # Eyebrow distance
    ],
}

# Logging configuration
LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filename": str(DATA_DIR / "sober_scan.log"),
            "mode": "a",
        },
    },
    "loggers": {"sober_scan": {"level": "DEBUG", "handlers": ["console", "file"], "propagate": False}},
}

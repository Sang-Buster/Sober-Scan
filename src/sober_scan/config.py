"""Configuration settings for sober_scan."""

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict

# Base paths
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "models"

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# Blood Alcohol Concentration (BAC) thresholds
class BACLevel(Enum):
    """Blood Alcohol Concentration levels."""

    SOBER = 0
    MILD = 1  # 0.01 - 0.05 BAC
    MODERATE = 2  # 0.06 - 0.10 BAC
    SEVERE = 3  # 0.11+ BAC


# Mapping BAC level to probability thresholds
BAC_THRESHOLDS = {
    BACLevel.SOBER: 0.3,  # Below this probability -> sober
    BACLevel.MILD: 0.5,  # Between sober and this -> mild
    BACLevel.MODERATE: 0.7,  # Between mild and this -> moderate
    BACLevel.SEVERE: 1.0,  # Above moderate -> severe
}

# Model paths and configurations
DEFAULT_MODELS = {
    "traditional": {
        "path": str(MODEL_DIR / "traditional_svm.joblib"),
        "type": "svm",
    },
    "cnn": {
        "path": str(MODEL_DIR / "deep_cnn.pt"),
        "input_size": (224, 224),
    },
    "gnn": {
        "path": str(MODEL_DIR / "graph_nn.pt"),
        "landmarks": 68,
    },
    "video": {
        "path": str(MODEL_DIR / "lstm_temporal.pt"),
        "sequence_length": 16,
    },
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

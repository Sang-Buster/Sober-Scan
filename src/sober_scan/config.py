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


# Model types
class ModelType(str, Enum):
    """Types of models that can be downloaded."""

    DLIB_SHAPE_PREDICTOR = "dlib-shape-predictor"
    TRADITIONAL_SVM = "traditional-svm"
    DEEP_CNN = "deep-cnn"
    DEEP_GNN = "deep-gnn"
    TEMPORAL_LSTM = "temporal-lstm"
    ALL = "all"


# Model URLs
MODEL_URLS = {
    ModelType.DLIB_SHAPE_PREDICTOR: "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks_GTX.dat.bz2",
    ModelType.TRADITIONAL_SVM: "N/A",
    ModelType.DEEP_CNN: "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    ModelType.DEEP_GNN: "https://github.com/ZhaoJ9014/face.evoLVe/raw/master/misc/model_ir_se50.pth",
    ModelType.TEMPORAL_LSTM: "N/A",
}

# Model file names
MODEL_FILENAMES = {
    ModelType.DLIB_SHAPE_PREDICTOR: "shape_predictor_68_face_landmarks.dat",
    ModelType.TRADITIONAL_SVM: "traditional_svm.joblib",
    ModelType.DEEP_CNN: "deep_cnn.pt",
    ModelType.DEEP_GNN: "graph_nn.pt",
    ModelType.TEMPORAL_LSTM: "lstm_temporal.pt",
}

# Model descriptions
MODEL_DESCRIPTIONS = {
    ModelType.DLIB_SHAPE_PREDICTOR: "68-point facial landmark predictor model from dlib",
    ModelType.TRADITIONAL_SVM: "Pre-trained SVM model for intoxication classification",
    ModelType.DEEP_CNN: "Pre-trained CNN model for intoxication detection from images",
    ModelType.DEEP_GNN: "Pre-trained Graph Neural Network for landmark-based intoxication detection",
    ModelType.TEMPORAL_LSTM: "Pre-trained LSTM model for video-based intoxication detection",
}


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

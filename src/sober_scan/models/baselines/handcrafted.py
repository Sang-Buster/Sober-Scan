"""``HandcraftedFeaturesLR``: logistic regression on the project's existing handcrafted features.

The existing ``feature_extraction.extract_features`` produces a small
fixed bag of numeric features (face/forehead/cheek redness, several
landmark distances, EAR, MAR, brow distances, face position+size).
This baseline scales them and fits a logistic regression. It is the
canonical \u201ccan our handcrafted features detect intoxication at all
across subjects?\u201d question, asked honestly.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sober_scan.corpus import IntoxicationCorpus, Photo
from sober_scan.feature_extraction import extract_features
from sober_scan.utils import load_image

_FEATURE_KEYS: List[str] = [
    "face_redness",
    "forehead_redness",
    "cheeks_redness",
    "landmark_dist_36_45",
    "landmark_dist_48_54",
    "landmark_dist_51_57",
    "landmark_dist_21_22",
    "eye_aspect_ratio",
    "left_brow_eye_dist",
    "right_brow_eye_dist",
    "mouth_aspect_ratio",
    "face_rel_x",
    "face_rel_y",
    "face_rel_size",
]


def _features_to_vector(features: dict) -> Optional[np.ndarray]:
    """Project the feature dict onto the fixed schema.

    Returns ``None`` if the dict is empty (face detection failed) so the
    caller can decide whether to drop the photo or fall back to a default.
    """
    if not features:
        return None
    return np.array([float(features.get(key, 0.0)) for key in _FEATURE_KEYS])


@lru_cache(maxsize=None)
def _extract_for_path(path: Path) -> Optional[np.ndarray]:
    """Extract a feature vector for the image at ``path``, memoized.

    Cached across folds so each photo is read and face-detected once
    per process. Returns ``None`` when the image won't load or no face
    is found.
    """
    image = load_image(path)
    if image is None:
        return None
    features = extract_features(image)
    return _features_to_vector(features)


class HandcraftedFeaturesLR:
    """Logistic regression on the existing handcrafted feature bag."""

    def __init__(self) -> None:
        self._pipeline: Optional[Pipeline] = None
        self._fallback_proba: float = 0.5

    def fit(
        self, train: IntoxicationCorpus, *, threshold: float
    ) -> "HandcraftedFeaturesLR":
        labels = train.binary_labels(threshold=threshold)

        X: List[np.ndarray] = []
        y: List[int] = []
        for photo, label in zip(train, labels):
            vec = _extract_for_path(photo.path)
            if vec is None:
                continue
            X.append(vec)
            y.append(label)

        if not X:
            raise RuntimeError(
                "no usable training photos: feature extraction failed for all"
            )

        self._fallback_proba = sum(y) / len(y)
        self._pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=0,
                    ),
                ),
            ]
        )
        self._pipeline.fit(np.array(X), np.array(y))
        return self

    def predict_proba(self, photo: Photo) -> float:
        if self._pipeline is None:
            raise RuntimeError("HandcraftedFeaturesLR must be fit before predicting")

        vec = _extract_for_path(photo.path)
        if vec is None:
            return self._fallback_proba

        return float(self._pipeline.predict_proba(vec.reshape(1, -1))[0, 1])

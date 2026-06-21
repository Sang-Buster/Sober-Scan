"""``ImageNetFrozenLR``: logistic regression on frozen MobileNetV2 features.

Uses the pretrained MobileNetV2 backbone *correctly*:
- RGB input (no first-layer surgery),
- face crop with margin (so background isn't a confound),
- ImageNet normalization,
- backbone frozen and in ``eval`` mode (no batch-norm drift, no dropout).

This is the second Tier 1 ceiling \u2014 if even a real visual representation
can't separate sober from drunk across subjects, the dataset is the limit.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import models, transforms

from sober_scan.corpus import IntoxicationCorpus, Photo
from sober_scan.feature_extraction import detect_face_and_landmarks
from sober_scan.utils import load_image

_IMAGE_SIZE = 224
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def _backbone() -> torch.nn.Module:
    """Singleton MobileNetV2 with the classifier head stripped."""
    model = models.mobilenet_v2(weights="DEFAULT")
    model.classifier = torch.nn.Identity()  # output the 1280-d pooled features
    model.eval()
    model.to(_DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((_IMAGE_SIZE, _IMAGE_SIZE)),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
)


def _face_crop(image: np.ndarray) -> Optional[np.ndarray]:
    """Crop to the detected face with a 20% margin. ``None`` if no face."""
    face_rect, _ = detect_face_and_landmarks(image)
    if face_rect is None:
        return None
    x1, y1, x2, y2 = face_rect
    h, w = image.shape[:2]
    margin_x = int((x2 - x1) * 0.2)
    margin_y = int((y2 - y1) * 0.2)
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


@lru_cache(maxsize=None)
def _extract_for_path(path: Path) -> Optional[np.ndarray]:
    """Return a 1280-d ImageNet feature vector for the face in ``path``."""
    image = load_image(path)
    if image is None:
        return None
    crop = _face_crop(image)
    if crop is None:
        return None
    if crop.shape[2] == 1:  # grayscale -> RGB
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    tensor = _TRANSFORM(crop).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        features = _backbone()(tensor).cpu().numpy()
    return features.flatten()


class ImageNetFrozenLR:
    """Frozen MobileNetV2 + StandardScaler + LogisticRegression."""

    def __init__(self) -> None:
        self._pipeline: Optional[Pipeline] = None
        self._fallback_proba: float = 0.5

    def fit(
        self, train: IntoxicationCorpus, *, threshold: float
    ) -> "ImageNetFrozenLR":
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
            raise RuntimeError("no usable training photos for ImageNetFrozenLR")

        self._fallback_proba = sum(y) / len(y)
        self._pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        C=0.1,  # heavier regularization for 1280-d features
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=0,
                    ),
                ),
            ]
        )
        self._pipeline.fit(np.array(X), np.array(y))
        return self

    def predict_proba(self, photo: Photo) -> float:
        if self._pipeline is None:
            raise RuntimeError("ImageNetFrozenLR must be fit before predicting")
        vec = _extract_for_path(photo.path)
        if vec is None:
            return self._fallback_proba
        return float(self._pipeline.predict_proba(vec.reshape(1, -1))[0, 1])

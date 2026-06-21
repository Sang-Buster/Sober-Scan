"""``HybridFeaturesLR``: logistic regression on handcrafted \u2295 ImageNet features.

The two best individual signals on this dataset are:

- ``HandcraftedFeaturesLR`` (14-d redness + landmark geometry + EAR/MAR)
- the frozen MobileNetV2 1280-d face-crop embedding

These two are leveraging *different* information \u2014 handcrafted captures
*state* (flushing, eye openness), ImageNet captures *appearance* in a
generic semantic feature space. Concatenating them gives the
classifier both axes at once. If the ImageNet features add anything
on top of handcrafted, this baseline should beat the best of the two
parents and become the new floor.

Both extractors are already cached at module level, so building the
joint vector is essentially free after the first pass.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sober_scan.corpus import IntoxicationCorpus, Photo
from sober_scan.models.baselines.handcrafted import (
    _extract_for_path as _extract_handcrafted,
)
from sober_scan.models.baselines.imagenet import (
    _extract_for_path as _extract_imagenet,
)


@lru_cache(maxsize=None)
def _extract_hybrid(path: Path) -> Optional[np.ndarray]:
    """Return the 1294-d ``[handcrafted (14) | imagenet (1280)]`` vector.

    Returns ``None`` if *either* extractor fails (image won't load, no
    face detected, etc.). Failing partially would silently bias the
    learned weights so we drop the photo instead.
    """
    handcrafted = _extract_handcrafted(path)
    imagenet = _extract_imagenet(path)
    if handcrafted is None or imagenet is None:
        return None
    return np.concatenate([handcrafted, imagenet])


class HybridFeaturesLR:
    """Logistic regression on the concatenated handcrafted+ImageNet vector."""

    def __init__(self) -> None:
        self._pipeline: Optional[Pipeline] = None
        self._fallback_proba: float = 0.5

    def fit(
        self, train: IntoxicationCorpus, *, threshold: float
    ) -> "HybridFeaturesLR":
        labels = train.binary_labels(threshold=threshold)
        X: List[np.ndarray] = []
        y: List[int] = []
        for photo, label in zip(train, labels):
            vec = _extract_hybrid(photo.path)
            if vec is None:
                continue
            X.append(vec)
            y.append(label)

        if not X:
            raise RuntimeError("no usable training photos for HybridFeaturesLR")

        self._fallback_proba = sum(y) / len(y)
        # L1 penalty so the LR has to *select* features rather than spread
        # mass across all 1294 dims. With L2 the 1280 ImageNet features
        # drown out the 14 handcrafted ones simply by count; L1 lets the
        # 14 handcrafted features earn their place explicitly.
        self._pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        penalty="l1",
                        solver="liblinear",
                        C=0.1,
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
            raise RuntimeError("HybridFeaturesLR must be fit before predicting")
        vec = _extract_hybrid(photo.path)
        if vec is None:
            return self._fallback_proba
        return float(self._pipeline.predict_proba(vec.reshape(1, -1))[0, 1])

"""``SiameseDelta``: within-subject baseline-vs-query classifier.

The strongest supervision signal in ``data/testing_data`` is the
within-subject pairing (each participant has both ``bd`` baseline
photos and ``ad`` after-drinking photos). Absolute-classification
models throw that signal away by treating every photo independently.
``SiameseDelta`` exploits it directly:

- Pick one ``bd`` photo per subject as that subject's *baseline reference*.
- For training, build pairs ``(bd_reference, query)`` from the same
  subject, with label = ``1`` iff the query's BAC \u2265 threshold.
- Represent each photo as a frozen MobileNetV2 embedding (the same
  features ``ImageNetFrozenLR`` uses, cached across folds).
- Train logistic regression on the *delta* vector
  ``feature(query) - feature(bd_reference)``.
- At inference, find the held-out subject's baseline reference on
  disk and score the query the same way.

The delta framing means the classifier is trying to learn "what changes
about a face when its owner gets drunk" rather than "what does a drunk
face look like in absolute terms" \u2014 the latter is essentially impossible
with 11 subjects.
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sober_scan.corpus import IntoxicationCorpus, Photo
from sober_scan.models.baselines.imagenet import _extract_for_path


@lru_cache(maxsize=None)
def _find_bd_reference(parent_dir: Path, subject_id: str) -> Optional[Path]:
    """Return the lexicographically-first ``Px_bd_*.jpg`` for a subject."""
    candidates = sorted(parent_dir.glob(f"{subject_id}_bd_*.*"))
    candidates = [c for c in candidates if c.suffix.lower() in (".jpg", ".jpeg", ".png")]
    return candidates[0] if candidates else None


class SiameseDelta:
    """Logistic regression on ``feature(query) - feature(bd_reference)``."""

    def __init__(self) -> None:
        self._pipeline: Optional[Pipeline] = None
        self._fallback_proba: float = 0.5

    def fit(
        self, train: IntoxicationCorpus, *, threshold: float
    ) -> "SiameseDelta":
        labels = train.binary_labels(threshold=threshold)

        # Per-subject reference photo and feature, computed once.
        subject_bd_features: Dict[str, np.ndarray] = {}
        subject_bd_paths: Dict[str, Path] = {}
        for photo, label in zip(train, labels):
            if label != 0 or photo.bac is None or photo.bac > 0.0:
                continue
            # ``bd`` photos have bac == 0.0; among those, keep the first
            # one we see per subject as the reference.
            if photo.subject_id in subject_bd_features:
                continue
            feature = _extract_for_path(photo.path)
            if feature is None:
                continue
            subject_bd_features[photo.subject_id] = feature
            subject_bd_paths[photo.subject_id] = photo.path

        X: List[np.ndarray] = []
        y: List[int] = []
        for photo, label in zip(train, labels):
            bd_feature = subject_bd_features.get(photo.subject_id)
            if bd_feature is None:
                continue
            # Skip the reference photo itself \u2014 trivially zero delta.
            if photo.path == subject_bd_paths.get(photo.subject_id):
                continue
            query_feature = _extract_for_path(photo.path)
            if query_feature is None:
                continue
            X.append(query_feature - bd_feature)
            y.append(label)

        if not X:
            raise RuntimeError("no usable training pairs for SiameseDelta")

        self._fallback_proba = sum(y) / len(y)
        self._pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
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
            raise RuntimeError("SiameseDelta must be fit before predicting")

        if photo.path is None:
            return self._fallback_proba

        bd_path = _find_bd_reference(photo.path.parent, photo.subject_id)
        if bd_path is None:
            return self._fallback_proba

        bd_feature = _extract_for_path(bd_path)
        query_feature = _extract_for_path(photo.path)
        if bd_feature is None or query_feature is None:
            return self._fallback_proba

        delta = query_feature - bd_feature
        return float(self._pipeline.predict_proba(delta.reshape(1, -1))[0, 1])

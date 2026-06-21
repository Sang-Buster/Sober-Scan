"""``CalibratedBaseline``: per-fold Platt-scaling wrapper around any baseline.

For a screening-style use case you want a model whose predicted
probabilities mean something \u2014 a score of 0.7 should correspond to a
~70% empirical chance of being in the positive class. Raw scores from
the existing baselines (a Platt-style logistic regression with
``class_weight="balanced"``; a neural network with a sigmoid head; a
constant prevalence) are *not* calibrated in that sense.

This wrapper does *per-fold* calibration so calibration data never
leaks across LOSO folds:

1. Within each LOSO train fold, randomly hold out ``val_fraction`` of
   the photos (stratified by label) as a *calibration set*.
2. Fit the underlying baseline on the remaining 1 \u2212 ``val_fraction``.
3. Score the calibration set with the fitted baseline.
4. Fit a single-feature logistic regression on
   ``(raw_score \u2192 label)`` over the calibration set \u2014 Platt scaling.
5. At ``predict_proba`` time, apply the calibration mapping to the
   underlying baseline's raw score.

This trades 20% of each LOSO training fold for calibrated probabilities.
For our 92-photo / 11-subject dataset that's a real cost: per-fold
training sees ~64 photos instead of ~80. We accept that cost when
``--calibrate`` is set; without it, no calibration data is taken.
"""

from typing import Callable, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sober_scan.corpus import IntoxicationCorpus, Photo

_Factory = Callable[[], "object"]


class CalibratedBaseline:
    """Sigmoid (Platt) calibration wrapper around any baseline factory."""

    def __init__(
        self,
        base_factory: _Factory,
        *,
        val_fraction: float = 0.2,
        seed: int = 0,
    ) -> None:
        self._base_factory = base_factory
        self._val_fraction = val_fraction
        self._seed = seed
        self._base: Optional[object] = None
        self._calibrator: Optional[LogisticRegression] = None
        self._fallback_proba: float = 0.5

    def fit(
        self, train: IntoxicationCorpus, *, threshold: float
    ) -> "CalibratedBaseline":
        photos: List[Photo] = list(train)
        labels = train.binary_labels(threshold=threshold)

        # Stratified split so both fit and cal subsets have both classes.
        try:
            fit_idx, cal_idx = train_test_split(
                list(range(len(photos))),
                test_size=self._val_fraction,
                stratify=labels,
                random_state=self._seed,
            )
        except ValueError:
            # Stratification fails when one class has too few samples;
            # fall back to a non-stratified split.
            fit_idx, cal_idx = train_test_split(
                list(range(len(photos))),
                test_size=self._val_fraction,
                random_state=self._seed,
            )

        fit_corpus = IntoxicationCorpus(photos=tuple(photos[i] for i in fit_idx))
        cal_corpus = IntoxicationCorpus(photos=tuple(photos[i] for i in cal_idx))
        cal_labels = np.array([labels[i] for i in cal_idx], dtype=int)

        # Train base on fit subset.
        self._base = self._base_factory()
        self._base.fit(fit_corpus, threshold=threshold)

        # Score cal subset with the base.
        cal_scores = np.array(
            [self._base.predict_proba(photo) for photo in cal_corpus]
        )

        self._fallback_proba = (
            float(cal_labels.mean()) if len(cal_labels) > 0 else 0.5
        )

        if len(np.unique(cal_labels)) < 2:
            # Single-class calibration set: can't fit a Platt scaler.
            # Fall through to identity; predict_proba returns raw score.
            self._calibrator = None
        else:
            # Single-feature LR \u2014 Platt scaling. Default L2 C=1.0 is mild
            # regularisation to prevent overfit on small cal sets.
            self._calibrator = LogisticRegression(random_state=self._seed)
            self._calibrator.fit(cal_scores.reshape(-1, 1), cal_labels)

        return self

    def predict_proba(self, photo: Photo) -> float:
        if self._base is None:
            raise RuntimeError("CalibratedBaseline must be fit before predicting")
        raw = self._base.predict_proba(photo)
        if self._calibrator is None:
            return float(raw)
        calibrated = self._calibrator.predict_proba(
            np.array([[raw]], dtype=float)
        )[0, 1]
        return float(calibrated)

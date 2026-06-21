"""``MajorityClassBaseline``: predict the training prevalence, always.

This is the sanity floor: any classifier that can't beat it is providing
no information beyond the class distribution. Useful especially under
LOSO with imbalanced single-class folds, where the prevalence itself
already gives 60\u201370% pooled accuracy.
"""

from typing import Optional

from sober_scan.corpus import IntoxicationCorpus, Photo


class MajorityClassBaseline:
    """Predicts a constant probability equal to the training positive rate."""

    def __init__(self) -> None:
        self._positive_rate: Optional[float] = None

    def fit(
        self, train: IntoxicationCorpus, *, threshold: float
    ) -> "MajorityClassBaseline":
        labels = train.binary_labels(threshold=threshold)
        self._positive_rate = sum(labels) / len(labels) if labels else 0.0
        return self

    def predict_proba(self, photo: Photo) -> float:
        if self._positive_rate is None:
            raise RuntimeError("MajorityClassBaseline must be fit before predicting")
        return self._positive_rate

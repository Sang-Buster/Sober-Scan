"""End-to-end LOSO evaluation runner.

Ties together corpus, splitter, baseline, and metrics into a single
function. Anything resembling \u201cresult bookkeeping\u201d (per-fold prediction
arrays, threshold handling, score-to-decision conversion) lives here,
not on the caller.
"""

from typing import Callable, Protocol

import numpy as np

from sober_scan.corpus import IntoxicationCorpus, Photo
from sober_scan.evaluation.loso import loso_splits
from sober_scan.evaluation.metrics import FoldResult, LOSOReport, aggregate


class _Baseline(Protocol):
    def fit(
        self, train: IntoxicationCorpus, *, threshold: float
    ) -> "_Baseline": ...

    def predict_proba(self, photo: Photo) -> float: ...


def evaluate_baseline(
    corpus: IntoxicationCorpus,
    baseline_factory: Callable[[], _Baseline],
    *,
    threshold: float,
    decision_cutoff: float = 0.5,
) -> LOSOReport:
    """Run LOSO on ``corpus`` using a fresh ``baseline`` per fold.

    ``baseline_factory`` is called once per fold to get a fresh, unfit
    instance \u2014 sharing state across folds would silently leak test
    photos into training.
    """
    fold_results = []
    for fold in loso_splits(corpus):
        baseline = baseline_factory().fit(fold.train, threshold=threshold)

        y_true = np.array(fold.test.binary_labels(threshold=threshold), dtype=int)
        y_score = np.array(
            [baseline.predict_proba(p) for p in fold.test], dtype=float
        )
        y_pred = (y_score >= decision_cutoff).astype(int)

        fold_results.append(
            FoldResult(
                held_out_subject=fold.held_out_subject,
                y_true=y_true,
                y_pred=y_pred,
                y_score=y_score,
            )
        )

    return aggregate(fold_results)

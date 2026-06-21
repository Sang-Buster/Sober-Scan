"""Aggregate cross-validation predictions into a single honest report.

``FoldResult`` captures the predictions for a single LOSO fold.
``aggregate`` pools them into a ``LOSOReport`` with both per-fold and
pooled metrics. Pooled metrics use *concatenated* per-fold predictions
rather than averaging fold-level metrics: averaging hides single-class
folds and over-weights small folds. Pooling treats every example
equally.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)


@dataclass(frozen=True)
class FoldResult:
    """Predictions and labels from a single LOSO fold.

    ``y_score`` holds the predicted probability of the positive (drunk)
    class; ``y_pred`` is the discrete 0/1 decision at the chosen threshold.
    """

    held_out_subject: str
    y_true: np.ndarray
    y_pred: np.ndarray
    y_score: np.ndarray


@dataclass(frozen=True)
class LOSOReport:
    """Summary of a complete LOSO run."""

    per_fold: Tuple[FoldResult, ...]
    pooled_accuracy: float
    pooled_balanced_accuracy: float
    pooled_auc: Optional[float]
    pooled_brier: float


def aggregate(folds: Sequence[FoldResult]) -> LOSOReport:
    """Pool per-fold predictions and compute the headline metrics.

    Pooled AUC is ``None`` if the pooled labels contain only one class
    (rare in practice with LOSO, but possible with single-subject tests).
    Pooled Brier (mean squared error between predicted probability and
    true label) is always well-defined.
    """
    y_true = np.concatenate([f.y_true for f in folds])
    y_pred = np.concatenate([f.y_pred for f in folds])
    y_score = np.concatenate([f.y_score for f in folds])

    auc: Optional[float]
    if len(np.unique(y_true)) < 2:
        auc = None
    else:
        auc = float(roc_auc_score(y_true, y_score))

    return LOSOReport(
        per_fold=tuple(folds),
        pooled_accuracy=float(accuracy_score(y_true, y_pred)),
        pooled_balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        pooled_auc=auc,
        pooled_brier=float(brier_score_loss(y_true, y_score)),
    )

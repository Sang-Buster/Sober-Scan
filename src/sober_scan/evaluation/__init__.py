"""Cross-subject evaluation primitives.

Public surface:

- ``loso_splits(corpus)`` \u2014 yield one ``LOSOFold`` per subject
- ``LOSOFold`` \u2014 held-out subject + (train, test) sub-corpora
- ``FoldResult`` \u2014 predictions for a single fold
- ``LOSOReport`` \u2014 pooled metrics across folds
- ``aggregate(folds)`` \u2014 turn ``FoldResult``s into a ``LOSOReport``
"""

from sober_scan.evaluation.loso import LOSOFold, loso_splits
from sober_scan.evaluation.metrics import FoldResult, LOSOReport, aggregate
from sober_scan.evaluation.runner import evaluate_baseline

__all__ = [
    "FoldResult",
    "LOSOFold",
    "LOSOReport",
    "aggregate",
    "evaluate_baseline",
    "loso_splits",
]

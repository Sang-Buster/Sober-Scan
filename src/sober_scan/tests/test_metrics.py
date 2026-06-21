"""Tests for LOSO metric aggregation."""

import numpy as np

from sober_scan.evaluation.metrics import FoldResult, aggregate


def _fold(subject: str, y_true: list, y_pred: list, y_score: list) -> FoldResult:
    return FoldResult(
        held_out_subject=subject,
        y_true=np.array(y_true, dtype=int),
        y_pred=np.array(y_pred, dtype=int),
        y_score=np.array(y_score, dtype=float),
    )


def test_pooled_accuracy_concatenates_predictions_across_folds() -> None:
    folds = [
        _fold("P1", y_true=[0, 0], y_pred=[0, 1], y_score=[0.1, 0.7]),  # 1/2 correct
        _fold("P2", y_true=[1, 1, 1], y_pred=[1, 1, 0], y_score=[0.8, 0.9, 0.3]),  # 2/3 correct
    ]

    report = aggregate(folds)

    assert report.pooled_accuracy == 3 / 5


def test_pooled_auc_uses_concatenated_scores() -> None:
    # Perfect ranking: sober scores < drunk scores
    folds = [
        _fold("P1", y_true=[0, 0], y_pred=[0, 0], y_score=[0.05, 0.10]),
        _fold("P2", y_true=[1, 1], y_pred=[1, 1], y_score=[0.90, 0.95]),
    ]

    report = aggregate(folds)

    assert report.pooled_auc == 1.0


def test_balanced_accuracy_averages_per_class_recall() -> None:
    # Imbalanced: 8 sober, 2 drunk.
    # Always-predict-sober: 100% recall on sober, 0% recall on drunk.
    # Balanced accuracy = (1.0 + 0.0) / 2 = 0.5; plain accuracy = 0.8.
    folds = [
        _fold(
            "P1",
            y_true=[0] * 8 + [1] * 2,
            y_pred=[0] * 10,
            y_score=[0.1] * 10,
        ),
    ]

    report = aggregate(folds)

    assert report.pooled_accuracy == 0.8
    assert report.pooled_balanced_accuracy == 0.5


def test_pooled_brier_measures_probability_squared_error() -> None:
    # Perfect probabilities -> Brier = 0.
    perfect = [
        _fold("P1", y_true=[0, 1], y_pred=[0, 1], y_score=[0.0, 1.0]),
    ]
    assert aggregate(perfect).pooled_brier == 0.0

    # Calibrated 0.5 for everything on a balanced set -> Brier = 0.25.
    halfway = [
        _fold("P1", y_true=[0, 1, 0, 1], y_pred=[0, 1, 0, 1], y_score=[0.5, 0.5, 0.5, 0.5]),
    ]
    assert aggregate(halfway).pooled_brier == 0.25

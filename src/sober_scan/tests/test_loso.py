"""Tests for leave-one-subject-out cross-validation splitting."""

from pathlib import Path

from sober_scan.corpus import IntoxicationCorpus
from sober_scan.evaluation.loso import loso_splits


def _write_empty(folder: Path, *names: str) -> None:
    for name in names:
        (folder / name).touch()


def test_yields_one_fold_per_subject(tmp_path: Path) -> None:
    _write_empty(
        tmp_path,
        "P1_bd_1.jpg",
        "P2_bd_1.jpg",
        "P3_bd_1.jpg",
    )
    corpus = IntoxicationCorpus.from_folder(tmp_path).with_known_bac()

    folds = list(loso_splits(corpus))

    assert len(folds) == 3


def test_fold_order_is_deterministic(tmp_path: Path) -> None:
    _write_empty(
        tmp_path,
        "P11_bd_1.jpg",
        "P2_bd_1.jpg",
        "P1_bd_1.jpg",
        "P3_bd_1.jpg",
    )
    corpus = IntoxicationCorpus.from_folder(tmp_path).with_known_bac()

    held_out_order = [fold.held_out_subject for fold in loso_splits(corpus)]
    # Numeric, not lexicographic, would be more natural but harder to
    # implement; we accept lexicographic order so long as it's stable.
    assert held_out_order == sorted(held_out_order)


def test_held_out_subject_does_not_appear_in_train(tmp_path: Path) -> None:
    _write_empty(
        tmp_path,
        "P1_bd_1.jpg",
        "P1_ad_080_1.jpg",
        "P2_bd_1.jpg",
        "P2_ad_120_1.jpg",
        "P3_bd_1.jpg",
    )
    corpus = IntoxicationCorpus.from_folder(tmp_path).with_known_bac()

    for fold in loso_splits(corpus):
        train_subjects = {p.subject_id for p in fold.train}
        test_subjects = {p.subject_id for p in fold.test}
        assert fold.held_out_subject not in train_subjects
        assert test_subjects == {fold.held_out_subject}
        assert len(fold.train) + len(fold.test) == len(corpus)

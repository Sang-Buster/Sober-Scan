"""Tests for ``MajorityClassBaseline``.

The majority-class baseline is the sanity floor every other model has
to beat. It predicts a constant probability equal to the training-set
prevalence of the positive class.
"""

from pathlib import Path

from sober_scan.corpus import IntoxicationCorpus, Photo
from sober_scan.models.baselines import MajorityClassBaseline


def _write_empty(folder: Path, *names: str) -> None:
    for name in names:
        (folder / name).touch()


def test_predicts_training_set_positive_prevalence(tmp_path: Path) -> None:
    # Training set: 1 sober (bd) + 9 drunk (ad at BAC 0.10) -> prevalence = 0.9
    _write_empty(
        tmp_path,
        "P1_bd_1.jpg",
        *(f"P{i}_ad_100_1.jpg" for i in range(2, 11)),
    )
    train = IntoxicationCorpus.from_folder(tmp_path).with_known_bac()

    baseline = MajorityClassBaseline().fit(train, threshold=0.05)

    held_out_photo = Photo(subject_id="P99", bac=0.0)
    assert baseline.predict_proba(held_out_photo) == 0.9

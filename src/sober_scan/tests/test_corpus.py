"""Tests for ``IntoxicationCorpus`` \u2014 the labeled-photo collection."""

from pathlib import Path

from sober_scan.corpus import IntoxicationCorpus


def _write_empty(folder: Path, *names: str) -> None:
    for name in names:
        (folder / name).touch()


def test_from_folder_loads_parseable_filenames(tmp_path: Path) -> None:
    _write_empty(
        tmp_path,
        "P1_bd_1.jpg",
        "P1_ad_080_1.jpg",
        "P2_bd_1.jpg",
        "hidethepainharold.jpg",  # non-conforming, should be ignored
    )

    corpus = IntoxicationCorpus.from_folder(tmp_path)

    assert len(corpus) == 3


def test_corpus_reports_distinct_subjects(tmp_path: Path) -> None:
    _write_empty(
        tmp_path,
        "P1_bd_1.jpg",
        "P1_ad_080_1.jpg",
        "P2_bd_1.jpg",
        "P11_ad_120_1.jpg",
    )

    corpus = IntoxicationCorpus.from_folder(tmp_path)

    assert corpus.subjects == frozenset({"P1", "P2", "P11"})


def test_with_known_bac_drops_unlabeled_photos(tmp_path: Path) -> None:
    _write_empty(
        tmp_path,
        "P1_bd_1.jpg",  # bac = 0.0  (known)
        "P1_ad_080_1.jpg",  # bac = 0.080 (known)
        "P7_ad_1533_1.jpg",  # bac = None (timestamp)
        "P6_ad_1.jpg",  # bac = None (no bac field)
    )

    labeled = IntoxicationCorpus.from_folder(tmp_path).with_known_bac()

    assert len(labeled) == 2
    assert all(photo.bac is not None for photo in labeled)


def test_binary_labels_apply_threshold(tmp_path: Path) -> None:
    # Filenames load in lexicographic order, so order is:
    #   ad_046 (bac 0.046, label 0)
    #   ad_080 (bac 0.080, label 1)
    #   ad_120 (bac 0.120, label 1)
    #   bd     (bac 0.000, label 0)
    _write_empty(
        tmp_path,
        "P1_bd_1.jpg",
        "P1_ad_046_1.jpg",
        "P1_ad_080_1.jpg",
        "P1_ad_120_1.jpg",
    )

    corpus = IntoxicationCorpus.from_folder(tmp_path).with_known_bac()

    assert corpus.binary_labels(threshold=0.05) == [0, 1, 1, 0]


def test_binary_labels_rejects_unknown_bac(tmp_path: Path) -> None:
    import pytest

    _write_empty(
        tmp_path,
        "P1_bd_1.jpg",
        "P7_ad_1533_1.jpg",  # bac = None
    )

    corpus = IntoxicationCorpus.from_folder(tmp_path)

    with pytest.raises(ValueError, match="unknown BAC"):
        corpus.binary_labels(threshold=0.05)

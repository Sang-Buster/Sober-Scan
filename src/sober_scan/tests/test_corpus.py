"""Tests for ``IntoxicationCorpus`` \u2014 the labeled-photo collection."""

from pathlib import Path

import pytest

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
    _write_empty(
        tmp_path,
        "P1_bd_1.jpg",
        "P7_ad_1533_1.jpg",  # bac = None
    )

    corpus = IntoxicationCorpus.from_folder(tmp_path)

    with pytest.raises(ValueError, match="unknown BAC"):
        corpus.binary_labels(threshold=0.05)


def test_backfill_resolves_timestamp_filenames_via_session_notes(tmp_path: Path) -> None:
    """Filename timestamps are resolved to the nearest BAC measurement."""
    _write_empty(
        tmp_path,
        "P7_bd_1.jpg",  # already labeled (bac = 0.0)
        "P7_ad_1533_1.jpg",  # 15:33 \u2192 nearest 15:35 (0.089)
        "P7_ad_1550_1.jpg",  # 15:50 \u2192 nearest 15:46 (0.046)
    )
    (tmp_path / "P7.txt").write_text(
        "Consumed Alcohol\n3:35 PM\nBlood Alcohol Level: 0.089% ABV\n\n"
        "Post-Quiz Alcohol Level Measured\n3:46 PM\nBlood Alcohol Level: 0.046% ABV\n"
    )

    backfilled = IntoxicationCorpus.from_folder(tmp_path).with_session_notes_backfill(
        notes_dir=tmp_path
    )

    by_name = {photo.path.name: photo for photo in backfilled}
    assert by_name["P7_bd_1.jpg"].bac == 0.0  # untouched
    assert by_name["P7_ad_1533_1.jpg"].bac == 0.089
    assert by_name["P7_ad_1550_1.jpg"].bac == 0.046


def test_backfill_uses_last_measurement_for_timestamp_less_ad_photos(tmp_path: Path) -> None:
    """``Px_ad_n.jpg`` with no inner timestamp gets the last (most recent) BAC.

    This is a documented heuristic: photos without an explicit time
    were taken at the *end* of the session in the data we have
    (per P6's session notes, the ``Photo #2`` event follows all BAC
    measurements).
    """
    _write_empty(
        tmp_path,
        "P6_ad_1.jpg",
        "P6_ad_2.jpg",
        "P6_ad_3.jpg",
    )
    (tmp_path / "P6.txt").write_text(
        "1543 BAC 1 0.087\n"
        "1548 BAC 2 0.036\n"
        "1558 BAC 3 0.11\n"
        "1612 Photo #2\n"
    )

    backfilled = IntoxicationCorpus.from_folder(tmp_path).with_session_notes_backfill(
        notes_dir=tmp_path
    )

    for photo in backfilled:
        assert photo.bac == 0.11


def test_backfill_leaves_unresolvable_photos_with_bac_none(tmp_path: Path) -> None:
    """If there's no ``.txt`` (or no measurements in it), bac stays None."""
    _write_empty(
        tmp_path,
        "P7_ad_1533_1.jpg",
    )

    backfilled = IntoxicationCorpus.from_folder(tmp_path).with_session_notes_backfill(
        notes_dir=tmp_path
    )

    assert backfilled.photos[0].bac is None

"""Tests for the testing_data filename parser.

The data/testing_data/ directory uses a controlled filename convention:

    P{subject}_bd_{n}.{ext}        # before-drinking baseline (BAC = 0)
    P{subject}_ad_{BBB}_{n}.{ext}  # after-drinking at BAC = BBB/1000
    P{subject}_ad_{n}.{ext}        # after-drinking, BAC unknown

We treat 4-digit "BAC" values as time-of-day timestamps and refuse to
fabricate a label; the caller can backfill from session notes later.
"""

from pathlib import Path

from sober_scan.corpus.parser import parse_testing_data_filename


def test_parses_before_drinking_filename() -> None:
    photo = parse_testing_data_filename(Path("P1_bd_1.jpg"))
    assert photo is not None
    assert photo.subject_id == "P1"
    assert photo.bac == 0.0


def test_parses_after_drinking_filename_with_bac() -> None:
    photo = parse_testing_data_filename(Path("P11_ad_102_3.jpg"))
    assert photo is not None
    assert photo.subject_id == "P11"
    assert photo.bac == 0.102


def test_four_digit_ad_field_is_timestamp_not_bac() -> None:
    photo = parse_testing_data_filename(Path("P7_ad_1533_1.jpg"))
    assert photo is not None
    assert photo.subject_id == "P7"
    assert photo.bac is None


def test_after_drinking_filename_without_bac_field() -> None:
    photo = parse_testing_data_filename(Path("P6_ad_1.jpg"))
    assert photo is not None
    assert photo.subject_id == "P6"
    assert photo.bac is None


def test_returns_none_for_non_conforming_filename() -> None:
    assert parse_testing_data_filename(Path("hidethepainharold.jpg")) is None
    assert parse_testing_data_filename(Path("IMG_5455.JPG")) is None
    assert parse_testing_data_filename(Path("P1.txt")) is None
    assert parse_testing_data_filename(Path("P1_bd.jpg")) is None  # missing index

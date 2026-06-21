"""Tests for ``parse_session_notes``: extract (time, BAC) pairs from Px.txt.

The session notes appear in three different freeform styles across
participants. We only need to parse the three subjects (P6, P7, P9)
whose photo filenames carry timestamps instead of explicit BAC values,
so the tests use real snippets from those three files.
"""

from sober_scan.corpus.session_notes import (
    BacMeasurement,
    nearest_bac,
    parse_session_notes,
)


def test_parses_compact_24h_format() -> None:
    """P6-style notes: each measurement on one line as ``HHMM BAC N value``."""
    text = """1518 Session Start
1518 Photo 1
--------------------------------------------------------------
Drinking start
1543 BAC 1 0.087
1548 BAC 2 0.036
1558 BAC 3 0.11
--------------------------------------------------------------
1612 Photo #2
"""

    measurements = parse_session_notes(text)

    assert measurements == [
        BacMeasurement(time_minutes=15 * 60 + 43, bac=0.087),
        BacMeasurement(time_minutes=15 * 60 + 48, bac=0.036),
        BacMeasurement(time_minutes=15 * 60 + 58, bac=0.11),
    ]


def test_parses_table_12h_bare_abv() -> None:
    """P9-style notes: same three-line layout, but no 'Blood Alcohol Level:' prefix."""
    text = """Alcohol Level Measured
11:14 AM
0.026% ABV

Alcohol Level Measured
11:26 AM
0.12% ABV (after 2nd bottle)

Alcohol Level Measured
11:33 AM
0.186% ABV (after 3rd drink)
"""

    measurements = parse_session_notes(text)

    assert measurements == [
        BacMeasurement(time_minutes=11 * 60 + 14, bac=0.026),
        BacMeasurement(time_minutes=11 * 60 + 26, bac=0.12),
        BacMeasurement(time_minutes=11 * 60 + 33, bac=0.186),
    ]


def test_nearest_bac_finds_smallest_time_delta() -> None:
    measurements = [
        BacMeasurement(time_minutes=15 * 60 + 35, bac=0.089),
        BacMeasurement(time_minutes=15 * 60 + 46, bac=0.046),
    ]

    # 15:33 \u2192 closest is 15:35 (\u0394 = 2 min) vs 15:46 (\u0394 = 13 min).
    assert nearest_bac(target_minutes=15 * 60 + 33, measurements=measurements) == measurements[0]
    # 15:50 \u2192 closest is 15:46 (\u0394 = 4 min) vs 15:35 (\u0394 = 15 min).
    assert nearest_bac(target_minutes=15 * 60 + 50, measurements=measurements) == measurements[1]


def test_nearest_bac_returns_none_for_empty_measurements() -> None:
    assert nearest_bac(target_minutes=600, measurements=[]) is None


def test_parses_table_12h_with_blood_alcohol_level_prefix() -> None:
    """P7-style notes: table layout where activity / time / details are three lines.

    The BAC line is preceded by the 12h timestamp on the previous line.
    """
    text = """Consumed Alcohol
3:35 PM
Blood Alcohol Level: 0.089% ABV

Picture Taken (After Alcohol)
3:36 PM
Post-alcohol documentation

Post-Quiz Alcohol Level Measured
3:46 PM
Blood Alcohol Level: 0.046% ABV
"""

    measurements = parse_session_notes(text)

    assert measurements == [
        BacMeasurement(time_minutes=15 * 60 + 35, bac=0.089),
        BacMeasurement(time_minutes=15 * 60 + 46, bac=0.046),
    ]

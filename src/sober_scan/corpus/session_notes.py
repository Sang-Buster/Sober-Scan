"""Parse ``data/testing_data/Px.txt`` session notes for BAC backfill.

Each ``Px.txt`` records the chronology of one experimental session. We
only care about extracting the timed BAC measurements so we can resolve
the BAC of photos whose filename encodes a time-of-day stamp rather
than an explicit BAC value.

Three input dialects exist across the participants whose .txt files we
actually need to read (P6, P7, P9):

- **Compact 24h:** ``1543 BAC 1 0.087`` \u2014 time and BAC on one line
- **Table 12h, prefixed:** ``Consumed Alcohol`` / ``3:35 PM`` /
  ``Blood Alcohol Level: 0.089% ABV`` \u2014 three-line block
- **Table 12h, bare:** ``Alcohol Level Measured`` / ``11:14 AM`` /
  ``0.026% ABV`` \u2014 three-line block, no "Blood Alcohol Level" prefix

Only these three formats are supported; the parser is intentionally
narrow because the dataset is fixed.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class BacMeasurement:
    """A single timed BAC reading from a session.

    Attributes:
        time_minutes: Minutes since midnight (so 15:43 == ``15*60 + 43``).
        bac: Blood alcohol concentration as a mass fraction (e.g. ``0.087``).
    """

    time_minutes: int
    bac: float


_COMPACT_24H_LINE = re.compile(
    r"^(?P<hhmm>\d{4})\s+BAC\s+\d+\s+(?P<bac>\d+\.\d+)\s*$",
    re.MULTILINE,
)

# Matches either "Blood Alcohol Level: 0.089% ABV" or bare "0.026% ABV"
# (P7 vs P9 styles); both end in "% ABV".
_BAC_LINE = re.compile(r"(?P<bac>\d+\.\d+)\s*%\s*ABV", re.IGNORECASE)

# Matches "3:35 PM" / "11:14 AM" 12-hour clock entries.
_TIME_12H_LINE = re.compile(
    r"^\s*(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*(?P<ampm>AM|PM)\s*$",
    re.IGNORECASE,
)


def _to_minutes_12h(hour: int, minute: int, ampm: str) -> int:
    """Convert 12-hour clock to minutes-since-midnight."""
    if ampm.upper() == "PM" and hour != 12:
        hour += 12
    elif ampm.upper() == "AM" and hour == 12:
        hour = 0
    return hour * 60 + minute


def nearest_bac(
    *, target_minutes: int, measurements: Sequence[BacMeasurement]
) -> Optional[BacMeasurement]:
    """Return the measurement closest in time to ``target_minutes``.

    Ties (equal time delta) resolve to whichever comes first in
    ``measurements`` because that's what ``min`` does in CPython.
    """
    if not measurements:
        return None
    return min(measurements, key=lambda m: abs(m.time_minutes - target_minutes))


def parse_session_notes(text: str) -> List[BacMeasurement]:
    """Extract all timed BAC measurements from a session-notes file."""
    measurements: List[BacMeasurement] = []

    # Compact 24h format \u2014 each measurement is one self-contained line.
    for match in _COMPACT_24H_LINE.finditer(text):
        hhmm = match.group("hhmm")
        hours = int(hhmm[:2])
        minutes = int(hhmm[2:])
        measurements.append(
            BacMeasurement(
                time_minutes=hours * 60 + minutes,
                bac=float(match.group("bac")),
            )
        )

    # Table 12h format \u2014 BAC line preceded by a 12h timestamp line.
    lines = text.splitlines()
    for i, line in enumerate(lines):
        bac_match = _BAC_LINE.search(line)
        if not bac_match:
            continue
        # Walk back to find the most recent 12h time line.
        for j in range(i - 1, max(-1, i - 4), -1):
            time_match = _TIME_12H_LINE.match(lines[j])
            if time_match:
                measurements.append(
                    BacMeasurement(
                        time_minutes=_to_minutes_12h(
                            int(time_match.group("hour")),
                            int(time_match.group("minute")),
                            time_match.group("ampm"),
                        ),
                        bac=float(bac_match.group("bac")),
                    )
                )
                break

    return measurements

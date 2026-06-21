"""Parse ``data/testing_data/`` filenames into labeled ``Photo`` objects.

Filename convention (controlled study at ERAU):

    P{subject}_bd_{n}.{ext}        before-drinking baseline (BAC = 0)
    P{subject}_ad_{BBB}_{n}.{ext}  after-drinking at BAC = BBB / 1000
    P{subject}_ad_{n}.{ext}        after-drinking, BAC unknown

We treat any 4-digit "BAC" field as a time-of-day timestamp and refuse to
fabricate a label; the caller can backfill via session notes later.
"""

import re
from pathlib import Path
from typing import Optional, Union

from sober_scan.corpus.photo import Photo

_BD_PATTERN = re.compile(r"^(P\d+)_bd_\d+\.\w+$", re.IGNORECASE)
_AD_BAC_PATTERN = re.compile(r"^(P\d+)_ad_(\d{3})_\d+\.\w+$", re.IGNORECASE)
_AD_TIMESTAMP_PATTERN = re.compile(r"^(P\d+)_ad_\d{4}_\d+\.\w+$", re.IGNORECASE)
_AD_NOBAC_PATTERN = re.compile(r"^(P\d+)_ad_\d+\.\w+$", re.IGNORECASE)


def parse_testing_data_filename(path: Union[Path, str]) -> Optional[Photo]:
    """Parse a single testing_data filename into a ``Photo``.

    Returns ``None`` when the filename does not match the convention.
    A returned ``Photo`` may still have ``bac=None`` when the filename
    matches the convention but does not encode a usable BAC label
    (e.g. the field is a time-of-day stamp, or BAC is simply absent).
    """
    name = Path(path).name

    match = _BD_PATTERN.match(name)
    if match:
        return Photo(subject_id=match.group(1), bac=0.0, path=Path(path))

    match = _AD_BAC_PATTERN.match(name)
    if match:
        return Photo(
            subject_id=match.group(1),
            bac=int(match.group(2)) / 1000.0,
            path=Path(path),
        )

    match = _AD_TIMESTAMP_PATTERN.match(name)
    if match:
        return Photo(subject_id=match.group(1), bac=None, path=Path(path))

    match = _AD_NOBAC_PATTERN.match(name)
    if match:
        return Photo(subject_id=match.group(1), bac=None, path=Path(path))

    return None

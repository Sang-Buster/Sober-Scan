"""The ``Photo`` value type: a single labeled image in a corpus."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Photo:
    """A single labeled facial image.

    Attributes:
        subject_id: A stable identifier for the person depicted (e.g. ``"P1"``).
            Two ``Photo`` objects with the same ``subject_id`` are guaranteed
            to be the same person.
        bac: Blood alcohol concentration in mass-fraction units
            (e.g. ``0.08`` for 0.08% BAC). ``None`` means the BAC label
            could not be derived from the source and the photo is
            unsuitable for supervised training/evaluation.
        path: Filesystem path to the image. May be missing when the photo
            is synthetic (e.g. in tests).
    """

    subject_id: str
    bac: Optional[float]
    path: Optional[Path] = None

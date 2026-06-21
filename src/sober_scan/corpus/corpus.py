"""The ``IntoxicationCorpus``: a labeled photo collection."""

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import FrozenSet, Iterator, List, Optional, Tuple

from sober_scan.corpus.parser import parse_testing_data_filename
from sober_scan.corpus.photo import Photo
from sober_scan.corpus.session_notes import nearest_bac, parse_session_notes

_FILENAME_TIMESTAMP_PATTERN = re.compile(
    r"^P\d+_ad_(?P<hhmm>\d{4})_\d+\.\w+$", re.IGNORECASE
)
_FILENAME_NO_BAC_PATTERN = re.compile(r"^P\d+_ad_\d+\.\w+$", re.IGNORECASE)


def _filename_timestamp_minutes(filename: str) -> Optional[int]:
    """Return the HHMM-encoded minutes-since-midnight, or None if absent."""
    match = _FILENAME_TIMESTAMP_PATTERN.match(filename)
    if not match:
        return None
    hhmm = match.group("hhmm")
    hours = int(hhmm[:2])
    minutes = int(hhmm[2:])
    # Reject impossible times (e.g. 2575 \u2014 a 4-digit number that isn't a time).
    if hours > 23 or minutes > 59:
        return None
    return hours * 60 + minutes

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class IntoxicationCorpus:
    """A collection of labeled facial photos, keyed by subject.

    Construction is via the ``from_folder`` factory rather than directly:
    it's the only way to make sure parsing rules are applied consistently.
    """

    photos: Tuple[Photo, ...]

    @classmethod
    def from_folder(cls, folder: Path) -> "IntoxicationCorpus":
        """Load every parseable image file directly inside ``folder``.

        Image files whose filename does not match the testing_data
        convention are silently ignored.
        """
        loaded = []
        for path in sorted(Path(folder).iterdir()):
            if path.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            photo = parse_testing_data_filename(path)
            if photo is not None:
                loaded.append(photo)
        return cls(photos=tuple(loaded))

    def __len__(self) -> int:
        return len(self.photos)

    def __iter__(self) -> Iterator[Photo]:
        return iter(self.photos)

    @property
    def subjects(self) -> FrozenSet[str]:
        """The set of distinct subject IDs in this corpus."""
        return frozenset(photo.subject_id for photo in self.photos)

    def with_known_bac(self) -> "IntoxicationCorpus":
        """Drop photos whose BAC could not be derived from the filename."""
        kept = tuple(photo for photo in self.photos if photo.bac is not None)
        return IntoxicationCorpus(photos=kept)

    def with_session_notes_backfill(
        self, *, notes_dir: Path
    ) -> "IntoxicationCorpus":
        """Resolve ``bac=None`` photos by reading the corresponding ``Px.txt``.

        For each photo with no usable filename BAC we read
        ``{notes_dir}/{subject_id}.txt`` and try to recover a BAC value:

        - If the filename encodes a 4-digit ``HHMM`` timestamp, we use
          the BAC measurement closest in time.
        - If the filename is timestamp-less (e.g. ``P6_ad_1.jpg``), we
          fall back to the *last* measurement in the session notes,
          on the heuristic that those photos were taken after all
          recorded measurements.

        Photos whose BAC is already known are passed through unchanged.
        Photos for which no matching ``.txt`` exists, or whose ``.txt``
        contains no measurements, retain ``bac=None``.
        """
        notes_cache: dict = {}

        def _measurements_for(subject_id: str):
            if subject_id in notes_cache:
                return notes_cache[subject_id]
            path = Path(notes_dir) / f"{subject_id}.txt"
            measurements = (
                parse_session_notes(path.read_text()) if path.exists() else []
            )
            notes_cache[subject_id] = measurements
            return measurements

        new_photos: List[Photo] = []
        for photo in self.photos:
            if photo.bac is not None:
                new_photos.append(photo)
                continue
            measurements = _measurements_for(photo.subject_id)
            if not measurements:
                new_photos.append(photo)
                continue

            filename = photo.path.name if photo.path is not None else ""
            target_minutes = _filename_timestamp_minutes(filename)
            if target_minutes is not None:
                resolved = nearest_bac(
                    target_minutes=target_minutes, measurements=measurements
                )
            elif _FILENAME_NO_BAC_PATTERN.match(filename):
                # Heuristic: the photos with no timestamp were taken after
                # all recorded measurements.
                resolved = measurements[-1]
            else:
                resolved = None

            if resolved is None:
                new_photos.append(photo)
            else:
                new_photos.append(replace(photo, bac=resolved.bac))

        return IntoxicationCorpus(photos=tuple(new_photos))

    def binary_labels(self, *, threshold: float) -> List[int]:
        """Return one ``0``/``1`` label per photo: ``1`` if ``bac >= threshold``.

        Raises ``ValueError`` if any photo has ``bac is None``: BAC labels
        must be resolved before classification; silent imputation here would
        be a data-quality hazard. Filter with ``with_known_bac()`` first.
        """
        labels = []
        for photo in self.photos:
            if photo.bac is None:
                raise ValueError(
                    f"corpus contains photo with unknown BAC: {photo.path}; "
                    "call `with_known_bac()` before labeling."
                )
            labels.append(1 if photo.bac >= threshold else 0)
        return labels

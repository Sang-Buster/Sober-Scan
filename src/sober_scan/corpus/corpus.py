"""The ``IntoxicationCorpus``: a labeled photo collection."""

from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet, Iterator, List, Tuple

from sober_scan.corpus.parser import parse_testing_data_filename
from sober_scan.corpus.photo import Photo

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

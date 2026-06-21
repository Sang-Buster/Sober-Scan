"""Leave-one-subject-out cross-validation.

Each fold holds out exactly one subject's photos for testing, and uses
all remaining subjects' photos for training. This is the only honest
split given the small per-subject sample size in ``data/testing_data``.
"""

from dataclasses import dataclass
from typing import Iterator

from sober_scan.corpus import IntoxicationCorpus


@dataclass(frozen=True)
class LOSOFold:
    """One leave-one-subject-out fold.

    Attributes:
        held_out_subject: ID of the subject reserved for evaluation.
        train: All photos from the other subjects.
        test: All photos from the held-out subject.
    """

    held_out_subject: str
    train: IntoxicationCorpus
    test: IntoxicationCorpus


def loso_splits(corpus: IntoxicationCorpus) -> Iterator[LOSOFold]:
    """Yield one ``LOSOFold`` per subject in the corpus.

    Subjects are iterated in lexicographic order so the iteration is
    deterministic across runs.
    """
    for subject in sorted(corpus.subjects):
        train = IntoxicationCorpus(
            photos=tuple(p for p in corpus if p.subject_id != subject)
        )
        test = IntoxicationCorpus(
            photos=tuple(p for p in corpus if p.subject_id == subject)
        )
        yield LOSOFold(held_out_subject=subject, train=train, test=test)

"""Subject-aware data model for Sober-Scan corpora.

Public surface (intentionally small):

- ``Photo`` — a single labeled image (subject, BAC, path)
- ``parse_testing_data_filename`` — parse a ``data/testing_data/`` filename
  into a ``Photo``

The internal split between filename parsing, corpus loading, and CV
splitting lives behind these names.
"""

from sober_scan.corpus.corpus import IntoxicationCorpus
from sober_scan.corpus.parser import parse_testing_data_filename
from sober_scan.corpus.photo import Photo

__all__ = ["IntoxicationCorpus", "Photo", "parse_testing_data_filename"]

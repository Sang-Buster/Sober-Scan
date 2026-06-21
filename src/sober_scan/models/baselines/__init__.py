"""Honest baselines for cross-subject intoxication detection.

Every more-sophisticated model has to beat these to earn its complexity.

Public surface:

- ``MajorityClassBaseline`` \u2014 predicts the training-set prevalence
"""

from sober_scan.models.baselines.handcrafted import HandcraftedFeaturesLR
from sober_scan.models.baselines.imagenet import ImageNetFrozenLR
from sober_scan.models.baselines.majority import MajorityClassBaseline

__all__ = [
    "HandcraftedFeaturesLR",
    "ImageNetFrozenLR",
    "MajorityClassBaseline",
]

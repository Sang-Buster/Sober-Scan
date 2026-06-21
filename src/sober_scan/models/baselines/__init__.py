"""Honest baselines for cross-subject intoxication detection.

Every more-sophisticated model has to beat these to earn its complexity.

Public surface:

- ``MajorityClassBaseline`` \u2014 predicts the training-set prevalence
"""

from sober_scan.models.baselines.handcrafted import HandcraftedFeaturesLR
from sober_scan.models.baselines.imagenet import ImageNetFrozenLR
from sober_scan.models.baselines.loso_cnn import LOSOTrainedCNN
from sober_scan.models.baselines.majority import MajorityClassBaseline
from sober_scan.models.baselines.siamese import SiameseDelta

__all__ = [
    "HandcraftedFeaturesLR",
    "ImageNetFrozenLR",
    "LOSOTrainedCNN",
    "MajorityClassBaseline",
    "SiameseDelta",
]

# Drowsiness is a rule, not a learned model

The four previous drowsiness models (`drowsiness_{svm,knn,nb,rf}.joblib`) were trained by `extract_features_from_folder` in the legacy training command, which computed each training label by applying `label = 1 if EAR < threshold else 0` and then included the same EAR value in the feature vector passed to the classifier. The classifier had to learn a step function on one of its inputs — recoverable trivially by any non-degenerate model — so the reported 100% accuracy was a mathematical tautology, not evidence of learning. We removed the four models from `MODEL_URLS` and made `sober-scan detect --type drowsiness` apply the EAR < 0.2 rule directly, with no model file involved.

## Consequences

- `sober-scan train --detection-type drowsiness` returns a deprecation error pointing at the rule.
- The `models/drowsiness_*.joblib` artifacts remain on disk (and in the LFS cache) but are no longer reachable through the CLI and should not be redistributed.
- We accept that this collapses the "drowsiness modeling" surface area to zero. The published EAR threshold (Soukupová & Čech, 2016) is the right rule for this task and the codebase carries no value-add beyond it.

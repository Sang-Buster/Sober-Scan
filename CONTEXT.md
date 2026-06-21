# Sober-Scan

A Python package and CLI for honest cross-subject intoxication detection
from facial images, with an EAR-rule fallback for drowsiness. The
domain language below is what the code uses; treat these as the
canonical terms in issues, docstrings, commit messages, and any
follow-on work.

## Language

### Subjects and photos

**Subject**:
A participant in the controlled study, identified by `P{n}` (e.g. `P1`,
`P11`). Two photos with the same subject ID are guaranteed to be the
same person.
_Avoid_: user, person, individual, participant (in code).

**Photo**:
A single labeled facial image, defined by its `(subject_id, bac, path)`
triple. The unit of input throughout the package.
_Avoid_: image, picture, sample (these are ambiguous with raw numpy
arrays or generic ML usage).

**bd photo / ad photo**:
The filename convention in `data/testing_data/`. `bd` = before-drinking
baseline (BAC = 0). `ad` = after-drinking, with the BAC either encoded
in the filename or recoverable from session notes.
_Avoid_: pre / post, baseline / post-drinking. Use the filename codes.

**Session notes**:
A `P{n}.txt` file alongside the photos that records timed BAC
measurements for that participant's session. Used to backfill `Photo.bac`
when the filename doesn't encode BAC directly.
_Avoid_: log, transcript, notes.

### Labels and thresholds

**BAC**:
Blood Alcohol Concentration as a mass fraction (e.g. `0.08` ≈ the US
driving limit). Always stored as a fraction, never as a percentage.
_Avoid_: ABV, alcohol level, blood-alcohol.

**Threshold**:
The BAC value above which a photo is labeled `drunk` for binary
classification. The project default is `0.05` (≈ EU driving limit).
_Avoid_: cutoff, boundary.

**Backfill**:
The process of resolving `Photo.bac is None` by parsing the subject's
session notes. Two strategies:

- `Px_ad_HHMM_n.jpg` (timestamped filename) → nearest BAC measurement
- `Px_ad_n.jpg` (no timestamp) → last recorded measurement (heuristic)
  _Avoid_: imputation, recovery.

### Evaluation

**Corpus**:
An immutable, subject-aware collection of `Photo`s. Constructed via
`IntoxicationCorpus.from_folder(...)` and transformed by chainable
methods (`with_known_bac()`, `with_session_notes_backfill(...)`).
_Avoid_: dataset (overloaded with PyTorch's `Dataset` and the
registered-data-management concept in `dataset_management.py`).

**LOSO**:
Leave-one-subject-out cross-validation. Each fold holds out every
photo of exactly one subject. The only evaluation methodology this
project uses; random splits would leak subject identity across train
and test.
_Avoid_: k-fold, cross-validation (be specific).

**Fold**:
One iteration of LOSO. A `LOSOFold` carries a `held_out_subject` plus
a `train` and `test` sub-corpus.
_Avoid_: split, partition.

**Baseline**:
A registered classifier under `models/baselines/`. Implements
`fit(train, *, threshold) -> self` and `predict_proba(photo) -> float`.
Each baseline is freshly instantiated per fold so no state leaks across
the LOSO loop.
_Avoid_: model (overloaded with PyTorch nn.Module), classifier
(less specific), estimator.

**Tier 1 / Tier 2**:
The category of a baseline by ambition.

- Tier 1: sanity-floor baselines (`MajorityClassBaseline`,
  `HandcraftedFeaturesLR`, `ImageNetFrozenLR`). Cheap to build.
- Tier 2: methodology-fixed sophisticated baselines (`LOSOTrainedCNN`,
  `SiameseDelta`). Built only after Tier 1 numbers are on paper.

A Tier 2 baseline that cannot beat the best Tier 1 result has not
earned its complexity.
_Avoid_: simple / complex, weak / strong.

### Features and detection

**EAR / MAR**:
Eye Aspect Ratio / Mouth Aspect Ratio. Geometric ratios computed from
dlib's 68-point landmarks. EAR is the canonical drowsiness signal in
the literature; MAR captures mouth-open / yawn state.
_Avoid_: spelling these out (the abbreviations are the literature
standard).

**Drowsiness rule**:
The EAR < 0.2 heuristic, applied directly in `commands/detect.py`.
Replaces the four previous trained "drowsiness models" (SVM, KNN, NB,
RF), which were learning a step function on their own input feature
and were therefore tautologies.
_Avoid_: drowsiness model, drowsiness classifier.

**Face crop**:
The detected face bounding box expanded by a 20% margin on each side.
The unit of input to every Tier 2 baseline that touches pixels (the
shipped CNN's training pipeline didn't crop and instead saw whole
photos including background — see ADR-0001).
_Avoid_: face region, face patch.

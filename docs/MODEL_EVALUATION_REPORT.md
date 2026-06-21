# Model Evaluation Report

**Generated:** June 21, 2026 (refactor v0.1.x WIP)
**Methodology:** Leave-one-subject-out (LOSO) cross-validation on `data/testing_data/`
**Threshold:** Binary classification, drunk iff BAC \u2265 0.05

> **This report supersedes all earlier evaluation numbers in this project.**
> Previous reports (the January 2026 "all four drowsiness models = 100% accuracy" / "intoxication CNN = 81% accuracy, AUC 0.92" headline) were produced under a leaky evaluation setup and do not reflect the models' actual cross-subject behavior. See the _Errata_ section for details on what changed.

---

## Headline numbers

| Model                                | Pooled accuracy | Pooled balanced accuracy | Pooled ROC-AUC |
| ------------------------------------ | --------------: | -----------------------: | -------------: |
| `MajorityClassBaseline`              |          60.76% |                   50.00% |    0.218\u00b9 |
| `HandcraftedFeaturesLR`              |      **62.03%** |               **61.32%** |     **0.5995** |
| `ImageNetFrozenLR`                   |          48.10% |                   46.44% |         0.5000 |
| Existing `intoxication_cnn.pt`\u00b2 |          62.03% |                   53.90% |         0.4832 |

\u00b9 The AUC of the constant-prediction baseline under LOSO is _not_ 0.5: the model emits the training-set prevalence, which varies by which subject is held out, and that variation happens to anti-correlate with the held-out subject's positive rate. The 0.218 is the structural signature of a non-informative predictor under leave-one-out splits, not a property of the model.

\u00b2 Evaluated by running inference on all 79 labeled photos in `data/testing_data/` (the CNN was trained on a separate `data/training/intoxication/` corpus and therefore was not retrained per fold). The previously published 81% / AUC 0.92 figures for this same checkpoint were on an internal random split with within-subject contamination; see _Errata_.

---

## What the numbers say

1. **The shipped intoxication CNN has no real cross-subject discriminative ability.** Honest AUC of 0.48 \u2014 statistically indistinguishable from the AUC of "predict sober always" \u2014 means that, given a photo of a new person the model has never seen, it cannot rank "drunk" above "sober" any better than a coin flip. The 62% accuracy is essentially the class prior leaking through (the model predicts sober for 70 of 79 photos, and 48 of 79 photos really are sober).

2. **The simplest handcrafted features (redness + EAR/MAR + landmark distances) modestly beat everything else** in balanced accuracy and AUC. This is the floor any future model has to clear, _not_ the existing CNN.

3. **Frozen ImageNet features underperform.** A 1280-d MobileNetV2 representation, frozen and fed to logistic regression, scores 48.10% \u2014 below majority class. The interpretation: with 11 subjects and ~7 photos each, the rich representation is discriminative for _face identity_, and identity does not generalize across LOSO folds. Identity-overfitting is exactly the failure mode that caused the previous 81% number.

4. **Per-fold variance is enormous.** With 11 subjects, fold sizes range from 3 to 11 photos, and 4 of the 11 subjects (P1, P2, P6, P9) have only sober photos in `testing_data`. Single-class folds give pseudo-perfect accuracy if the model predicts sober and pseudo-zero if it predicts drunk, contributing nothing to discrimination. The pooled metrics absorb this; the per-fold tables in the JSON dumps surface it.

---

## Reproducibility

```bash
sober-scan evaluate baseline majority      --data data/testing_data --threshold 0.05
sober-scan evaluate baseline handcrafted   --data data/testing_data --threshold 0.05
sober-scan evaluate baseline imagenet      --data data/testing_data --threshold 0.05
```

Each command runs the full LOSO loop and prints a metrics table plus a per-fold breakdown. Add `--output-json path.json` for the full per-fold prediction arrays.

The existing CNN is not yet exposed through `sober-scan evaluate` as a named baseline; that's tracked in the project's open work.

### Dataset profile (`data/testing_data/`)

- 92 image files; 79 with a usable BAC label, 13 with `bac=None` (timestamp- or index-only filenames, no BAC encoded)
- 11 subjects: P1, P2, P3, P4, P5, P6, P7, P8, P9, P11, P12 (note: P10 absent)
- 48 sober (BAC < 0.05) / 31 drunk (BAC \u2265 0.05) photos in the labeled subset
- Per-subject photo counts: 3\u201316, median ~7

### Per-subject distribution at threshold 0.05

| Subject | Total | Sober | Drunk | Notes        |
| ------- | ----: | ----: | ----: | ------------ |
| P1      |     5 |     5 |     0 | single class |
| P2      |     6 |     6 |     0 | single class |
| P3      |    11 |     8 |     3 |              |
| P4      |     9 |     3 |     6 |              |
| P5      |     6 |     3 |     3 |              |
| P6      |     3 |     3 |     0 | single class |
| P7      |    10 |     8 |     2 |              |
| P8      |     8 |     3 |     5 |              |
| P9      |     3 |     3 |     0 | single class |
| P11     |     9 |     3 |     6 |              |
| P12     |     9 |     3 |     6 |              |

---

## Errata: what changed since the previous report

The January 2026 evaluation report claimed, broadly:

- All four "drowsiness" models (SVM/KNN/NB/RF) achieved **100% accuracy and ROC-AUC 1.0**.
- The intoxication CNN achieved **81% accuracy and ROC-AUC 0.92**.

Both claims were artifacts of the evaluation setup, not properties of the models.

### Drowsiness models: tautology, not learning

The function `extract_features_from_folder` (`commands/train.py`, prior to this refactor) computed each training label by applying `label = 1 if EAR < threshold else 0`. The same EAR value was then included in the feature vector passed to the classifier. The classifier therefore had to learn a step function on one of its inputs, which any non-degenerate model can do trivially. The "100% accuracy" was the mathematical certainty of recovering a threshold from the value being thresholded, not evidence of any modeling capacity.

**Fix:** drowsiness no longer has a model file. `sober-scan detect --type drowsiness` applies the EAR < 0.2 rule directly (the same rule the labels were derived from), and `sober-scan train --detection-type drowsiness` returns a deprecation message. The four drowsiness `.joblib` files have been removed from `MODEL_URLS` in `config.py`; the existing files in `models/` remain on disk but should not be redistributed.

### Intoxication CNN: split contamination + a methodology bug

Two compounding problems produced the 81% / 0.92 figure:

1. **Within-subject random split.** `IntoxicationDataset` was split with `torch.utils.data.random_split`, which shuffles at the image level. Multiple photos of the same person therefore appeared in both train and test, allowing the model to score well by recognizing faces rather than detecting impairment.
2. **Double split.** The intoxication training branch called `model.train(dataset, ...)`, which internally split 80/20 (`seed=42`) for its own validation loop, and then the outer code did _another_ `random_split` of the same dataset (`seed=42`) to produce a "test set". The two splits drew from the same pool, so a large fraction of the reported "test" examples had already been seen during training.

**Fix:** there is now a separate, honest evaluation path:

- `IntoxicationCorpus.from_folder(...)` constructs a labeled corpus that knows about subjects.
- `loso_splits(...)` yields one (train, test) fold per subject \u2014 the held-out subject's photos appear in _no_ training fold.
- `evaluate_baseline(...)` runs that loop and reports pooled metrics.

The legacy `commands/train.py` intoxication branch retains the original split logic and the double-split bug; we have not touched it in this refactor because the user committed to Tier 1 baselines first and to deferring Tier 2 (proper CNN retraining) until these honest numbers were on paper.

### Training data provenance

The CNN's training corpus (`data/training/intoxication/`) is internet-scraped: the "sober" folder contains the trainer's own webcam selfies, a stock meme image, and `IMG_*.JPG` phone snaps; the "drunk" folder contains hash-named files from social-media and IMDb image hosts. The two classes are therefore distinguishable largely by image-source distribution (controlled portrait vs paparazzi flash) rather than by facial signal. This is consistent with the cross-subject AUC collapsing to 0.48 the moment the model is asked to generalize beyond that distribution.

---

## What this means in practice

- **Drowsiness detection works as a rule, not as a model.** EAR < 0.2 is a well-established eye-closure heuristic from the literature, and `sober-scan detect --type drowsiness` applies it transparently. There is nothing to retrain.
- **Intoxication detection is an open problem on this dataset.** The honest cross-subject baseline shows that a one-shot photo of a previously unseen person carries, at best, weak signal at BAC \u2265 0.05 with the features we currently extract. AUC of ~0.60 (handcrafted LR) is not nothing, but it is far from a useful screening tool.
- **The shipped CNN should not be relied on for decisions about new individuals.** Until a model can beat the handcrafted LR under LOSO on this same data \u2014 and ideally on a larger and more diverse corpus \u2014 the right honest stance is "this is research code, not a fitness-to-drive screener."

---

## What we have _not_ done in this iteration

Per the explicit grilling agreement before work began:

1. **Tier 2 models** (a properly-configured CNN trained per-fold with face crops and RGB input, a within-subject Siamese model on bd/ad pairs, a hybrid CNN+handcrafted classifier) are deferred until you decide they're worth building on top of the Tier 1 floor.
2. **External data sourcing** (Sober\u2013Drunk by Koukiou et al.; any other public intoxication face corpora) is deferred until we know whether the gap is "more data" or "harder problem."
3. **The 15 ambiguous photos** in `data/testing_data/` whose BAC isn't encoded in the filename are excluded from this baseline. Backfilling via the `Px.txt` session notes is a future ticket.
4. **`data/new_data/`** (the flight-simulator before/after photo corpus, ~120 paired photos across ~10 participants) is intentionally untouched. It's a fatigue dataset, not an intoxication one, and even pre-training use is deferred.
5. **The intoxication training command's double-split bug** is documented above but not patched, because the user committed to honest-evaluation work first.
6. **Bootstrap confidence intervals** on the pooled metrics are deferred; the per-fold breakdown in the JSON output is currently the way to assess variance.

---

## Open work

If you decide to push beyond the floor, the candidate next moves in roughly descending expected value:

1. **Backfill the 15 ambiguous photos** by parsing each subject's `.txt` session notes for the BAC closest in time to each ad photo. Pushes the labeled corpus from 79 to ~94 photos and shrinks per-fold variance.
2. **Refit the intoxication CNN under LOSO**, properly: RGB input, face crop with margin, frozen MobileNetV2 backbone, small classifier head, augmentation, no internal-validation double-split. Compare its honest LOSO numbers against `HandcraftedFeaturesLR` on the same folds.
3. **Build the within-subject Siamese model.** Pairs `(bd_photo, query_photo)` of the same subject and predicts "is the query different from baseline?". The bd/ad pairing is the strongest supervision signal in this dataset; the absolute-classification framing throws it away.
4. **Collect more controlled data.** With 11 subjects, even an optimal model has high LOSO variance. An extension of the ERAU study (or use of `data/new_data/` participants when they next come in for a session) is the only way to materially shrink confidence intervals.

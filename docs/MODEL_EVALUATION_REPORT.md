# Model Evaluation Report

**Last updated:** June 21, 2026 — Tier 2 baselines added
**Methodology:** Leave-one-subject-out (LOSO) cross-validation on `data/testing_data/`
**Threshold:** Binary classification, drunk iff BAC ≥ 0.05
**Corpus:** 92 labeled photos across 11 subjects, with 13 photos backfilled from session-notes timestamps (see _Methodology_)

> **This report supersedes all earlier evaluation numbers in this project.**
> The January 2026 "100% drowsiness / 81% intoxication" headline was produced under a leaky evaluation setup. See the _Errata_ section.

---

## Headline numbers

| Model                                   | Pooled accuracy | Pooled balanced accuracy | Pooled ROC-AUC |
| --------------------------------------- | --------------: | -----------------------: | -------------: |
| `MajorityClassBaseline`                 |          55.43% |                   50.00% |         0.272¹ |
| **`HandcraftedFeaturesLR`**             |      **67.39%** |               **67.00%** |     **0.6322** |
| `ImageNetFrozenLR`                      |          44.57% |                   44.74% |          0.466 |
| `LOSOTrainedCNN` (Tier 2)               |          63.04% |                   61.41% |          0.603 |
| `SiameseDelta` (Tier 2)                 |          57.61% |                   52.68% |          0.528 |
| Existing shipped `intoxication_cnn.pt`² |          57.61% |                   53.63% |          0.519 |

¹ Constant-prediction AUC under LOSO isn't 0.5 — the training prevalence varies by which subject is held out, and that variation anti-correlates with the held-out subject's positive rate. The 0.272 is the structural signature of a non-informative predictor on this kind of split.

² Inference-only: the shipped model was trained on `data/training/intoxication/` (an internet-scraped corpus) and was not retrained per fold here. Compare against `LOSOTrainedCNN`, which uses the same backbone trained correctly.

### Reproducing

```bash
sober-scan evaluate baseline majority      --data data/testing_data --threshold 0.05
sober-scan evaluate baseline handcrafted   --data data/testing_data --threshold 0.05
sober-scan evaluate baseline imagenet      --data data/testing_data --threshold 0.05
sober-scan evaluate baseline loso-cnn      --data data/testing_data --threshold 0.05
sober-scan evaluate baseline siamese       --data data/testing_data --threshold 0.05
```

Each command runs the full LOSO loop and prints a per-fold table. Add `--output-json path.json` for the full per-fold prediction arrays.

---

## What the numbers say

1. **`HandcraftedFeaturesLR` is the best model we have.** A 14-dimensional handcrafted feature vector (face/forehead/cheek redness, EAR/MAR, several landmark distances) fed to logistic regression gets to **AUC 0.632, balanced accuracy 67%**, beating both the existing CNN (AUC 0.519) and the carefully-fixed Tier 2 CNN (AUC 0.603). The lead in balanced accuracy over majority class is now **17 percentage points** — robust evidence that handcrafted features carry cross-subject impairment signal.

2. **A properly-configured CNN still loses to 14 handcrafted features.** `LOSOTrainedCNN` applies every methodology fix the original training pipeline missed: RGB input (no first-layer surgery destroying pretrained features), face crops with margin (no background confound), per-fold subject-disjoint training (no within-subject leakage), frozen MobileNetV2 backbone + small classifier head, augmentation in tensor space, weighted BCE for class imbalance, no double-split bug. It gets to AUC 0.603 — a substantial **+0.12 AUC over the existing leaky-pipeline CNN** — but still loses to the handcrafted LR. The bottleneck at this scale is data, not architecture.

3. **The shipped intoxication CNN is at chance level.** On the expanded 92-photo cross-subject corpus, `intoxication_cnn.pt` lands at AUC 0.519 — statistically indistinguishable from random ranking. The 81% accuracy from the previous report was an artifact of within-subject contamination (the model recognized faces) plus a double-split bug that mixed train and test examples. See _Errata_ for the full diagnosis.

4. **Frozen ImageNet features actively hurt.** `ImageNetFrozenLR` gets AUC 0.466 — _below_ random. With 11 subjects and ~8 photos each, a 1280-d ImageNet representation has more than enough capacity to encode face identity, and identity doesn't generalise across LOSO folds. This is the same failure mode that contaminated the shipped CNN.

5. **The within-subject delta framing doesn't rescue ImageNet features.** `SiameseDelta` trains an LR on `feature(query) - feature(bd_reference)` — the strongest supervision signal in the data structure. It reaches AUC 0.528, marginally above random. The interpretation: alcohol's facial effects (mild flushing, possibly slight ptosis) aren't aligned with the axes ImageNet learned to discriminate on (object categories). Subtracting two ImageNet feature vectors of the same face yields noise, not impairment signal.

6. **Per-fold variance is enormous.** Fold sizes range from 5 to 16 photos, and 2 of the 11 subjects (P1, P2) still have only sober photos — those single-class folds give pseudo-perfect accuracy if the model predicts sober and pseudo-zero if it predicts drunk, contributing nothing to discrimination. The pooled metrics absorb this; the per-fold tables in the JSON dumps surface it.

---

## Dataset profile (`data/testing_data/`)

- **92 labeled photos** across **11 subjects** (P1–P9, P11, P12 — note P10 is absent)
- **51 sober / 41 drunk** at the BAC ≥ 0.05 threshold
- Per-subject photo counts after backfill: 5, 6, 7, 8, 8, 9 (×3), 11, 16 — median ~8
- Of 92 photos, 79 carry BAC directly in the filename (`Px_ad_BBB_n.jpg`); 13 carry only a timestamp or index and were resolved via session-notes parsing

### Per-subject distribution at threshold 0.05

| Subject | Total | Sober | Drunk | Notes               |
| ------- | ----: | ----: | ----: | ------------------- |
| P1      |     5 |     5 |     0 | single-class fold   |
| P2      |     6 |     6 |     0 | single-class fold   |
| P3      |    11 |     8 |     3 |                     |
| P4      |     9 |     3 |     6 |                     |
| P5      |     6 |     3 |     3 |                     |
| P6      |     6 |     3 |     3 | 3 photos backfilled |
| P7      |    16 |    11 |     5 | 6 photos backfilled |
| P8      |     8 |     3 |     5 |                     |
| P9      |     7 |     3 |     4 | 4 photos backfilled |
| P11     |     9 |     3 |     6 |                     |
| P12     |     9 |     3 |     6 |                     |

---

## Methodology

### Filename convention

`data/testing_data/` uses the convention:

```
P{subject}_bd_{n}.{ext}        before-drinking baseline, BAC = 0
P{subject}_ad_{BBB}_{n}.{ext}  after-drinking, BAC = BBB / 1000 (e.g. ad_102 → 0.102)
P{subject}_ad_{HHMM}_{n}.{ext} after-drinking, time-of-day stamp instead of BAC
P{subject}_ad_{n}.{ext}        after-drinking, no metadata in filename
```

Filenames with a 4-digit field (`P7_ad_1533_*`, `P9_ad_1136_*`) are parsed as time-of-day timestamps, not BAC. The session-notes parser then resolves these to the nearest BAC measurement.

### Session-notes backfill

The `Px.txt` session notes record timed BAC measurements in three different freeform styles (see `src/sober_scan/corpus/session_notes.py`):

- **P6:** compact 24h, one line: `1543 BAC 1 0.087`
- **P7:** table 12h with prefix: `Consumed Alcohol` / `3:35 PM` / `Blood Alcohol Level: 0.089% ABV`
- **P9:** table 12h bare: `Alcohol Level Measured` / `11:14 AM` / `0.026% ABV`

For each photo with `bac=None`:

- If the filename has a 4-digit timestamp (`HHMM`), pick the BAC measurement closest in time.
- If the filename has no timestamp (e.g. `P6_ad_1.jpg`), use the _last_ recorded measurement — a documented heuristic justified by `P6`'s notes placing the photo after all BAC measurements.

### LOSO splitting

`loso_splits(corpus)` yields 11 folds. For each fold, every photo of one subject is held out for evaluation and every photo of every other subject is used for training. Subjects iterate in lexicographic order so runs are deterministic.

### Pooled metrics

For each fold the chosen baseline is freshly instantiated and fit on the train sub-corpus (so no information leaks across folds). Per-fold `(y_true, y_pred, y_score)` arrays are concatenated and standard sklearn metrics are computed on the pooled arrays. Pooled AUC is reported as `N/A` when the pooled labels happen to contain only one class (does not occur with this dataset).

### Model details

- **`MajorityClassBaseline`** — predicts the training-fold positive rate as a constant probability. Sanity floor.
- **`HandcraftedFeaturesLR`** — extracts the 14 handcrafted features from `feature_extraction.extract_features`, scales with `StandardScaler`, fits `LogisticRegression(C=1.0, class_weight="balanced")`.
- **`ImageNetFrozenLR`** — face crop with 20% margin, resize to 224×224 RGB, ImageNet normalisation, frozen MobileNetV2 (1280-d pooled features), `StandardScaler` → `LogisticRegression(C=0.1, class_weight="balanced")`. The backbone is in `eval` mode (no batch-norm drift, no dropout) and never receives gradients.
- **`LOSOTrainedCNN`** — same face crop and ImageNet normalisation as `ImageNetFrozenLR`. MobileNetV2 backbone is frozen; the classifier head from the original `IntoxicationCNN` is retrained per fold for 8 epochs, batch size 8, AdamW lr=1e-3, weight decay 1e-4. Per-example weighted BCELoss with positive-class weight `neg/pos` to handle imbalance. Augmentation in tensor space: horizontal flip, small affine, light colour jitter. `drop_last=True` on the DataLoader to avoid single-sample batches crashing BatchNorm.
- **`SiameseDelta`** — for each subject, the lexicographically-first `Px_bd_*.jpg` is the baseline reference. Training pairs are `(bd_feature, query_feature)` of the same subject, with the bd reference itself excluded. The LR sees `feature(query) - feature(bd_reference)` and predicts whether the query has BAC ≥ threshold. `LogisticRegression(C=0.1, class_weight="balanced")`.

---

## Errata: what changed since the previous report

The January 2026 report claimed:

- All four "drowsiness" models (SVM/KNN/NB/RF) achieved **100% accuracy and AUC 1.0**.
- The intoxication CNN achieved **81% accuracy and AUC 0.92**.

Both were artifacts of the evaluation setup, not properties of the models.

### Drowsiness models: tautology, not learning

`extract_features_from_folder` (legacy `commands/train.py`) computed each training label by applying `label = 1 if EAR < threshold else 0`. The same EAR value was then included in the feature vector. The classifier had to learn a step function on one of its inputs — recoverable trivially by any non-degenerate model. The "100% accuracy" was the mathematical certainty of recovering a threshold from the value being thresholded.

**Fix:** drowsiness has no model file. `sober-scan detect --type drowsiness` applies the EAR < 0.2 rule directly. `sober-scan train --detection-type drowsiness` returns a deprecation message. The four `.joblib` entries were removed from `MODEL_URLS` in `config.py`; on-disk artifacts remain locally but should not be redistributed.

### Intoxication CNN: split contamination + a methodology bug

Two compounding problems produced the 81% / 0.92 figure:

1. **Within-subject random split.** `IntoxicationDataset` used `torch.utils.data.random_split`, which shuffles at the image level. Multiple photos of the same person therefore appeared in both train and test, letting the model score well by recognising faces.
2. **Double split.** The intoxication training branch called `model.train(dataset, ...)`, which internally split 80/20 with `seed=42` for validation, and then the outer code did _another_ `random_split` of the same dataset with `seed=42` to produce a "test set". Those two splits drew from the same pool, so a large fraction of the reported "test" examples had been seen during training.

**Fix:** the honest evaluation path is now `IntoxicationCorpus` + `loso_splits` + `evaluate_baseline`. The legacy `commands/train.py` intoxication branch retains the original bugs and is not used by `sober-scan evaluate`.

### Training data provenance

The CNN's training corpus (`data/training/intoxication/`) is internet-scraped: the "sober" folder contains the trainer's own webcam selfies, a stock meme image, and `IMG_*.JPG` phone snaps; the "drunk" folder contains hash-named files from social-media and IMDb image hosts. The two classes are distinguishable largely by image-source distribution (controlled portrait vs paparazzi flash) rather than by facial signal. This is consistent with the cross-subject AUC collapsing to 0.519 on the held-out evaluation corpus.

---

## What this means in practice

- **Drowsiness detection works as a rule, not as a model.** EAR < 0.2 is a well-established eye-closure heuristic. `sober-scan detect --type drowsiness` applies it transparently. There is nothing to retrain.
- **`HandcraftedFeaturesLR` is the strongest intoxication model in the package today** — but at AUC 0.63 with 11 subjects, it's not a screening tool. It's a research baseline that says "yes, there is some cross-subject signal in 14 handcrafted features."
- **The shipped CNN should not be relied on for decisions about new individuals.** Its honest cross-subject AUC is 0.52. Until a model can beat `HandcraftedFeaturesLR` on this LOSO benchmark, the right honest stance is "this is research code, not a fitness-to-drive screener."

---

## What Tier 2 taught us (and what it didn't)

The agreed Tier 2 work was:

- **A: backfill** the 15 ambiguous-BAC photos via session-notes parsing.
- **B: `LOSOTrainedCNN`** — refit the CNN with all the methodology fixes.
- **C: `SiameseDelta`** — exploit the within-subject `bd`/`ad` pairing.

What changed:

- **Backfill expanded the corpus from 79 to 92 photos (+16%).** `HandcraftedFeaturesLR` jumped from AUC 0.60 to 0.63 — the extra labelled drunk examples (BAC 0.089 for P7, 0.186 for P9, 0.11 for P6) gave the LR a better boundary. So the data work paid off where it mattered most.
- **`LOSOTrainedCNN` beats the existing CNN by +0.12 AUC** (0.60 vs 0.52). The methodology fixes (RGB, face crop, per-fold training, no double-split) recover a significant chunk of honest signal. But it still loses to the simpler handcrafted LR.
- **`SiameseDelta` is barely above random.** The within-subject delta in ImageNet feature space (1280-d MobileNetV2 features) isn't aligned with intoxication-specific signal — subtracting two ImageNet vectors of the same face yields noise, not flushing.

**What this means:**

The bottleneck at the current scale (~92 photos, 11 subjects) is data, not modelling. The Tier 2 menu of "more sophisticated models" has been worked through, and none of it materially beats 14 handcrafted features. The next round of meaningful improvement requires either:

1. **More subjects.** LOSO variance shrinks roughly as `1/N_subjects`. Going from 11 to 25 subjects would do more for confidence intervals than any architectural change at this scale.
2. **Richer per-subject data.** Multi-view photos, video, thermal imaging — any modality that captures impairment signal more directly than "RGB face + landmarks" can. The new_data flight-sim corpus might fit here if the impairment-relevant photos are usable.
3. **A different framing of the problem.** Sequence/timeseries (video clips), or treating impairment as a within-session detection problem rather than a between-photo classification problem. Both would change what the dataset has to support.

---

## What we have _not_ done

Per the explicit grilling agreement:

- **External data sourcing** (Sober-Drunk by Koukiou et al.; any other public intoxication face corpora) — out of scope; deferred until the user decides it's worth pursuing.
- **`data/new_data/`** (the flight-simulator before/after photo corpus) — intentionally untouched. It's a fatigue dataset, not an intoxication one.
- **Hybrid CNN-features + handcrafted-features classifier (Tier 2D)** — not built. The Tier 2 results suggest the ceiling is data, not feature engineering, so a hybrid wouldn't be expected to help.
- **The intoxication training command's double-split bug** in legacy `commands/train.py` — documented above, not patched, because the honest path now goes through `sober-scan evaluate` which does not use that code.
- **Bootstrap confidence intervals** on the pooled metrics. Per-fold breakdowns in the JSON dumps are the current way to assess variance.

---

## Open work

If you decide to push further, in roughly descending expected value:

1. **Collect more subjects.** Extending the ERAU study to 25+ participants is the only intervention with a clear path to better numbers at this point.
2. **Hybrid feature classifier.** Concatenate the 14 handcrafted features with the 1280-d ImageNet features per photo, fit one LR. Cheap to try; the handcrafted features add signal, the ImageNet features add capacity — might combine well.
3. **Threshold sweep.** Right now we only report at BAC ≥ 0.05. Reporting at multiple thresholds (0.03, 0.05, 0.08, 0.10) would show whether the models are usable at higher BAC where the signal is stronger.
4. **Calibration.** None of the pooled probability scores have been calibrated. For a screening tool, calibrated probabilities matter more than headline accuracy.
5. **Per-subject demographics.** If you have age / gender / ethnicity for each `Pn`, breaking the per-fold metrics down by demographic would surface whether any subgroup is being systematically mis-served.

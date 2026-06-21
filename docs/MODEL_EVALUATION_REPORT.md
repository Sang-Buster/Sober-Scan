# Model Evaluation Report

**Last updated:** June 21, 2026 — post-LOSO iterations (issues #1, #2, #3)
**Methodology:** Leave-one-subject-out (LOSO) cross-validation on `data/testing_data/`
**Corpus:** 92 labeled photos across 11 subjects, 13 photos backfilled from session-notes timestamps

> **This report supersedes all earlier evaluation numbers.** The January 2026
> headline ("drowsiness 100% / intoxication CNN 81%") was a leaky-split
> artifact. See the _Errata_ section.

---

## Headline numbers

Evaluated at the two thresholds we care about — BAC ≥ 0.05 (EU driving
limit, our default) and BAC ≥ 0.08 (US driving limit, where the model
turns out to be strongest). All baselines deterministic; CNN runs use
`torch.manual_seed(0)`.

### At threshold 0.05 (n_drunk = 41 of 92)

| Model                                     |        Acc |    Bal acc |        AUC |      Brier |
| ----------------------------------------- | ---------: | ---------: | ---------: | ---------: |
| `MajorityClassBaseline`                   |     55.43% |     50.00% |     0.272¹ |     0.2568 |
| **`HandcraftedFeaturesLR`**               | **67.39%** | **67.00%** | **0.6322** |     0.2781 |
| `ImageNetFrozenLR`                        |     44.57% |     44.74% |      0.466 |     0.4012 |
| `HybridFeaturesLR` (handcrafted+ImageNet) |     52.17% |     53.52% |      0.516 |     0.3055 |
| `LOSOTrainedCNN`                          |     61.96% |     61.38% |      0.606 |     0.2514 |
| `SiameseDelta`                            |     57.61% |     52.68% |      0.528 |     0.3633 |
| Existing shipped `intoxication_cnn.pt`²   |     57.61% |     53.63% |      0.519 |     0.3576 |
| `HandcraftedFeaturesLR` + `--calibrate`   |     57.61% |     56.03% |      0.534 | **0.2530** |

### At threshold 0.08 (n_drunk = 36 of 92)

| Model                                   |    Acc |    Bal acc |        AUC |      Brier |
| --------------------------------------- | -----: | ---------: | ---------: | ---------: |
| `MajorityClassBaseline`                 | 60.87% |     50.00% |     0.226¹ |     0.2487 |
| **`HandcraftedFeaturesLR`**             | 66.30% | **67.36%** | **0.6791** |     0.2608 |
| `ImageNetFrozenLR`                      | 44.57% |     42.06% |      0.403 |     0.4189 |
| `HybridFeaturesLR`                      | 41.30% |     40.38% |      0.424 |     0.2993 |
| `LOSOTrainedCNN`                        | 55.43% |     56.94% |      0.548 |     0.2785 |
| `SiameseDelta`                          | 63.04% |     53.77% |      0.396 |     0.3359 |
| Existing shipped `intoxication_cnn.pt`² | 60.87% |     52.98% |      0.538 |     0.3269 |
| `HandcraftedFeaturesLR` + `--calibrate` | 60.87% |     53.47% |      0.542 | **0.2406** |

¹ Constant-prediction AUC under LOSO isn't 0.5 — the training prevalence varies by which subject is held out, and that variation anti-correlates with the held-out subject's positive rate. Structural artifact, not signal.

² Inference-only: the shipped model was trained on a separate (internet-scraped) corpus. Compare against `LOSOTrainedCNN`, which uses the same MobileNetV2 backbone trained correctly per fold.

### Reproducing

```bash
sober-scan evaluate baseline majority      -t 0.05 -t 0.08
sober-scan evaluate baseline handcrafted   -t 0.05 -t 0.08
sober-scan evaluate baseline imagenet      -t 0.05 -t 0.08
sober-scan evaluate baseline hybrid        -t 0.05 -t 0.08
sober-scan evaluate baseline loso-cnn      -t 0.05 -t 0.08
sober-scan evaluate baseline siamese       -t 0.05 -t 0.08
sober-scan evaluate baseline handcrafted   -t 0.05 -t 0.08 --calibrate
```

Each command prints the table. Append `--output-json path.json` for full per-fold predictions.

---

## What's new since the previous report

Three open issues from the previous round were closed in this iteration.

### Issue #2 — Threshold sweep

`--threshold` is now repeatable (`-t 0.03 -t 0.05 -t 0.08 -t 0.10`).
Running the sweep on `HandcraftedFeaturesLR`:

| Threshold | n_drunk |        Acc |    Bal acc |        AUC |  Brier |
| --------- | ------: | ---------: | ---------: | ---------: | -----: |
| 0.03      |      53 |     0.5978 |     0.6001 |     0.5922 | 0.2829 |
| 0.05      |      41 |     0.6739 |     0.6700 |     0.6322 | 0.2781 |
| **0.08**  |  **36** | **0.6630** | **0.6736** | **0.6791** | 0.2608 |
| 0.10      |      19 |     0.6413 |     0.5793 |     0.6208 | 0.2604 |

**The model is strongest at the US legal threshold (BAC ≥ 0.08)**, not at our previous default of 0.05. Balanced accuracy is essentially flat from 0.05 to 0.08, but AUC moves from 0.63 to 0.68. Biological interpretation: at higher BAC the facial signal (flushing, ptosis) is more pronounced and the classifier has stronger features to work with. The drop-off at 0.10 is small-sample variance (only 19 drunk photos).

### Issue #1 — Hybrid CNN+handcrafted feature classifier

New baseline `HybridFeaturesLR` concatenates the 14-d handcrafted vector with the 1280-d frozen MobileNetV2 face-crop embedding, scales, and fits an L1-penalised logistic regression. The L1 penalty is deliberate: with an L2 model the 1280 ImageNet features drown out the 14 handcrafted ones by sheer count.

**Result: the hybrid is shelved.** AUC at threshold 0.05 is 0.516 (handcrafted alone gets 0.632); at 0.08 it's 0.424 (vs 0.679). Even with L1 sparsity selection the L1 model can't extract enough signal from the 14 handcrafted features against the high-dim ImageNet noise. The baseline stays registered (`sober-scan evaluate baseline hybrid`) so the result is reproducible, but it's not on the recommended path.

The meta-finding: **at this dataset scale (92 photos / 11 subjects / 1294 features), feature concatenation does not produce additive gains**. The right next step is richer handcrafted features (which requires more subjects per #4) or a fundamentally different fusion architecture, not in scope here.

### Issue #3 — Calibration

New CLI flag `--calibrate` wraps any baseline in per-fold Platt scaling: each LOSO train fold is stratified-split 80/20 into a fit subset and a calibration subset; the base baseline fits the 80%; a single-feature logistic regression learns the mapping from raw score to label on the 20%. No data leak across folds because the split happens _inside_ the train fold.

**Result on `HandcraftedFeaturesLR`:**

|                             |    Acc | Bal acc |    AUC |      Brier |
| --------------------------- | -----: | ------: | -----: | ---------: |
| Baseline (threshold 0.05)   | 0.6739 |  0.6700 | 0.6322 |     0.2781 |
| Calibrated (threshold 0.05) | 0.5761 |  0.5603 | 0.5342 | **0.2530** |
| Baseline (threshold 0.08)   | 0.6630 |  0.6736 | 0.6791 |     0.2608 |
| Calibrated (threshold 0.08) | 0.6087 |  0.5347 | 0.5417 | **0.2406** |

**Brier improves substantially** (~0.025 reduction at both thresholds) — the calibrated probabilities are genuinely more honest. **AUC drops substantially** (~0.10-0.14) because the base baseline now sees only ~64 of ~80 train photos per fold and Platt scaling is fit on ~14 calibration photos. Both shrink with more data per #4.

For a screening deployment that needs trustworthy probabilities, `--calibrate` is the right knob. For a benchmark headline number, leave it off.

Brier is now also part of `LOSOReport` and shows up in every output (single-threshold, sweep, JSON) regardless of `--calibrate`.

### `LOSOTrainedCNN` is now deterministic

The CNN baseline now sets `torch.manual_seed(0)` and uses a seeded `DataLoader` generator. Re-runs produce identical numbers to 4 decimal places. The previous report's "AUC 0.603 at threshold 0.05" was a single un-seeded sample; the deterministic value is **0.606**. Cosmetic difference, but matters for anyone re-running.

---

## What still holds from the previous report

- **`HandcraftedFeaturesLR` is the best baseline.** It wins at threshold 0.05 (Acc 67.39%, Bal acc 67.00%, AUC 0.6322) and is essentially tied with itself at 0.08 (Bal acc 67.36%, AUC 0.6791). No more sophisticated model in this report beats it.
- **The shipped CNN is at chance level on honest cross-subject data.** AUC 0.519 at 0.05, 0.538 at 0.08. The 81% / 0.92 from the previous evaluation was within-subject contamination plus a double-split bug.
- **Frozen ImageNet features actively hurt.** AUC 0.466 at threshold 0.05 — below random — because identity-overfitting dominates with 1280-d features on 11 subjects.
- **At ~92 photos / 11 subjects, the bottleneck is data.** Every architectural variant we've tried (proper CNN per fold, Siamese delta in ImageNet space, hybrid concat with L1) has failed to beat 14 handcrafted geometry+colour features. More subjects would do more for the headline number than any model change.

---

## Dataset profile

- **92 labeled photos** across **11 subjects** (P1–P9, P11, P12; P10 absent)
- Class balance at threshold 0.05: **51 sober / 41 drunk**
- Class balance at threshold 0.08: **56 sober / 36 drunk**
- 79 photos carry BAC in the filename (`Px_ad_BBB_n.jpg`); the remaining 13 were backfilled from session notes
- Two subjects (P1, P2) have only sober photos — those LOSO folds contribute to accuracy but not to discrimination

---

## Errata: what changed since January 2026

The January 2026 report claimed:

- All four "drowsiness" models (SVM/KNN/NB/RF): 100% accuracy / AUC 1.0
- The intoxication CNN: 81% accuracy / AUC 0.92

Both were evaluation-setup artifacts.

**Drowsiness models** were trained by computing each label as `EAR < threshold` and then including the same EAR value in the feature vector — the classifier learned a step function on one of its inputs. Pure tautology. Fixed by removing the four models from `MODEL_URLS` and making `sober-scan detect --type drowsiness` apply the EAR < 0.2 rule directly.

**Intoxication CNN** numbers had two compounding problems: `torch.utils.data.random_split` shuffled photos at the image level so the same person appeared in train and test; and the outer training code did a second `random_split` over the same dataset with the same seed for "test", drawing from the same pool the model had already seen. Fixed by routing all evaluation through `sober-scan evaluate baseline`, which uses subject-disjoint LOSO splits.

The legacy `commands/train.py` still contains the original random-split intoxication path; it is no longer reachable through the trusted evaluation surface.

---

## What we have _not_ done (intentional)

- **External data sourcing** (Sober-Drunk by Koukiou et al., other thermal datasets) — Tracked separately as issue [#5](https://github.com/Sang-Buster/Sober-Scan/issues/5).
- **`data/new_data/`** (the flight-simulator before/after photo corpus) — fatigue dataset, not intoxication.
- **Bootstrap confidence intervals** — per-fold breakdowns in JSON dumps remain the way to assess variance.
- **Reliability diagram plots** — the Brier scores carry the calibration story numerically; matplotlib plots would be a future polish item.
- **Fixing the legacy `train.py` double-split bug** — the path is dead code from the evaluation perspective, so we documented it instead of cleaning it up.

---

## Open work

In rough descending expected value:

1. **More subjects** — [#4](https://github.com/Sang-Buster/Sober-Scan/issues/4). Tier 2 has been worked through; the next big move on the headline number requires more N. Going from 11 to 25 subjects would shrink LOSO variance ~2.3× and likely let the CNN baselines actually justify their complexity.
2. **External Sober-Drunk evaluation** — [#5](https://github.com/Sang-Buster/Sober-Scan/issues/5). Anchor our AUC against published baselines; requires acquiring the Koukiou dataset.
3. **Per-subject demographics** — if you have age / gender / ethnicity per `Pn`, breaking per-fold metrics down by demographic would surface whether any subgroup is being systematically under-served. No issue filed yet.
4. **Reliability diagrams** for `--calibrate` runs. Numerical Brier already tells the story; plots would be polish.

# Leave-one-subject-out is the only honest evaluation

The shipped intoxication CNN reported 81% accuracy / AUC 0.92, but the test set was produced by `torch.utils.data.random_split` over a flat directory of photos — multiple photos of the same person appeared in both train and test, so the model was scoring on face identity, not impairment. Compounded by a double-split bug in the training command, the published numbers were not reproducible on truly unseen subjects. We adopt leave-one-subject-out cross-validation as the only evaluation methodology in this project: every baseline is fit per fold, and the held-out subject's photos never appear in any training set the model has seen.

## Consequences

- All metrics carry high per-fold variance because N_subjects = 11. We accept this; the alternative is a methodologically dishonest number with smaller error bars.
- Pooled metrics across folds use concatenated `(y_true, y_pred, y_score)` arrays rather than fold-mean metrics — this prevents single-class folds (P1, P2 are all-sober) from collapsing the average. ROC-AUC for a constant predictor under LOSO is _not_ 0.5; the training-prevalence variation across folds correlates structurally with held-out class distribution.
- `sober-scan evaluate baseline ...` is the only supported evaluation surface. The legacy `commands/train.py` still contains the random-split intoxication path but its output is not trusted.

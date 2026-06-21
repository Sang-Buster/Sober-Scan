# Baseline protocol: `fit(corpus, *, threshold)` + `predict_proba(photo)`

Every baseline under `models/baselines/` implements the same two-method protocol: `fit(train: IntoxicationCorpus, *, threshold: float) -> Self` and `predict_proba(photo: Photo) -> float`. The protocol is duck-typed via a `Protocol` class in `evaluation/runner.py` rather than an abstract base class: any object exposing those two methods can be passed to `evaluate_baseline`. The shape mirrors scikit-learn but takes domain types (`IntoxicationCorpus`, `Photo`) rather than `(X, y)` arrays, which is what allowed the per-baseline feature-extraction caches (`_extract_for_path`, `_face_tensor_for_path`) to live module-level and be reused freely across folds.

## Consequences

- Adding a new baseline is mechanical: one new file under `models/baselines/`, one `Callable` entry in `_BASELINE_FACTORIES` in `commands/evaluate.py`, one enum value in `BaselineName`. Tested by smoke run.
- Baselines that don't naturally produce probabilities (e.g. `MajorityClassBaseline` returns the training prevalence as a constant) just return whatever they have — `evaluate_baseline` thresholds at 0.5 to get a discrete prediction for accuracy/balanced-accuracy and uses the raw score for AUC.
- `evaluate_baseline` is constructed with a `baseline_factory: Callable[[], _Baseline]` rather than a single baseline instance, so each fold gets a fresh object. Sharing state across folds would silently leak test photos into training; the factory pattern makes that mistake structurally impossible.
- Hyperparameters (CNN epochs, LR's `C`, etc.) are per-class constructor defaults rather than threading through the protocol. We accept the asymmetry: the protocol is for the data flow; the hyperparameters belong with the model.

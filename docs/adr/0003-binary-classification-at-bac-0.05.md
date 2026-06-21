# Binary classification at BAC ≥ 0.05

The corpus has BAC values from 0 to 0.186, which permits regression (predict the continuous BAC), multi-class (the existing `BACLevel` enum in `config.py`), or binary at any threshold. We picked **binary at BAC ≥ 0.05** as the project's default classification target because (a) it matches the EU driving limit and the academic "visibly impaired" band, (b) it gives a balanced ~50/50 class split on the available data (48/31 pre-backfill, 51/41 post-backfill), and (c) it lets us defer the harder regression problem until we have more subjects.

## Consequences

- Photos with BAC < 0.05 (including all `bd` baselines) are labeled `0` (sober) in training and evaluation.
- Photos with BAC ≥ 0.05 are labeled `1` (drunk).
- The threshold is a `--threshold` CLI argument so future work (#2 in the open issues) can sweep it.
- We do **not** use the US legal threshold (BAC ≥ 0.08): with only 26 drunk photos at that cutoff the LOSO variance would dominate the result.
- `BACLevel` (SOBER / MILD / MODERATE / SEVERE) remains in `config.py` but is unused by any active code path.

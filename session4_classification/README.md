# Session 4 — Classification and Validation

## What it does

Loads the per-subject instantaneous frequency and envelope arrays from Session 3, computes `std_ifreq` (standard deviation of physiologically filtered instantaneous gamma frequency), and runs a nested LOSO classification with logistic regression.

Also runs group statistics, MMSE correlation, permutation test, dummy baselines, and cross-validation scheme comparison.

## Feature

**std_ifreq** — standard deviation of instantaneous gamma frequency, filtered to 25–80 Hz.

Before classification: log-transform (`log1p`) to reduce right-skewness.

## Why this feature

In healthy interneuron networks, gamma oscillations maintain a relatively stable instantaneous frequency near the network resonant frequency (~40 Hz). Disruption of PV+ interneurons causes the frequency to wander — higher `std_ifreq`.

This is a mechanistic interpretation. The numbers either support it or they don't. On ds004504: AD=7.17±1.89, CN=5.53±1.09, Mann-Whitney p<0.0001.

## Classifier

Logistic regression with L2 regularisation, `class_weight='balanced'`, `max_iter=1000`. C selected by inner LOSO loop.

Logistic regression was chosen because it gives stable results across all C values (gap 0.2–0.4% across the full grid), its decision boundary is interpretable, and it avoids the SVM hyperparameter sensitivity issue encountered with RBF kernels on this dataset.

## Validation

| Test | Result | What it means |
|------|--------|---------------|
| LOSO accuracy | 73.8% | Subject-level generalisation |
| Overfitting gap | 0.3% | No overfitting |
| Permutation test | 49.2% | No data leakage |
| Dummy baseline | 55.4% | Not a class imbalance artifact |
| 5-fold CV | 73.8% | Same result regardless of CV scheme |
| 10-fold CV | 73.8% | Same result regardless of CV scheme |

## Inputs

```
eeg_alzheimer_pipeline/stage3_features/sub-XXX_ifreq.npy
eeg_alzheimer_pipeline/stage3_features/sub-XXX_envelope.npy
eeg_alzheimer_pipeline/logs/session2_5_vmd_sweep.json
eeg_alzheimer_pipeline/logs/participants_metadata.csv
```

## Outputs

```
eeg_alzheimer_pipeline/logs/session4_final_results_single_feat.json
```

Contents:
```json
{
  "n_AD": 36, "n_CN": 29,
  "accuracy": 73.8, "balanced_acc": 74.0,
  "f1": 73.9, "auc": 0.781,
  "auc_ci": [0.656, 0.886],
  "sensitivity": 72.2, "specificity": 75.9,
  "overfitting_gap": 0.3,
  "permutation_acc": 49.2,
  "confusion_matrix": [[22, 7], [10, 26]]
}
```

## Runtime

~15 minutes on CPU.

## Common issues

**AUC near 0** — `predict_proba` is returning CN probability. Add `ad_idx = list(clf.classes_).index(1)` and use `clf.predict_proba(Xte)[0][ad_idx]`.

**Some subjects missing** — Check Session 3 ran completely. `len(all_features)` should be 88.

**MMSE column not found** — Check column name in `participants_metadata.csv`. It may be `MMSE` or `mmse`. The code handles both.

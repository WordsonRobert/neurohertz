# Results

## Final Numbers (ds004504, 36 AD + 29 CN, LOSO)

| Metric | Value |
|--------|-------|
| Accuracy | 73.8% |
| Balanced Accuracy | 74.0% |
| F1 (weighted) | 73.9% |
| AUC | 0.781 |
| AUC 95% CI | [0.656, 0.886] |
| Sensitivity (AD) | 72.2% |
| Specificity (CN) | 75.9% |
| Overfitting gap | 0.3% |
| Permutation test | 49.2% |

## Feature Statistics

| Feature | AD mean | CN mean | p-value |
|---------|---------|---------|---------|
| std_ifreq | 7.17 ± 1.89 | 5.53 ± 1.09 | < 0.0001 *** |
| cv_envelope | 0.692 ± 0.168 | 0.703 ± 0.370 | 0.037 * (reversed) |

Only `std_ifreq` was used for classification. `cv_envelope` shows reversed direction (AD < CN) and was excluded.

## MMSE Correlation within AD (n=36)

| Feature | r | p-value |
|---------|---|---------|
| std_ifreq | -0.428 | 0.009 ** |
| cv_envelope | -0.036 | 0.834 ns |

## VMD Hyperparameter Distribution (from sweep)

| Parameter | Distribution |
|-----------|-------------|
| Best K | {6: 19, 7: 15, 8: 18, 9: 36} |
| Best alpha | {1000: 8, 2000: 10, 3000: 15, 4000: 55} |
| Channel | P3: 82/88, Pz: 3/88, P4: 3/88 |

## Confusion Matrix

```
              Pred CN   Pred AD
  True CN:      22         7
  True AD:      10        26
```

## Benchmarks

| Method | Accuracy | AUC | Validation | Features |
|--------|----------|-----|------------|----------|
| Miltiadous 2021 | 77.01% | — | LOSO | 95 |
| Shamsi 2025 | 83.1% | 0.930 | 5-fold GroupKFold | many |
| **Ours** | **73.8%** | **0.781** | **LOSO** | **1** |

Shamsi 2025 used group-stratified k-fold, not LOSO. Direct comparison requires caution.

## What is and isn't claimed

**Is:** std_ifreq separates AD from CN (p<0.0001) on this dataset, correlates with MMSE severity within AD (r=-0.428, p=0.009), and generalises to held-out subjects with no leakage.

**Is not:** state-of-the-art accuracy, validated on independent datasets, or ready for clinical use.

# Session 1 — EEG Preprocessing

## What it does

Loads raw `.set` files, applies signal cleaning, and saves spike-free EEG arrays to Drive.

## Steps

1. Load each `.set` file with MNE-Python
2. Bandpass filter: 0.5–80 Hz (4th order Butterworth, zero-phase)
3. Notch filter: 50 Hz (IIR, Q=30) — Greek powerline frequency
4. Average re-reference across all 19 channels
5. MAD-based spike removal per channel — spikes > 5×robust_std replaced by linear interpolation
6. Quality check — flags channels where >80% power was removed
7. Save cleaned array as float32 `.npy`

## Why 80 Hz not 45 Hz

The published ds004504 derivatives use a 45 Hz cutoff. That cuts into the gamma band (30–50 Hz). We reprocess from raw to keep gamma intact.

## Inputs

```
data/sub-XXX/eeg/sub-XXX_task-eyesclosed_eeg.set   ← raw EEG
data/participants.tsv                               ← group labels
```

## Outputs

```
eeg_alzheimer_pipeline/stage2_spike_clean/sub-XXX_clean.npy   ← float32, shape (19, N_samples)
eeg_alzheimer_pipeline/logs/session1_log.json                 ← processed/failed/quality_issues
eeg_alzheimer_pipeline/logs/participants_metadata.csv         ← copy of participants.tsv
```

## Runtime

~5 minutes on A100, ~5 minutes on CPU (no GPU needed here).

## Checkpointing

Saves after every subject. Safe to stop and rerun — skips already-processed subjects.

## Expected output

```
========== SESSION 1 COMPLETE ==========
Processed:  88 subjects
Failed:     0 subjects
Issues:     23 subjects    ← normal, just flagged channels
Time:       4.5 minutes
```

## Common issues

**`File not found`** — Check `DATA_DIR` is set to the folder containing `sub-001/`, not to `sub-001/eeg/` directly.

**Many failed subjects** — Recheck the path. The `.set` file must be at `DATA_DIR/sub-XXX/eeg/sub-XXX_task-eyesclosed_eeg.set`.

**23 quality issues** — Normal. These are subjects with one or two borderline channels. They are not excluded, just flagged.

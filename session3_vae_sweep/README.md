# Session 3 — VAE Training + Per-Subject VMD Hyperparameter Sweep

## What it does

Two things in sequence:

### Part A — VAE Training

Trains a Variational Autoencoder (VAE) on gamma mode segments from CN (healthy) subjects only. The VAE learns what healthy gamma looks like. Its reconstruction score is then used as an unsupervised proxy for gamma signal quality.

**Architecture:** FC encoder (256 → 128 → 128 → latent 16) + symmetric decoder  
**Training data:** 1,386,856 overlapping 256-sample segments from 29 CN subjects × 15 channels  
**Loss:** beta-VAE with beta=0.01 (mean reduction, not sum)  
**Epochs:** 200, cosine annealing LR schedule  
**Sanity check:** CN reconstruction score must be less negative than AD. If AD > CN, the VAE has not converged — delete sweep log and rerun.

### Part B — Per-Subject VMD Hyperparameter Sweep

For each subject, runs VMD with 16 combinations of K and alpha on the parietal channel (P3, or Pz/P4/C3 as fallback). Selects the combination that maximises the VAE reconstruction score. This is fully unsupervised — no group labels are used.

**Grid:**
- K ∈ {6, 7, 8, 9}
- alpha ∈ {1000, 2000, 3000, 4000}

**Result on this dataset:** K and alpha varied across subjects (K=9 was most common, alpha=4000 was most common), confirming the sweep is not trivial.

### Part C — Final Feature Extraction

Reruns VMD once per subject using the selected best K and alpha, then saves the gamma mode, envelope, and instantaneous frequency arrays for Session 4.

## Inputs

```
eeg_alzheimer_pipeline/stage2_spike_clean/sub-XXX_clean.npy
eeg_alzheimer_pipeline/stage3_gamma/components/sub-XXX_gamma.npy   ← for VAE training only
eeg_alzheimer_pipeline/logs/session1_log.json
eeg_alzheimer_pipeline/logs/session2_vmd_log.json
```

## Outputs

```
eeg_alzheimer_pipeline/logs/session2_5_vmd_sweep.json    ← best K/alpha per subject
eeg_alzheimer_pipeline/logs/session3_extraction.json     ← extraction checkpoint
eeg_alzheimer_pipeline/stage3_features/
    sub-XXX_ifreq.npy       ← instantaneous frequency, shape (N,)
    sub-XXX_envelope.npy    ← amplitude envelope, shape (N,)
    sub-XXX_gamma.npy       ← gamma mode, shape (N,)
```

## Runtime

- VAE training: ~1 hour on A100, ~6 hours on CPU
- Sweep (88 subjects × 16 combos): ~30 minutes on A100, ~26 minutes on CPU (P3 only — manageable)
- Feature extraction: ~30 minutes on A100

**All cells must run in the same Colab session.** The VAE and DEVICE variables must be in memory when the sweep runs. If you disconnect, rerun all cells from the top.

## Checkpointing

The sweep saves after every subject. The extraction also saves after every subject. Both resume on rerun.

## Expected output at sanity check

```
VAE sanity check:
  CN: mean=-0.1368    ← should be less negative
  AD: mean=-0.2123    ← should be more negative
```

If CN is more negative than AD, delete `session2_5_vmd_sweep.json` and rerun.

## Common issues

**`DEVICE is not defined`** — You ran the sweep cell without running the VAE cell first. Run all cells top to bottom in one session.

**Sweep silently produced K=8, alpha=2000 for all subjects** — Same issue. The sweep was silently catching NameError. Check `n_failed_combos` in the sweep log — if it's 16 for every subject, it failed silently.

**VAE loss not decreasing** — Check CN segments count. Should be >500,000. If it's much lower, gamma_data wasn't loaded correctly.

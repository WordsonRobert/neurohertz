# Session 2 — VMD Decomposition

## What it does

Runs Variational Mode Decomposition (VMD) on every channel of every subject and saves the gamma mode, its amplitude envelope, and instantaneous frequency to Drive.

## Why VMD

VMD decomposes a signal into K band-limited modes by solving a constrained optimisation problem. Unlike bandpass filtering, VMD adaptively finds each mode's centre frequency. Unlike EMD, it doesn't suffer from mode-mixing. The gamma mode (closest to 40 Hz) can be cleanly isolated even when adjacent beta and high-gamma modes are present.

## VMD parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| K | 8 | Spans delta through gamma with sub-band resolution |
| alpha | 2000 | Moderate bandwidth — each mode spans ~5-10 Hz |
| tau | 0 | No noise tolerance (signal already cleaned) |
| init | 1 | Uniformly spaced initial frequencies |
| tol | 1e-7 | Convergence criterion |

## Steps

1. Load spike-clean `.npy` from Drive (Session 1 output)
2. Run VMD independently on each of 19 channels per subject
3. Identify gamma mode = mode with centre frequency closest to 40 Hz
4. Compute Hilbert transform → instantaneous frequency + amplitude envelope
5. Save components, envelopes, inst_freqs, omega arrays

## Inputs

```
eeg_alzheimer_pipeline/stage2_spike_clean/sub-XXX_clean.npy
eeg_alzheimer_pipeline/logs/session1_log.json
```

## Outputs

```
eeg_alzheimer_pipeline/stage3_gamma/
├── components/sub-XXX_gamma.npy     ← gamma mode, shape (n_channels, N)
├── envelopes/sub-XXX_envelope.npy   ← amplitude envelope, shape (n_channels, N)
├── inst_freq/sub-XXX_instfreq.npy   ← instantaneous frequency, shape (n_channels, N)
└── omega/sub-XXX_omega.json         ← all 8 mode centre frequencies per channel
eeg_alzheimer_pipeline/logs/session2_vmd_log.json
```

## Runtime

~3 hours on A100 (88 subjects × 19 channels × full-length signal at 500 Hz).  
**Do not run on CPU** — full-signal VMD at 500 Hz takes ~88 hours on CPU.

## Checkpointing

Saves after every subject. Safe to stop and rerun.

## Expected output

```
========== SESSION 2 COMPLETE ==========
Processed:  88 subjects
Failed:     0 subjects
Time:       194 minutes
```

## Common issues

**Taking >5 minutes per subject** — You are not on A100. Switch runtime.

**VMD returns None** — vmdpy occasionally fails on near-flat channels. These are silently skipped. Check `channels_skipped` in session2_vmd_log.json.

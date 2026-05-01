# NeuroHertz — VMD Gamma Instability as an EEG Biomarker for Alzheimer's Disease

## What this repo is

A pipeline that extracts a single EEG feature — the standard deviation of the instantaneous gamma frequency (`std_ifreq`) — from resting-state EEG and tests whether it separates Alzheimer's disease (AD) patients from cognitively normal (CN) controls.

**Results on ds004504 (36 AD, 29 CN, LOSO validation):**
- Accuracy: 73.8%
- AUC: 0.781 (95% CI [0.656, 0.886])
- Sensitivity: 72.2%, Specificity: 75.9%
- Overfitting gap: 0.3%
- Permutation test: 49.2% (chance level — no leakage)
- std_ifreq group difference: p < 0.0001
- std_ifreq vs MMSE within AD: r = -0.428, p = 0.009

No claims are made beyond what the numbers say. This is one dataset, one site, 65 subjects.

---

## Repo structure

```
neurohertz/
├── data/                        ← put your dataset here (see below)
├── session1_preprocess/         ← EEG loading, filtering, spike removal
├── session2_vmd/                ← VMD decomposition on all 19 channels
├── session3_vae_sweep/          ← VAE training + per-subject VMD hyperparameter sweep
├── session4_classification/     ← LOSO classification, stats, validation
└── results/                     ← output JSONs and plots go here
```

---

## Data

**Dataset:** OpenNeuro ds004504  
**URL:** https://openneuro.org/datasets/ds004504  
**What you need:**
- All 88 subject folders (`sub-001` through `sub-088`)
- Each folder has structure: `sub-XXX/eeg/sub-XXX_task-eyesclosed_eeg.set`
- `participants.tsv` from the dataset root

**Where to put it:**
```
data/
├── participants.tsv
├── sub-001/
│   └── eeg/
│       └── sub-001_task-eyesclosed_eeg.set
├── sub-002/
│   └── eeg/
│       └── sub-002_task-eyesclosed_eeg.set
...
```

**Download options:**  
Option A — OpenNeuro web interface (slow, manual)  
Option B — AWS CLI (fast):
```bash
aws s3 sync --no-sign-request s3://openneuro.org/ds004504 data/
```
Option C — openneuro-py:
```bash
pip install openneuro-py
openneuro-py download --dataset ds004504 --target data/
```

---

## Environment

All sessions run on **Google Colab Pro with A100 GPU** except Session 4 which runs on CPU.

```
Python 3.12
mne >= 1.6
vmdpy == 0.2
torch >= 2.3
stable-baselines3 >= 2.3
scikit-learn >= 1.4
scipy >= 1.13
numpy >= 2.0
pandas
tqdm
```

Install in Colab:
```python
!pip install mne vmdpy torch stable-baselines3 scikit-learn scipy numpy pandas tqdm
```

---

## Run order

```
Session 1  →  Session 2  →  Session 3  →  Session 4
~5 min         ~3 hours       ~3 hours       ~15 min
(CPU/GPU)      (A100)         (A100/CPU)     (CPU)
```

Each session saves checkpoints to Drive. If it disconnects, rerun — it resumes where it left off.

---

## Common mistakes

**"File not found" errors in Session 1**  
Check your `DATA_DIR` path. The `.set` files must be at `DATA_DIR/sub-XXX/eeg/sub-XXX_task-eyesclosed_eeg.set`.

**VMD taking forever in Session 2**  
You need A100. On CPU, full-signal VMD at 500Hz takes ~88 hours. Don't run Session 2 on CPU.

**VAE sanity check shows AD > CN**  
The VAE trained incorrectly. Delete the sweep log and rerun Session 3 from scratch. Check that CN segments: shows >500,000 segments before training starts.

**Session 3 sweep silently failing**  
`vae` and `DEVICE` must be in memory before the sweep cell runs. Run all cells in Session 3 top to bottom in the same Colab session without disconnecting.

**AUC near 0 in Session 4**  
Use `clf.decision_function()` or ensure `predict_proba` is using class index 1 (AD), not 0 (CN). Use `ad_idx = list(clf.classes_).index(1)`.

**Permutation test giving 100% on some runs**  
This is a known artifact of nested channel selection on small N. The single-feature (no channel selection) permutation test is the valid one. 49.2% = clean.

---

## Output files (in Drive after each session)

```
eeg_alzheimer_pipeline/
├── logs/
│   ├── participants_metadata.csv
│   ├── session1_log.json
│   ├── session2_vmd_log.json
│   ├── session2_5_vmd_sweep.json
│   ├── session3_extraction.json
│   └── session4_final_results_single_feat.json
├── stage2_spike_clean/          ← preprocessed .npy per subject
├── stage3_gamma/                ← VMD gamma components per subject
│   ├── components/
│   ├── envelopes/
│   ├── inst_freq/
│   └── omega/
└── stage3_features/             ← final ifreq + envelope after sweep
    ├── sub-001_ifreq.npy
    ├── sub-001_envelope.npy
    ...
```

---

## Author

Wordson Robert  
IISER Kolkata

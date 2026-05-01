# Data

This folder is empty in the repo. Put your dataset here.

## Dataset

**OpenNeuro ds004504**  
URL: https://openneuro.org/datasets/ds004504  
Version: 1.0.2  
License: CC0

88 subjects: 36 AD, 23 FTD, 29 CN  
19-channel 10-20 EEG, 500 Hz, ~10 minutes eyes-closed resting state  
Format: EEGLAB .set files

## Required folder structure

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
└── sub-088/
    └── eeg/
        └── sub-088_task-eyesclosed_eeg.set
```

## Download

**Option A — AWS CLI (fastest):**
```bash
pip install awscli
aws s3 sync --no-sign-request s3://openneuro.org/ds004504 data/
```

**Option B — openneuro-py:**
```bash
pip install openneuro-py
openneuro-py download --dataset ds004504 --target data/
```

**Option C — Web browser:**  
Go to https://openneuro.org/datasets/ds004504, click Download.  
This downloads a zip. Unzip into `data/`.

## What you actually need

From the full dataset you only strictly need:
- `participants.tsv` (group labels + MMSE scores)
- All 88 `sub-XXX_task-eyesclosed_eeg.set` files

The `.json` and channels `.tsv` files inside each subject folder are not used.

## Verification

After downloading, run in Python:
```python
import os
missing = []
for i in range(1, 89):
    sid = f'sub-{i:03d}'
    p   = f'data/{sid}/eeg/{sid}_task-eyesclosed_eeg.set'
    if not os.path.exists(p):
        missing.append(sid)
print(f'Missing: {missing if missing else "None"}')
```

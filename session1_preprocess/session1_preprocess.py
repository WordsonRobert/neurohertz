# session1_preprocess.py
# Run on Google Colab — CPU is fine, no GPU needed
# Runtime: ~5 minutes

# ── CELL 1: Install + mount ────────────────────────────────────────────────────
import subprocess
print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)

# !pip install -q mne scipy numpy tqdm   # uncomment in Colab

from google.colab import drive
drive.mount('/content/drive')

import os, json, time
import numpy as np
import pandas as pd
import mne
import warnings
warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')

# ── PATHS — edit these ─────────────────────────────────────────────────────────
DATA_DIR   = '/content/drive/MyDrive/ds004504-download'   # folder with sub-XXX/ subdirs
DRIVE_BASE = '/content/drive/MyDrive/eeg_alzheimer_pipeline'

for folder in [DRIVE_BASE, f'{DRIVE_BASE}/stage2_spike_clean', f'{DRIVE_BASE}/logs']:
    os.makedirs(folder, exist_ok=True)

print(f'Data dir: {DATA_DIR}')
print(f'Output:   {DRIVE_BASE}')

pts_file = f'{DATA_DIR}/participants.tsv'
assert os.path.exists(pts_file), f'participants.tsv not found at {pts_file}'
print('participants.tsv found.')

# ── CELL 2: Load participants ──────────────────────────────────────────────────
participants = pd.read_csv(pts_file, sep='\t')
participants.columns = [c.strip() for c in participants.columns]
print('Group breakdown:')
print(participants['Group'].value_counts())
participants.to_csv(f'{DRIVE_BASE}/logs/participants_metadata.csv', index=False)

all_subjects = sorted(participants['participant_id'].tolist())

# Verify .set files
missing, present = [], []
for sid in all_subjects:
    p = os.path.join(DATA_DIR, sid, 'eeg', f'{sid}_task-eyesclosed_eeg.set')
    if os.path.exists(p): present.append(sid)
    else: missing.append(sid)

print(f'\nFiles present: {len(present)}/88')
if missing:
    print(f'Missing: {missing}')
    raise FileNotFoundError(f'{len(missing)} .set files missing. Check DATA_DIR.')
else:
    print('All 88 .set files present.')

# ── CELL 3: Processing functions ───────────────────────────────────────────────
def load_and_filter(subject_id, data_dir):
    set_file = os.path.join(data_dir, subject_id, 'eeg',
                            f'{subject_id}_task-eyesclosed_eeg.set')
    if not os.path.exists(set_file):
        return None, f'File not found: {set_file}'
    try:
        raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose=False)
        raw.filter(l_freq=0.5, h_freq=80.0, fir_window='hamming', verbose=False)
        raw.notch_filter(freqs=[50.0], verbose=False)
        raw.set_eeg_reference('average', projection=False, verbose=False)
        data = raw.get_data()
        return {
            'data':         data,
            'sfreq':        raw.info['sfreq'],
            'ch_names':     raw.ch_names,
            'n_samples':    data.shape[1],
            'duration_min': data.shape[1] / raw.info['sfreq'] / 60
        }, None
    except Exception as e:
        return None, str(e)


def remove_spikes_amplitude(signal, threshold_std=5.0, interp_margin=10):
    signal = signal.copy()
    n      = len(signal)
    mad    = np.median(np.abs(signal - np.median(signal)))
    robust_std = mad / 0.6745
    threshold  = threshold_std * robust_std
    spike_mask = np.abs(signal) > threshold
    expanded   = np.zeros(n, dtype=bool)
    for idx in np.where(spike_mask)[0]:
        start = max(0, idx - interp_margin)
        end   = min(n, idx + interp_margin + 1)
        expanded[start:end] = True
    if expanded.any():
        good = np.where(~expanded)[0]
        bad  = np.where(expanded)[0]
        if len(good) > 1:
            signal[bad] = np.interp(bad, good, signal[good])
    return signal


def remove_spikes_all_channels(data, sfreq=500):
    cleaned = np.zeros_like(data)
    for ch in range(data.shape[0]):
        cleaned[ch] = remove_spikes_amplitude(data[ch])
    return cleaned


def quality_check(original, cleaned, ch_names):
    issues = []
    for ch in range(original.shape[0]):
        orig_std  = np.std(original[ch])
        clean_std = np.std(cleaned[ch])
        if clean_std < 0.01 * orig_std:
            issues.append(f'Channel {ch_names[ch]}: went near-flat')
        if orig_std > 0 and (orig_std - clean_std) / orig_std > 0.8:
            issues.append(f'Channel {ch_names[ch]}: >80% power reduction')
    return issues


print('Functions defined.')

# ── CELL 4: Main loop ──────────────────────────────────────────────────────────
from tqdm import tqdm

log_file = f'{DRIVE_BASE}/logs/session1_log.json'
log = {'processed': [], 'failed': [], 'quality_issues': {}, 'subject_info': {}}

if os.path.exists(log_file):
    with open(log_file) as f:
        log = json.load(f)
    print(f'Resuming — already processed: {len(log["processed"])} subjects')
else:
    print('Starting fresh.')

start_time = time.time()

for subject_id in tqdm(all_subjects, desc='Preprocessing'):
    if subject_id in log['processed']:
        continue

    t0             = time.time()
    result, error  = load_and_filter(subject_id, DATA_DIR)

    if error:
        tqdm.write(f'  FAILED {subject_id}: {error}')
        log['failed'].append({'subject': subject_id, 'error': error})
        with open(log_file, 'w') as f: json.dump(log, f, indent=2)
        continue

    spike_clean = remove_spikes_all_channels(result['data'])
    issues      = quality_check(result['data'], spike_clean, result['ch_names'])
    if issues:
        log['quality_issues'][subject_id] = issues

    np.save(f'{DRIVE_BASE}/stage2_spike_clean/{subject_id}_clean.npy',
            spike_clean.astype(np.float32))

    log['subject_info'][subject_id] = {
        'sfreq':            result['sfreq'],
        'n_samples':        result['n_samples'],
        'duration_min':     round(result['duration_min'], 2),
        'ch_names':         result['ch_names'],
        'process_time_sec': round(time.time() - t0, 1)
    }
    log['processed'].append(subject_id)

    with open(log_file, 'w') as f: json.dump(log, f, indent=2)

total = (time.time() - start_time) / 60
print(f'\n========== SESSION 1 COMPLETE ==========')
print(f'Processed: {len(log["processed"])}   Failed: {len(log["failed"])}   Issues: {len(log["quality_issues"])}')
print(f'Time: {total:.1f} minutes')
print(f'Output: {DRIVE_BASE}/stage2_spike_clean/')

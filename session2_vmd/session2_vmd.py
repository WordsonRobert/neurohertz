# session2_vmd.py
# Run on Google Colab A100 — required, CPU is too slow
# Runtime: ~3 hours

# ── CELL 1: Install + GPU check ────────────────────────────────────────────────
import subprocess
print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)

# !pip install -q vmdpy scipy numpy tqdm   # uncomment in Colab

import vmdpy
import numpy as np
print('vmdpy installed.')

# ── CELL 2: Mount Drive ────────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import os, json, time
import numpy as np

DRIVE_BASE = '/content/drive/MyDrive/eeg_alzheimer_pipeline'

for d in [f'{DRIVE_BASE}/stage3_gamma',
          f'{DRIVE_BASE}/stage3_gamma/components',
          f'{DRIVE_BASE}/stage3_gamma/envelopes',
          f'{DRIVE_BASE}/stage3_gamma/inst_freq',
          f'{DRIVE_BASE}/stage3_gamma/omega']:
    os.makedirs(d, exist_ok=True)

clean_files = [f for f in os.listdir(f'{DRIVE_BASE}/stage2_spike_clean')
               if f.endswith('_clean.npy')]
print(f'Found {len(clean_files)} clean subjects')

# ── CELL 3: Load metadata ──────────────────────────────────────────────────────
import pandas as pd

participants = pd.read_csv(f'{DRIVE_BASE}/logs/participants_metadata.csv')

with open(f'{DRIVE_BASE}/logs/session1_log.json') as f:
    session1_log = json.load(f)

processed_subjects = session1_log['processed']
quality_issues     = session1_log['quality_issues']

# Build bad channel lookup
bad_channels = {}
for subj, issues in quality_issues.items():
    bad_chs = []
    for issue in issues:
        try:
            ch = issue.split('Channel ')[1].split(':')[0]
            bad_chs.append(ch)
        except:
            pass
    bad_channels[subj] = bad_chs

print(f'Subjects to process: {len(processed_subjects)}')

# ── CELL 4: VMD functions ──────────────────────────────────────────────────────
from scipy.signal import hilbert
from vmdpy import VMD

VMD_K     = 8
VMD_ALPHA = 2000
VMD_TAU   = 0
VMD_DC    = 0
VMD_INIT  = 1
VMD_TOL   = 1e-7
SFREQ     = 500
TARGET_HZ = 40.0


def run_vmd_single_channel(signal):
    try:
        u, _, omega = VMD(signal, VMD_ALPHA, VMD_TAU, VMD_K,
                          VMD_DC, VMD_INIT, VMD_TOL)
        return u, omega[-1, :] * SFREQ
    except:
        return None, None


def process_subject_vmd(subject_id, bad_chs=[]):
    clean_path = f'{DRIVE_BASE}/stage2_spike_clean/{subject_id}_clean.npy'
    if not os.path.exists(clean_path):
        return None, 'Clean file not found'

    data     = np.load(clean_path)
    ch_names = session1_log['subject_info'][subject_id]['ch_names']

    results = {
        'gamma_components': {}, 'envelopes': {},
        'inst_freqs': {}, 'omega_hz': {},
        'gamma_center_hz': {}, 'skipped_channels': bad_chs
    }

    for ch_idx, ch_name in enumerate(ch_names):
        if ch_name in bad_chs:
            continue
        signal = data[ch_idx].astype(np.float64)
        u, omega_hz = run_vmd_single_channel(signal)
        if u is None:
            continue
        gamma_idx  = np.argmin(np.abs(omega_hz - TARGET_HZ))
        gamma_mode = u[gamma_idx]

        analytic  = hilbert(gamma_mode)
        envelope  = np.abs(analytic).astype(np.float32)
        phase     = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(phase) * SFREQ / (2 * np.pi)
        inst_freq = np.append(inst_freq, inst_freq[-1]).astype(np.float32)

        results['gamma_components'][ch_name] = gamma_mode.astype(np.float32)
        results['envelopes'][ch_name]        = envelope
        results['inst_freqs'][ch_name]       = inst_freq
        results['omega_hz'][ch_name]         = omega_hz.astype(np.float32)
        results['gamma_center_hz'][ch_name]  = float(omega_hz[gamma_idx])

    return results, None


print(f'VMD functions defined. K={VMD_K}, alpha={VMD_ALPHA}')

# ── CELL 5: Test on sub-001 ────────────────────────────────────────────────────
t0 = time.time()
test_result, error = process_subject_vmd('sub-001', bad_channels.get('sub-001', []))
elapsed = time.time() - t0
print(f'sub-001: {elapsed:.1f}s, channels={len(test_result["gamma_components"])}')
print(f'Estimated total: {elapsed * len(processed_subjects) / 60:.0f} minutes')

# ── CELL 6: Full run ───────────────────────────────────────────────────────────
from tqdm import tqdm

vmd_log_file = f'{DRIVE_BASE}/logs/session2_vmd_log.json'
vmd_log = {'processed': [], 'failed': [], 'gamma_summary': {}}

if os.path.exists(vmd_log_file):
    with open(vmd_log_file) as f: vmd_log = json.load(f)
    print(f'Resuming — already done: {len(vmd_log["processed"])} subjects')
else:
    print('Starting fresh.')

start_time = time.time()

for subject_id in tqdm(processed_subjects, desc='VMD'):
    if subject_id in vmd_log['processed']:
        continue

    t0      = time.time()
    bad_chs = bad_channels.get(subject_id, [])
    result, error = process_subject_vmd(subject_id, bad_chs)

    if error:
        tqdm.write(f'  FAILED {subject_id}: {error}')
        vmd_log['failed'].append({'subject': subject_id, 'error': error})
        with open(vmd_log_file, 'w') as f: json.dump(vmd_log, f, indent=2)
        continue

    ch_names   = list(result['gamma_components'].keys())
    components = np.array([result['gamma_components'][ch] for ch in ch_names], dtype=np.float32)
    envelopes  = np.array([result['envelopes'][ch]        for ch in ch_names], dtype=np.float32)
    inst_freqs = np.array([result['inst_freqs'][ch]       for ch in ch_names], dtype=np.float32)

    np.save(f'{DRIVE_BASE}/stage3_gamma/components/{subject_id}_gamma.npy',   components)
    np.save(f'{DRIVE_BASE}/stage3_gamma/envelopes/{subject_id}_envelope.npy', envelopes)
    np.save(f'{DRIVE_BASE}/stage3_gamma/inst_freq/{subject_id}_instfreq.npy', inst_freqs)

    with open(f'{DRIVE_BASE}/stage3_gamma/omega/{subject_id}_omega.json', 'w') as f:
        json.dump({ch: result['omega_hz'][ch].tolist() for ch in ch_names}, f)

    vmd_log['gamma_summary'][subject_id] = {
        'channels_processed': ch_names,
        'channels_skipped':   bad_chs,
        'mean_gamma_hz':      round(np.mean(list(result['gamma_center_hz'].values())), 2),
        'process_time_sec':   round(time.time() - t0, 1)
    }
    vmd_log['processed'].append(subject_id)
    with open(vmd_log_file, 'w') as f: json.dump(vmd_log, f, indent=2)

total = (time.time() - start_time) / 60
print(f'\n========== SESSION 2 COMPLETE ==========')
print(f'Processed: {len(vmd_log["processed"])}   Failed: {len(vmd_log["failed"])}')
print(f'Time: {total:.1f} minutes')

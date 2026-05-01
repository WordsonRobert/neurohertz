# session3_vae_sweep.py
# Run on Google Colab — A100 recommended for VAE training
# ALL CELLS MUST RUN IN THE SAME SESSION — vae and DEVICE must stay in memory
# Runtime: ~2 hours total

# ── CELL 1: Install + setup ────────────────────────────────────────────────────
# !pip install -q vmdpy torch   # uncomment in Colab

from google.colab import drive
drive.mount('/content/drive')

import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

import os, json
import numpy as np
import pandas as pd
from tqdm import tqdm

DRIVE_BASE   = '/content/drive/MyDrive/eeg_alzheimer_pipeline'
KEY_CHANNELS = ['Fp1','Fp2','F3','Fz','F4','T3','C3','C4','T4','T6','P3','Pz','P4','O1','O2']

with open(f'{DRIVE_BASE}/logs/session1_log.json') as f:
    session1_log = json.load(f)
with open(f'{DRIVE_BASE}/logs/session2_vmd_log.json') as f:
    session2_log = json.load(f)

participants = pd.read_csv(f'{DRIVE_BASE}/logs/participants_metadata.csv')
processed    = session2_log['processed']

ad_subjects = participants[(participants['Group']=='A') &
                           (participants['participant_id'].isin(processed))]['participant_id'].tolist()
cn_subjects = participants[(participants['Group']=='C') &
                           (participants['participant_id'].isin(processed))]['participant_id'].tolist()

print(f'AD: {len(ad_subjects)}, CN: {len(cn_subjects)}')

# ── CELL 2: Load gamma components for VAE training ─────────────────────────────
gamma_data = {}
print('Loading gamma components...')

for subject_id in tqdm(cn_subjects + ad_subjects[:8]):
    ch_names = session1_log['subject_info'][subject_id]['ch_names']
    skipped  = session2_log['gamma_summary'][subject_id]['channels_skipped']
    good_chs = [c for c in ch_names if c not in skipped]
    try:
        gamma_arr = np.load(f'{DRIVE_BASE}/stage3_gamma/components/{subject_id}_gamma.npy')
        gamma_data[subject_id] = {ch: gamma_arr[i] for i, ch in enumerate(good_chs)}
    except:
        pass

print(f'Loaded {len(gamma_data)} subjects.')

# ── CELL 3: Define and train VAE ───────────────────────────────────────────────
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

SEGMENT_LEN = 256
LATENT_DIM  = 16
VAE_EPOCHS  = 200
VAE_BATCH   = 256
VAE_LR      = 1e-3


class GammaVAE(nn.Module):
    def __init__(self, input_len=SEGMENT_LEN, latent_dim=LATENT_DIM):
        super().__init__()
        hidden = 128
        self.encoder_fc = nn.Sequential(
            nn.Linear(input_len, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.decoder   = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),     nn.ReLU(),
            nn.Linear(hidden, input_len)
        )

    def encode(self, x):
        h = self.encoder_fc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def reconstruction_score(self, x):
        self.eval()
        with torch.no_grad():
            recon, mu, logvar = self.forward(x)
            return -torch.mean((recon - x) ** 2, dim=1)


def vae_loss(recon, x, mu, logvar, beta=0.01):
    recon_loss = nn.functional.mse_loss(recon, x, reduction='mean')
    kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss


# Build CN training segments
print('Building VAE training dataset...')
cn_segments = []
for subject_id in cn_subjects:
    for ch in KEY_CHANNELS:
        if ch not in gamma_data.get(subject_id, {}):
            continue
        signal = gamma_data[subject_id][ch].astype(np.float32)
        for start in range(0, len(signal) - SEGMENT_LEN, SEGMENT_LEN // 2):
            seg = signal[start : start + SEGMENT_LEN]
            std = seg.std()
            if std > 1e-10:
                seg = (seg - seg.mean()) / std
                if np.abs(seg).max() < 10:
                    cn_segments.append(seg)

cn_segments = np.array(cn_segments, dtype=np.float32)
print(f'CN segments: {len(cn_segments)} × {SEGMENT_LEN}')

X_train  = torch.tensor(cn_segments).to(DEVICE)
dataset  = TensorDataset(X_train)
loader   = DataLoader(dataset, batch_size=VAE_BATCH, shuffle=True)

vae       = GammaVAE().to(DEVICE)
optimizer = optim.Adam(vae.parameters(), lr=VAE_LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=VAE_EPOCHS)

print(f'Training VAE — {VAE_EPOCHS} epochs...')
prev_loss = float('inf')
for epoch in range(VAE_EPOCHS):
    epoch_loss = 0
    vae.train()
    for (batch,) in loader:
        optimizer.zero_grad()
        recon, mu, logvar = vae(batch)
        loss = vae_loss(recon, batch, mu, logvar)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()
    avg = epoch_loss / len(loader)
    if (epoch + 1) % 20 == 0:
        print(f'  Epoch {epoch+1}/{VAE_EPOCHS}: loss={avg:.4f}  (prev={prev_loss:.4f})')
        prev_loss = avg

# Sanity check — CN must be less negative than AD
print('\nVAE sanity check:')
vae.eval()
for group_name, group_subjects in [('CN', cn_subjects[:8]), ('AD', ad_subjects[:8])]:
    scores = []
    for s in group_subjects:
        if s not in gamma_data: continue
        ch  = 'Fz' if 'Fz' in gamma_data[s] else list(gamma_data[s].keys())[0]
        sig = gamma_data[s][ch][:SEGMENT_LEN].astype(np.float32)
        std = sig.std()
        if std < 1e-10: continue
        seg = (sig - sig.mean()) / std
        x   = torch.tensor(seg).unsqueeze(0).to(DEVICE)
        scores.append(vae.reconstruction_score(x).item())
    print(f'  {group_name}: mean={np.mean(scores):.4f}')
print('Target: CN mean > AD mean (less negative)')

# ── CELL 4: Per-subject VMD hyperparameter sweep ───────────────────────────────
from vmdpy import VMD
from scipy.signal import hilbert

SFREQ        = 500.0
TARGET_HZ    = 40.0
K_VALUES     = [6, 7, 8, 9]
ALPHA_VALUES = [1000, 2000, 3000, 4000]

SWEEP_FILE = f'{DRIVE_BASE}/logs/session2_5_vmd_sweep.json'
if os.path.exists(SWEEP_FILE):
    with open(SWEEP_FILE) as f: vmd_sweep_log = json.load(f)
    print(f'Resuming — already done: {len(vmd_sweep_log)} subjects')
else:
    vmd_sweep_log = {}

print(f'Grid: {len(K_VALUES)*len(ALPHA_VALUES)} combos per subject, P3 channel')

for subject_id in tqdm(processed, desc='VMD sweep'):
    if subject_id in vmd_sweep_log:
        continue

    clean_path = f'{DRIVE_BASE}/stage2_spike_clean/{subject_id}_clean.npy'
    if not os.path.exists(clean_path):
        continue

    data     = np.load(clean_path).astype(np.float64)
    ch_names = session1_log['subject_info'][subject_id]['ch_names']
    skipped  = session2_log['gamma_summary'][subject_id]['channels_skipped']
    good_chs = [c for c in ch_names if c not in skipped]

    target_ch = None
    for ch in ['P3', 'Pz', 'P4', 'C3', 'Cz']:
        if ch in good_chs and ch in ch_names:
            target_ch = ch
            break
    if target_ch is None:
        continue

    signal = data[ch_names.index(target_ch)].copy()

    best_vae_score = -np.inf
    best_K = best_alpha = best_std_ifreq = best_cv_env = best_gamma_hz = None
    sweep_results = {}
    n_failed = 0

    for K in K_VALUES:
        for alpha in ALPHA_VALUES:
            key = f'K{K}_a{alpha}'
            try:
                u, _, omega = VMD(signal, alpha, 0, K, 0, 1, 1e-7)
                omega_hz    = omega[-1] * SFREQ
                gamma_idx   = np.argmin(np.abs(omega_hz - TARGET_HZ))
                gamma_mode  = u[gamma_idx]

                analytic  = hilbert(gamma_mode)
                envelope  = np.abs(analytic)
                phase     = np.unwrap(np.angle(analytic))
                inst_freq = np.diff(phase) * SFREQ / (2 * np.pi)

                valid = (inst_freq > 25) & (inst_freq < 80)
                if valid.sum() < 1000: continue

                std_ifreq = float(np.std(inst_freq[valid]))
                cv_env    = float(np.std(envelope) / (np.mean(envelope) + 1e-10))

                seg = gamma_mode[:SEGMENT_LEN].astype(np.float32)
                seg_std = seg.std()
                if seg_std < 1e-10: continue
                seg_norm = (seg - seg.mean()) / seg_std

                with torch.no_grad():
                    x_tensor  = torch.tensor(seg_norm).unsqueeze(0).to(DEVICE)
                    vae_score = float(vae.reconstruction_score(x_tensor).cpu().item())

                sweep_results[key] = {
                    'vae_score': vae_score, 'std_ifreq': std_ifreq,
                    'cv_envelope': cv_env, 'gamma_hz': float(omega_hz[gamma_idx])
                }

                if vae_score > best_vae_score:
                    best_vae_score = vae_score
                    best_K, best_alpha = K, alpha
                    best_std_ifreq, best_cv_env = std_ifreq, cv_env
                    best_gamma_hz = float(omega_hz[gamma_idx])

            except Exception:
                n_failed += 1

    if best_K is None:
        tqdm.write(f'  All combos failed: {subject_id} ({n_failed} errors)')
        continue

    vmd_sweep_log[subject_id] = {
        'best_K': best_K, 'best_alpha': best_alpha,
        'best_vae_score': best_vae_score, 'std_ifreq': best_std_ifreq,
        'cv_envelope': best_cv_env, 'best_gamma_hz': best_gamma_hz,
        'channel': target_ch, 'n_failed_combos': n_failed,
        'all_results': sweep_results
    }
    with open(SWEEP_FILE, 'w') as f: json.dump(vmd_sweep_log, f, indent=2)

print(f'\nSweep complete: {len(vmd_sweep_log)} subjects')

from collections import Counter
print('K distribution:    ', dict(sorted(Counter(v['best_K']     for v in vmd_sweep_log.values()).items())))
print('Alpha distribution:', dict(sorted(Counter(v['best_alpha'] for v in vmd_sweep_log.values()).items())))

# ── CELL 5: Final feature extraction with best K/alpha ─────────────────────────
OUT_DIR           = f'{DRIVE_BASE}/stage3_features'
SESSION3_LOG_FILE = f'{DRIVE_BASE}/logs/session3_extraction.json'
os.makedirs(OUT_DIR, exist_ok=True)

if os.path.exists(SESSION3_LOG_FILE):
    with open(SESSION3_LOG_FILE) as f: session3_log = json.load(f)
else:
    session3_log = {'processed': []}

print(f'Extracting final features for {len(vmd_sweep_log)} subjects...')

for subject_id, params in tqdm(vmd_sweep_log.items(), desc='Extraction'):
    if subject_id in session3_log['processed']:
        continue

    clean_path = f'{DRIVE_BASE}/stage2_spike_clean/{subject_id}_clean.npy'
    if not os.path.exists(clean_path):
        continue

    data      = np.load(clean_path).astype(np.float64)
    ch_names  = session1_log['subject_info'][subject_id]['ch_names']
    target_ch = params['channel']
    signal    = data[ch_names.index(target_ch)].copy()

    u, _, omega = VMD(signal, params['best_alpha'], 0, params['best_K'], 0, 1, 1e-7)
    omega_hz    = omega[-1] * SFREQ
    gamma_mode  = u[np.argmin(np.abs(omega_hz - TARGET_HZ))]

    analytic  = hilbert(gamma_mode)
    envelope  = np.abs(analytic)
    phase     = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) * SFREQ / (2 * np.pi)
    inst_freq = np.append(inst_freq, inst_freq[-1])

    np.save(f'{OUT_DIR}/{subject_id}_gamma.npy',    gamma_mode.astype(np.float32))
    np.save(f'{OUT_DIR}/{subject_id}_envelope.npy', envelope.astype(np.float32))
    np.save(f'{OUT_DIR}/{subject_id}_ifreq.npy',    inst_freq.astype(np.float32))

    session3_log['processed'].append(subject_id)

with open(SESSION3_LOG_FILE, 'w') as f: json.dump(session3_log, f, indent=2)
print(f'\nSession 3 complete. Features saved to: {OUT_DIR}')

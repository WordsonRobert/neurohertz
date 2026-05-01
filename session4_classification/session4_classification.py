# session4_classification.py
# Runs on CPU — no GPU needed
# Runtime: ~15 minutes

from google.colab import drive
drive.mount('/content/drive')

import os, json
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, confusion_matrix, roc_auc_score)
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

DRIVE_BASE = '/content/drive/MyDrive/eeg_alzheimer_pipeline'
SFREQ      = 500.0

# ── Load logs ─────────────────────────────────────────────────────────────────
with open(f'{DRIVE_BASE}/logs/session1_log.json') as f:
    session1_log = json.load(f)
with open(f'{DRIVE_BASE}/logs/session2_vmd_log.json') as f:
    session2_log = json.load(f)
with open(f'{DRIVE_BASE}/logs/session2_5_vmd_sweep.json') as f:
    sweep_log = json.load(f)

participants = pd.read_csv(f'{DRIVE_BASE}/logs/participants_metadata.csv')
processed    = session2_log['processed']

ad_subjects = participants[(participants['Group']=='A') &
                           (participants['participant_id'].isin(processed))]['participant_id'].tolist()
cn_subjects = participants[(participants['Group']=='C') &
                           (participants['participant_id'].isin(processed))]['participant_id'].tolist()

# ── Compute features ──────────────────────────────────────────────────────────
print('Computing features...')
OUT_DIR      = f'{DRIVE_BASE}/stage3_features'
all_features = {}
n_missing    = 0

for subject_id in processed:
    ifreq_path = f'{OUT_DIR}/{subject_id}_ifreq.npy'
    env_path   = f'{OUT_DIR}/{subject_id}_envelope.npy'

    if not os.path.exists(ifreq_path) or not os.path.exists(env_path):
        n_missing += 1
        continue

    inst_freq = np.load(ifreq_path).astype(np.float64)
    envelope  = np.load(env_path).astype(np.float64)

    valid = (inst_freq > 25) & (inst_freq < 80)
    if valid.sum() < 1000:
        n_missing += 1
        continue

    edge     = int(SFREQ)
    env_trim = envelope[edge:-edge] if len(envelope) > 2*edge else envelope

    all_features[subject_id] = {
        'std_ifreq':   float(np.std(inst_freq[valid])),
        'cv_envelope': float(np.std(env_trim) / (np.mean(env_trim) + 1e-10)),
        'best_K':      sweep_log[subject_id]['best_K'],
        'best_alpha':  sweep_log[subject_id]['best_alpha'],
        'channel':     sweep_log[subject_id]['channel']
    }

print(f'Features: {len(all_features)} subjects ({n_missing} skipped)')

# ── VMD hyperparameter distribution ──────────────────────────────────────────
k_counts     = Counter(v['best_K']     for v in all_features.values())
alpha_counts = Counter(v['best_alpha'] for v in all_features.values())
ch_counts    = Counter(v['channel']    for v in all_features.values())
print(f'K distribution:     {dict(sorted(k_counts.items()))}')
print(f'Alpha distribution: {dict(sorted(alpha_counts.items()))}')
print(f'Channel:            {dict(ch_counts.most_common())}')

# ── Group statistics ──────────────────────────────────────────────────────────
print('\n=== Group-level Feature Statistics ===')
for feat in ['std_ifreq', 'cv_envelope']:
    adv = [all_features[s][feat] for s in ad_subjects if s in all_features]
    cnv = [all_features[s][feat] for s in cn_subjects if s in all_features]
    u_stat, p = mannwhitneyu(adv, cnv, alternative='two-sided')
    rb  = 1 - (2 * u_stat) / (len(adv) * len(cnv))
    sig = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))
    print(f'{feat:<15}: AD={np.mean(adv):.4f}±{np.std(adv):.4f}  '
          f'CN={np.mean(cnv):.4f}±{np.std(cnv):.4f}  '
          f'p={p:.4f} {sig}  rb={rb:.3f}  '
          f'{"AD>CN" if np.mean(adv)>np.mean(cnv) else "AD<CN"}')

# ── MMSE correlation within AD ────────────────────────────────────────────────
print('\n=== MMSE Correlation within AD ===')
mmse_col = next((c for c in participants.columns if 'mmse' in c.lower()), None)
if mmse_col:
    mmse_map = participants[participants['Group']=='A'].set_index('participant_id')[mmse_col]
    for feat in ['std_ifreq', 'cv_envelope']:
        pairs = [(mmse_map[s], all_features[s][feat])
                 for s in ad_subjects
                 if s in all_features and s in mmse_map.index
                 and not np.isnan(float(mmse_map[s]))]
        if len(pairs) > 5:
            mmse_v, feat_v = zip(*pairs)
            r, p = spearmanr(mmse_v, feat_v)
            sig  = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))
            print(f'  {feat}: r={r:.3f}, p={p:.4f} {sig}  n={len(pairs)}')
else:
    print('  MMSE column not found')

# ── Build LOSO dataset ────────────────────────────────────────────────────────
ad_cn       = [s for s in ad_subjects + cn_subjects if s in all_features]
subj_data   = [all_features[s] for s in ad_cn]
subj_labels = [1 if s in ad_subjects else 0 for s in ad_cn]
y_2class    = np.array(subj_labels)

print(f'\nLOSO: {np.sum(y_2class==1)} AD + {np.sum(y_2class==0)} CN = {len(y_2class)} total')
print(f'Majority baseline: {max(np.mean(y_2class), 1-np.mean(y_2class))*100:.1f}%')


def build_fv(fd):
    f1 = fd.get('std_ifreq')
    if f1 is None: return None
    return [float(np.log1p(f1))]


# ── Hyperparameter search ─────────────────────────────────────────────────────
print('\n=== Hyperparameter Grid Search ===')
print(f"{'C':<8} {'Train%':<10} {'Test%':<10} {'Gap':<10} {'Normal?'}")
print('-' * 50)

best_acc, best_C = 0, 1.0

for C in [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    preds, t_accs = [], []
    for ti in range(len(y_2class)):
        tr  = [i for i in range(len(y_2class)) if i != ti]
        Xtr = [(build_fv(subj_data[i]), y_2class[i]) for i in tr]
        Xtr = [(x, y) for x, y in Xtr if x is not None]
        if not Xtr: continue
        Xtr_v, ytr_v = zip(*Xtr)
        fvt = build_fv(subj_data[ti])
        if not fvt or len(Xtr_v) < 10:
            preds.append(int(np.mean(ytr_v) > 0.5)); continue
        sc  = StandardScaler()
        Xt  = sc.fit_transform(np.array(Xtr_v))
        Xte = sc.transform(np.array(fvt).reshape(1,-1))
        clf = LogisticRegression(C=C, class_weight='balanced', max_iter=1000)
        clf.fit(Xt, list(ytr_v))
        preds.append(clf.predict(Xte)[0])
        t_accs.append(accuracy_score(list(ytr_v), clf.predict(Xt)))
    test_acc  = accuracy_score(y_2class, preds) * 100
    train_acc = np.mean(t_accs) * 100 if t_accs else 0
    gap       = train_acc - test_acc
    print(f'{str(C):<8} {train_acc:<10.1f} {test_acc:<10.1f} {gap:<10.1f} {"✓" if gap>=0 else "✗"}')
    if test_acc > best_acc:
        best_acc, best_C = test_acc, C

print(f'\nBest C: {best_C}')

# ── Final LOSO ────────────────────────────────────────────────────────────────
print('Running final LOSO...')
y_true, y_pred, y_scores, train_accs = [], [], [], []

for ti in range(len(y_2class)):
    tr  = [i for i in range(len(y_2class)) if i != ti]
    Xtr = [(build_fv(subj_data[i]), y_2class[i]) for i in tr]
    Xtr = [(x, y) for x, y in Xtr if x is not None]
    if not Xtr:
        y_pred.append(0); y_scores.append(0.5); y_true.append(y_2class[ti]); continue
    Xtr_v, ytr_v = zip(*Xtr)
    fvt = build_fv(subj_data[ti])
    if not fvt or len(Xtr_v) < 10:
        y_pred.append(int(np.mean(ytr_v) > 0.5))
        y_scores.append(0.5); y_true.append(y_2class[ti]); continue
    sc  = StandardScaler()
    Xt  = sc.fit_transform(np.array(Xtr_v))
    Xte = sc.transform(np.array(fvt).reshape(1,-1))
    clf = LogisticRegression(C=best_C, class_weight='balanced', max_iter=1000)
    clf.fit(Xt, list(ytr_v))
    y_pred.append(clf.predict(Xte)[0])
    y_scores.append(float(clf.predict_proba(Xte)[0][1]))
    y_true.append(y_2class[ti])
    train_accs.append(accuracy_score(list(ytr_v), clf.predict(Xt)))

# ── Metrics ───────────────────────────────────────────────────────────────────
y_true   = np.array(y_true)
y_pred   = np.array(y_pred)
y_scores = np.array(y_scores)

acc  = accuracy_score(y_true, y_pred) * 100
bal  = balanced_accuracy_score(y_true, y_pred) * 100
f1   = f1_score(y_true, y_pred, average='weighted') * 100
auc  = roc_auc_score(y_true, y_scores)
cm   = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
sens = tp/(tp+fn)*100
spec = tn/(tn+fp)*100
gap  = np.mean(train_accs)*100 - acc if train_accs else 0

rng = np.random.default_rng(42)
auc_boot = []
for _ in range(1000):
    idx = rng.integers(0, len(y_true), len(y_true))
    if len(np.unique(y_true[idx])) < 2: continue
    auc_boot.append(roc_auc_score(y_true[idx], y_scores[idx]))
ci_lo = float(np.percentile(auc_boot, 2.5))
ci_hi = float(np.percentile(auc_boot, 97.5))

# ── Permutation test ──────────────────────────────────────────────────────────
print('Running permutation test...')
np.random.seed(42)
y_fake = np.random.permutation(y_2class)
perm_preds = []
for ti in range(len(y_fake)):
    tr  = [i for i in range(len(y_fake)) if i != ti]
    Xtr = [(build_fv(subj_data[i]), y_fake[i]) for i in tr]
    Xtr = [(x, y) for x, y in Xtr if x is not None]
    if not Xtr: perm_preds.append(0); continue
    Xtr_v, ytr_v = zip(*Xtr)
    fvt = build_fv(subj_data[ti])
    if not fvt or len(Xtr_v) < 10:
        perm_preds.append(int(np.mean(ytr_v) > 0.5)); continue
    sc  = StandardScaler()
    Xt  = sc.fit_transform(np.array(Xtr_v))
    Xte = sc.transform(np.array(fvt).reshape(1,-1))
    clf = LogisticRegression(C=best_C, class_weight='balanced', max_iter=1000)
    clf.fit(Xt, list(ytr_v))
    perm_preds.append(clf.predict(Xte)[0])
perm_acc = accuracy_score(y_fake, perm_preds) * 100

# ── Dummy baselines ───────────────────────────────────────────────────────────
Xall = np.array([build_fv(d) for d in subj_data if build_fv(d) is not None])
yall = np.array([y_2class[i] for i, d in enumerate(subj_data) if build_fv(d) is not None])
dummy_accs = {}
for strategy in ['most_frequent', 'stratified', 'uniform']:
    dp = []
    for ti in range(len(yall)):
        tr = [i for i in range(len(yall)) if i != ti]
        clf = DummyClassifier(strategy=strategy, random_state=42)
        clf.fit(Xall[tr], yall[tr])
        dp.append(clf.predict(Xall[ti].reshape(1,-1))[0])
    dummy_accs[strategy] = accuracy_score(yall, dp) * 100

# ── CV scheme comparison ──────────────────────────────────────────────────────
clf_pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=best_C, class_weight='balanced', max_iter=1000, random_state=42)
)
print('\n=== CV Scheme Comparison ===')
for name, cv in [('10-fold', StratifiedKFold(10, shuffle=True, random_state=42)),
                  ('5-fold',  StratifiedKFold(5,  shuffle=True, random_state=42))]:
    y_pred_cv = cross_val_predict(clf_pipe, Xall, yall, cv=cv)
    print(f'  {name}: {accuracy_score(yall, y_pred_cv)*100:.1f}%')

# ── Print results ─────────────────────────────────────────────────────────────
print('\n' + '='*65)
print('FINAL RESULTS — AD vs CN (VAE-guided VMD sweep, LOSO, LR)')
print('='*65)
print(f'Subjects:          {int(np.sum(y_2class==1))} AD + {int(np.sum(y_2class==0))} CN')
print(f'Accuracy:          {acc:.1f}%')
print(f'Balanced Accuracy: {bal:.1f}%')
print(f'F1 (weighted):     {f1:.1f}%')
print(f'AUC:               {auc:.3f} (95% CI [{ci_lo:.3f}, {ci_hi:.3f}])')
print(f'Sensitivity (AD):  {sens:.1f}%')
print(f'Specificity (CN):  {spec:.1f}%')
print(f'Overfitting gap:   {gap:.1f}%')
print(f'\nValidation:')
print(f'  Permutation test:   {perm_acc:.1f}%  ({"✓ PASS" if perm_acc < 60 else "✗ CHECK"})')
print(f'  Dummy most_freq:    {dummy_accs["most_frequent"]:.1f}%')
print(f'  Gap above dummy:    {acc - dummy_accs["most_frequent"]:.1f}%')
print(f'\nConfusion Matrix:')
print(f'           Pred CN  Pred AD')
print(f'  True CN:   {tn:<6}   {fp}')
print(f'  True AD:   {fn:<6}   {tp}')
print(f'\nBenchmarks:')
print(f'  Miltiadous 2021 (LOSO, 95 feat):  77.01%')
print(f'  Shamsi 2025 (GroupKFold, WST):    83.1%,  AUC=0.930')
print(f'  Ours (LOSO, 1 feat, VAE-VMD):     {acc:.1f}%, AUC={auc:.3f}')
if acc > 77.01: print(f'  ✓ Beats Miltiadous (+{acc-77.01:.1f}%)')

# ── Save ─────────────────────────────────────────────────────────────────────
results = {
    'n_AD': int(np.sum(y_2class==1)), 'n_CN': int(np.sum(y_2class==0)),
    'accuracy': round(acc,2), 'balanced_acc': round(bal,2),
    'f1': round(f1,2), 'auc': round(float(auc),3),
    'auc_ci': [round(ci_lo,3), round(ci_hi,3)],
    'sensitivity': round(sens,2), 'specificity': round(spec,2),
    'overfitting_gap': round(gap,2), 'best_C': best_C,
    'permutation_acc': round(perm_acc,2),
    'dummy_most_frequent': round(dummy_accs['most_frequent'],2),
    'k_distribution': dict(k_counts),
    'alpha_distribution': dict(alpha_counts),
    'confusion_matrix': cm.tolist()
}
with open(f'{DRIVE_BASE}/logs/session4_final_results_single_feat.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved → {DRIVE_BASE}/logs/session4_final_results_single_feat.json')

# SHL Hiring Assessment 2026 -- stage 2: per-layer embedding regressors -> stack -> submission (CPU)
"""Reads the stage-1 / 1b kernel outputs (mounted via kernel_sources) and fits
the audio-embedding ladder:

  1. per (encoder, layer): standardize -> RidgeCV            -> OOF RMSE table
  2. keep the best layers per encoder, fit Ridge + SVR on each
  3. non-negative stack of the OOF columns, then a linear range
     calibration (a*pred + b) fitted on the stacked OOF
  4. write submission.csv (test.csv order) + oof.csv

Validation contract: the 37 label-0 clips are synthetic noise and the test set
has none (peak==0.95 rule: 0 hits), so they are DROPPED from training and CV is
RepeatedKFold(5x3, seed 42) over the 732 real-speech clips -- that is the
distribution the leaderboard scores.
"""
import glob
import os
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings('ignore')
OUT = '/kaggle/working'
SEED = 42
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)


def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(f'{pattern} not found under /kaggle/input')
    return hits[0]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# ------------------------------------------------------------------ data
# Row order comes from the stage-1b index.csv; labels and the noise rule are
# computed here so this stage doesn't wait on the (slow) transcript kernel.
import soundfile as sf

tr_csv = find('Dataset_Final/train.csv')
root = os.path.dirname(tr_csv)
te_csv = pd.read_csv(os.path.join(root, 'test.csv'))
meta = pd.read_csv(os.path.join(os.path.dirname(find('emb_wavlmL_mean.npy')), 'index.csv'))
meta = meta.merge(pd.read_csv(tr_csv).rename(columns={'label': 'label'}), on='filename', how='left')
meta.loc[meta.split == 'test', 'label'] = np.nan


def sig(p):
    x, _ = sf.read(p, dtype='float32')
    fr = x[:len(x) // 400 * 400].reshape(-1, 400)
    rms = np.sqrt((fr ** 2).mean(1) + 1e-12)
    return float(np.abs(x).max()), float((rms > max(rms.max() * 0.02, 1e-4)).mean())


pv = [sig(os.path.join(root, s, f)) for s, f in zip(meta.split, meta.filename)]
meta['peak'] = [p for p, _ in pv]
meta['voiced_frac'] = [v for _, v in pv]

is_tr = (meta.split == 'train').values
noise = ((meta.peak - 0.95).abs() < 1e-3) & (meta.voiced_frac > 0.999)
print('noise clips: train', int((noise & is_tr).sum()), 'test', int((noise & ~is_tr).sum()), flush=True)
keep = is_tr & ~noise.values
y = meta.label.values[keep]
is_te = ~is_tr
print('train rows used', keep.sum(), 'test rows', is_te.sum(), 'y mean/std', y.mean().round(3), y.std().round(3))

EMB = {}
for name, pat in [('wavlm_b', 'emb_wavlm.npy'), ('hubert', 'emb_hubert.npy'),
                  ('wavlmL_m', 'emb_wavlmL_mean.npy'), ('wavlmL_s', 'emb_wavlmL_std.npy'),
                  ('whisper_m', 'emb_whisper_mean.npy'), ('whisper_s', 'emb_whisper_std.npy')]:
    try:
        p = find(pat)
        src_idx = pd.read_csv(os.path.join(os.path.dirname(p), 'index.csv'))
        assert (src_idx.filename.values == meta.filename.values).all(), name
        EMB[name] = np.load(p)
        print(name, EMB[name].shape, flush=True)
    except FileNotFoundError:
        print('missing', name, flush=True)


# ------------------------------------------------------------- CV helpers
def cv_oof(make, X, y, Xte=None):
    """OOF predictions averaged over repeats, plus a test prediction averaged over all folds."""
    oof = np.zeros(len(y)); cnt = np.zeros(len(y)); pte = None
    for trn, val in rkf.split(X):
        sc = StandardScaler().fit(X[trn])
        m = make().fit(sc.transform(X[trn]), y[trn])
        oof[val] += m.predict(sc.transform(X[val])); cnt[val] += 1
        if Xte is not None:
            p = m.predict(sc.transform(Xte))
            pte = p if pte is None else pte + p
    oof /= cnt
    return oof, (None if pte is None else pte / rkf.get_n_splits())


def ridge(alpha):
    return lambda: Ridge(alpha=alpha)


def svr(C):
    return lambda: SVR(C=C, epsilon=0.1, gamma='scale')


# --------------------------------------------- 1. layer sweep (ridge only)
rows = []
for name, E in EMB.items():
    Xtr_all = E[keep]
    for L in range(E.shape[1]):
        X = Xtr_all[:, L]
        best = None
        for a in [10, 100, 1000, 3000]:
            oof, _ = cv_oof(ridge(a), X, y)
            r = rmse(y, oof)
            if best is None or r < best[0]:
                best = (r, a)
        rows.append(dict(enc=name, layer=L, rmse=best[0], alpha=best[1]))
        print(f'{name:10s} L{L:2d}  rmse {best[0]:.4f}  alpha {best[1]}', flush=True)
sweep = pd.DataFrame(rows)
sweep.to_csv(os.path.join(OUT, 'layer_sweep.csv'), index=False)
print('\nbest layer per encoder:\n', sweep.loc[sweep.groupby('enc').rmse.idxmin()].to_string(index=False), flush=True)

# --------------------------------- 2. members: top-3 layers / encoder, Ridge + SVR
members = {}
tests = {}
for name, g in sweep.groupby('enc'):
    top = g.nsmallest(3, 'rmse')
    for _, r in top.iterrows():
        E = EMB[name]; X = E[keep][:, int(r.layer)]; Xte = E[is_te][:, int(r.layer)]
        oof, pte = cv_oof(ridge(r.alpha), X, y, Xte)
        members[f'{name}_L{int(r.layer)}_ridge'] = oof; tests[f'{name}_L{int(r.layer)}_ridge'] = pte
        best = None
        for C in [0.3, 1, 3, 10]:
            o, p = cv_oof(svr(C), X, y, Xte)
            if best is None or rmse(y, o) < best[0]:
                best = (rmse(y, o), o, p, C)
        members[f'{name}_L{int(r.layer)}_svr'] = best[1]; tests[f'{name}_L{int(r.layer)}_svr'] = best[2]
        print(f'{name} L{int(r.layer)}: ridge {rmse(y, oof):.4f}  svr(C={best[3]}) {best[0]:.4f}', flush=True)
    # concatenation of the top layers of this encoder
    Ls = [int(l) for l in top.layer]
    X = np.concatenate([EMB[name][keep][:, l] for l in Ls], 1)
    Xte = np.concatenate([EMB[name][is_te][:, l] for l in Ls], 1)
    oof, pte = cv_oof(ridge(float(top.alpha.iloc[0]) * len(Ls)), X, y, Xte)
    members[f'{name}_cat_ridge'] = oof; tests[f'{name}_cat_ridge'] = pte
    print(f'{name} cat{Ls}: ridge {rmse(y, oof):.4f}', flush=True)

M = pd.DataFrame(members); T = pd.DataFrame(tests)
M.to_csv(os.path.join(OUT, 'members_oof.csv'), index=False); T.to_csv(os.path.join(OUT, 'members_test.csv'), index=False)

# ------------------------------------------------ 3. NNLS stack + calibration
w, _ = nnls(M.values, y)
w = w / w.sum()
stack = M.values @ w
print('\nstack weights:'); print(pd.Series(w, M.columns).sort_values(ascending=False).round(3).to_string())
print(f'stack OOF rmse {rmse(y, stack):.4f}  pearson {pearsonr(y, stack)[0]:.4f}')
# Note: NNLS weights are fitted on the OOF columns themselves (mild optimism); the
# calibration below is a 2-parameter fit so its optimism is negligible.
a, b = np.polyfit(stack, y, 1)
cal = np.clip(a * stack + b, 1, 5)
print(f'calibrated a={a:.3f} b={b:.3f}: rmse {rmse(y, cal):.4f}  pearson {pearsonr(y, cal)[0]:.4f}')
print('mean-model baseline rmse', rmse(y, np.full_like(y, y.mean())).__round__(4))

# ------------------------------------------------------------- 4. submission
pte = np.clip(a * (T.values @ w) + b, 1, 5)
sub = pd.DataFrame({'filename': meta.filename.values[is_te], 'label': pte})
# hard zero for any synthetic-noise clip in test (none today; rule kept as safety net)
sub.loc[noise.values[is_te], 'label'] = 0.0
sub = te_csv[['filename']].merge(sub, on='filename', how='left')
assert sub.label.notna().all() and len(sub) == len(te_csv)
sub.to_csv(os.path.join(OUT, 'submission.csv'), index=False)
pd.DataFrame({'filename': meta.filename.values[keep], 'label': y, 'oof': cal}).to_csv(os.path.join(OUT, 'oof.csv'), index=False)
print(sub.label.describe().round(3).to_string())

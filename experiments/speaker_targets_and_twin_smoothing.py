# SHL Hiring Assessment 2026 -- stage 11: speaker-aware training variants, zero GPU (CPU, ~10 min)
"""The label is a per-speaker property (same-voice twins differ by 0.25 on average
vs 0.44 for random pairs) and the test set has 46 clips with a >0.9-cosine twin
inside the test set. Three zero-cost ideas, each measured with the SAME
speaker-grouped folds as v5 (same clustering, same seeds):

  base   : v5's members (fixed to the layers v5 picked, so no 35-min sweep)
  denoise: training targets replaced by their speaker-cluster mean (within the
           training fold only) -> less rater noise in y
  spkavg : training rows collapsed to one averaged row per speaker cluster
  smooth : at prediction time, shrink each clip's prediction toward the mean of
           its voice twins (val fold twins in CV, test twins at inference)

Outputs: submission.csv for the best variant (only if it beats base), oof.csv, report.
"""
import glob
import os
import warnings

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.optimize import nnls
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings('ignore')
OUT = '/kaggle/working'
SEEDS = [42, 101, 2024]
GROUP_TH = 0.30
TWIN_TH = 0.20                   # tighter: cosine > 0.8 -> "same voice"
LAYERS = {'wavlmL_m': [(21, 100), (20, 100)], 'wavlmL_s': [(21, 100), (22, 100)],
          'whisper_m': [(32, 1000), (30, 1000)], 'whisper_s': [(32, 1000), (31, 1000)],
          'roberta': [(20, 3000), (21, 3000)], 'qwen': [(17, 3000), (18, 3000)]}
SVR_C = {'wavlmL_m': 10, 'wavlmL_s': 10, 'whisper_m': 10, 'whisper_s': 10, 'roberta': 1, 'qwen': 1}


def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(f'{pattern} not found under /kaggle/input')
    return hits[0]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# ------------------------------------------------------------------ data
tr_csv = find('Dataset_Final/train.csv')
root = os.path.dirname(tr_csv)
te_csv = pd.read_csv(os.path.join(root, 'test.csv'))
meta = pd.read_csv(os.path.join(os.path.dirname(find('emb_wavlmL_mean.npy')), 'index.csv'))
meta = meta.merge(pd.read_csv(tr_csv), on='filename', how='left')
EMB = {'wavlmL_m': np.load(find('emb_wavlmL_mean.npy'), mmap_mode='r'),
       'wavlmL_s': np.load(find('emb_wavlmL_std.npy'), mmap_mode='r'),
       'whisper_m': np.load(find('emb_whisper_mean.npy'), mmap_mode='r'),
       'whisper_s': np.load(find('emb_whisper_std.npy'), mmap_mode='r'),
       'roberta': np.load(find('emb_roberta.npy'), mmap_mode='r'),
       'qwen': np.load(find('emb_qwen.npy'), mmap_mode='r')}


def sig(p):
    x, _ = sf.read(p, dtype='float32')
    fr = x[:len(x) // 400 * 400].reshape(-1, 400)
    rms = np.sqrt((fr ** 2).mean(1) + 1e-12)
    return abs(float(np.abs(x).max()) - 0.95) < 1e-3 and float((rms > max(rms.max() * 0.02, 1e-4)).mean()) > 0.999


noise = np.array([sig(os.path.join(root, s, f)) for s, f in zip(meta.split, meta.filename)])
is_tr = (meta.split == 'train').values
is_te = ~is_tr
keep = is_tr & ~noise
y = meta.label.values[keep]
print('kept', keep.sum(), 'test', is_te.sum(), 'test noise', int(noise[is_te].sum()))

# speaker structure from WavLM-large layer 4 (voice, not content)
V = np.array(EMB['wavlmL_m'][:, 4]).astype(np.float64)
V -= V[keep].mean(0)
V /= np.linalg.norm(V, axis=1, keepdims=True)
groups = AgglomerativeClustering(n_clusters=None, distance_threshold=GROUP_TH, metric='cosine',
                                 linkage='average').fit_predict(V[keep])
twins_tr = AgglomerativeClustering(n_clusters=None, distance_threshold=TWIN_TH, metric='cosine',
                                   linkage='average').fit_predict(V[keep])
twins_te = AgglomerativeClustering(n_clusters=None, distance_threshold=TWIN_TH, metric='cosine',
                                   linkage='average').fit_predict(V[is_te])
print('groups', len(set(groups)), '| train twin clusters', len(set(twins_tr)),
      '| test twin clusters', len(set(twins_te)), 'multi-clip test clusters',
      int((pd.Series(twins_te).value_counts() > 1).sum()))
ybin = np.round(y * 2).astype(int)
SPLITS = [s for seed in SEEDS
          for s in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(y, ybin, groups)]


def group_mean(vals, g):
    return pd.Series(vals).groupby(g).transform('mean').values


def fit_predict(make, X, ytr, Xval, Xte, mode, gtr):
    if mode == 'spkavg':
        df = pd.DataFrame(X); df['g'] = gtr
        Xf = df.groupby('g').mean().values
        yf = pd.Series(ytr).groupby(gtr).mean().values
    else:
        Xf, yf = X, (group_mean(ytr, gtr) if mode == 'denoise' else ytr)
    sc = StandardScaler().fit(Xf)
    m = make().fit(sc.transform(Xf), yf)
    return m.predict(sc.transform(Xval)), m.predict(sc.transform(Xte))


def members(mode):
    M, T = {}, {}
    for name, specs in LAYERS.items():
        for L, alpha in specs:
            X = np.array(EMB[name][keep, L]); Xte = np.array(EMB[name][is_te, L])
            for tag, make in [('ridge', lambda a=alpha: Ridge(alpha=a)),
                              ('svr', lambda C=SVR_C[name]: SVR(C=C, epsilon=0.1, gamma='scale'))]:
                oof = np.zeros(len(y)); cnt = np.zeros(len(y)); pte = 0
                for trn, val in SPLITS:
                    pv, pt = fit_predict(make, X[trn], y[trn], X[val], Xte, mode, twins_tr[trn])
                    oof[val] += pv; cnt[val] += 1; pte = pte + pt
                M[f'{name}_L{L}_{tag}'] = oof / cnt; T[f'{name}_L{L}_{tag}'] = pte / len(SPLITS)
    return pd.DataFrame(M), pd.DataFrame(T)


def stack(M, T, smooth=0.0):
    nest = np.zeros(len(y))
    for trn, val in SPLITS[:5]:
        w_, _ = nnls(M.values[trn], y[trn]); w_ /= w_.sum()
        a_, b_ = np.polyfit(M.values[trn] @ w_, y[trn], 1)
        p = a_ * (M.values[val] @ w_) + b_
        if smooth:
            p = (1 - smooth) * p + smooth * group_mean(p, twins_tr[val])
        nest[val] = np.clip(p, 1, 5)
    w, _ = nnls(M.values, y); w /= w.sum()
    a, b = np.polyfit(M.values @ w, y, 1)
    pte = a * (T.values @ w) + b
    if smooth:
        pte = (1 - smooth) * pte + smooth * group_mean(pte, twins_te)
    return rmse(y, nest), np.clip(pte, 1, 5), nest


results = {}
for mode in ['base', 'denoise', 'spkavg']:
    M, T = members(mode)
    best_single = M.apply(lambda c: rmse(y, c)).min()
    for smooth in [0.0, 0.3, 0.5, 0.7]:
        r, pte, nest = stack(M, T, smooth)
        results[(mode, smooth)] = (r, pte, nest)
        print(f'{mode:8s} smooth={smooth:.1f}  best single {best_single:.4f}  nested stack rmse {r:.4f}', flush=True)

best = min(results, key=lambda k: results[k][0])
r, pte, nest = results[best]
base_r = results[('base', 0.0)][0]
print(f'\nBEST {best}: {r:.4f}  (base {base_r:.4f}, delta {r - base_r:+.4f})')
sub = pd.DataFrame({'filename': meta.filename.values[is_te], 'label': pte})
sub = te_csv[['filename']].merge(sub, on='filename', how='left')
assert sub.label.notna().all()
sub.to_csv(os.path.join(OUT, 'submission.csv'), index=False)
pd.DataFrame({'filename': meta.filename.values[keep], 'label': y, 'group': groups,
              'twin': twins_tr, 'oof_nested': nest}).to_csv(os.path.join(OUT, 'oof.csv'), index=False)
print(sub.label.describe().round(3).to_string())

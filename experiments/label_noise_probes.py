# SHL Hiring Assessment 2026 -- stage 20: LB hypothesis probes, no GPU (CPU, ~20 min)
"""Residual analysis showed (a) train labels carry ~0.33 RMSE of rater noise (voice twins
differ by 0.47 std) while the LB top is 0.306 -> test labels are probably cleaner;
(b) true-2.0 clips are 28 % of our squared error (bias +0.6). CV cannot reward fixes
for (a) because it scores against the noisy labels, so these variants go to the LB:

  S0 reference   : v17 recipe (members on raw labels, NNLS, linear cal, twin smoothing, +0.2 v3-FT)
  S1/S2 scale    : S0 with the spread around the mean scaled x0.9 / x1.1
  S3 denoised    : same members refit on speaker-twin-mean targets (within fold), same stack
  S4 nonlinear   : HistGradientBoosting stacker on the member OOFs (can bend the low end)
  S5 S3+S4       : denoised members + nonlinear stacker
Every variant's nested grouped CV is printed for the record.
"""
import glob
import os
import warnings

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.optimize import nnls
from scipy.stats import pearsonr
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings('ignore')
OUT = '/kaggle/working'
SEEDS = [42, 101, 2024]
GROUP_TH, TWIN_TH, SMOOTH = 0.30, 0.20, 0.5
QL = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
BLOCKS = {'wavlmL_m': [21, 20], 'wavlmL_s': [21, 22], 'whisper_m': [32, 30], 'whisper_s': [32, 31],
          'roberta': [20, 21], 'qwen': [17, 18], 'qwen7_whisper': [18, 16], 'qwen7_ctc': [18, 20], 'roberta_ctc': [19, 18],
          'q7p_ctc': [20, 22], 'q7p_whisper': [14, 16]}
SVR_C = {k: (10 if k.startswith(('wavlm', 'whisper')) else 1) for k in BLOCKS}
ALPHAS = {k: ([300, 3000, 30000] if SVR_C[k] == 1 else [100, 300, 1000, 3000]) for k in BLOCKS}


def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(f'{pattern} not found under /kaggle/input')
    return hits[0]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


tr_csv = find('Dataset_Final/train.csv'); root = os.path.dirname(tr_csv)
train = pd.read_csv(tr_csv); test = pd.read_csv(os.path.join(root, 'test.csv'))
idx = pd.read_csv(os.path.join(os.path.dirname(find('emb_wavlmL_mean.npy')), 'index.csv'))
assert (idx.filename[idx.split == 'train'].values == train.filename.values).all()
E = {k: np.load(find(f), mmap_mode='r') for k, f in
     [('wavlmL_m', 'emb_wavlmL_mean.npy'), ('wavlmL_s', 'emb_wavlmL_std.npy'), ('whisper_m', 'emb_whisper_mean.npy'),
      ('whisper_s', 'emb_whisper_std.npy'), ('roberta', 'emb_roberta.npy'), ('qwen', 'emb_qwen.npy'),
      ('qwen7_whisper', 'emb_qwen7_whisper.npy'), ('qwen7_ctc', 'emb_qwen7_ctc.npy'), ('roberta_ctc', 'emb_roberta_ctc.npy'),
      ('q7p_ctc', 'emb_q7p_ctc.npy'), ('q7p_whisper', 'emb_q7p_whisper.npy')]}
col = lambda name, L: (QL.index(L) if name.startswith(('qwen7', 'q7p')) else L)  # noqa: E731


def is_noise(p):
    x, _ = sf.read(p, dtype='float32'); fr = x[:len(x) // 400 * 400].reshape(-1, 400)
    rms = np.sqrt((fr ** 2).mean(1) + 1e-12)
    return abs(float(np.abs(x).max()) - 0.95) < 1e-3 and float((rms > max(rms.max() * .02, 1e-4)).mean()) > 0.999


noise_tr = np.array([is_noise(os.path.join(root, 'train', f)) for f in train.filename])
IS_TR = (idx.split == 'train').values; IS_TE = ~IS_TR
KEEP = IS_TR.copy(); KEEP[IS_TR] = ~noise_tr
y = train.label.values[~noise_tr].astype(float)
V = np.array(E['wavlmL_m'][:, 4], dtype=np.float64); V -= V[KEEP].mean(0); V /= np.linalg.norm(V, axis=1, keepdims=True)
cl = lambda th, m: AgglomerativeClustering(n_clusters=None, distance_threshold=th, metric='cosine', linkage='average').fit_predict(V[m])  # noqa: E731
groups, twins_tr, twins_te = cl(GROUP_TH, KEEP), cl(TWIN_TH, KEEP), cl(TWIN_TH, IS_TE)
ybin = np.round(y * 2).astype(int)
SPLITS = [s for seed in SEEDS for s in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(y, ybin, groups)]
gmean = lambda v, g: pd.Series(v).groupby(g).transform('mean').values  # noqa: E731
y_den = np.zeros(len(y))                         # per-fold denoised targets are computed inside cv_fit
ft2 = pd.read_csv(find('oof_ft2.csv')).set_index('filename').loc[train.filename[~noise_tr]].oof.values
ft2_te = pd.read_csv(find('test_ft2.csv')).pred.values; ft_te = pd.read_csv(find('test_ft.csv')).pred.values
print('kept', len(y), 'groups', len(set(groups)), 'twin clusters', len(set(twins_tr)), flush=True)


def cv_fit(make, X, Xte, denoise):
    oof = np.zeros(len(y)); cnt = np.zeros(len(y)); pte = 0
    for trn, val in SPLITS:
        yt = gmean(y[trn], twins_tr[trn]) if denoise else y[trn]
        sc = StandardScaler().fit(X[trn]); m = make().fit(sc.transform(X[trn]), yt)
        oof[val] += m.predict(sc.transform(X[val])); cnt[val] += 1; pte = pte + m.predict(sc.transform(Xte))
    return oof / cnt, pte / len(SPLITS)


def members(denoise):
    M, T = {}, {}
    for name, layers in BLOCKS.items():
        for L in layers:
            X = np.array(E[name][KEEP, col(name, L)], dtype=np.float32); Xte = np.array(E[name][IS_TE, col(name, L)], dtype=np.float32)
            best = min(((rmse(y, o), o, p) for a in ALPHAS[name] for o, p in [cv_fit(lambda a=a: Ridge(alpha=a), X, Xte, denoise)]), key=lambda t: t[0])
            M[f'{name}_L{L}_ridge'], T[f'{name}_L{L}_ridge'] = best[1], best[2]
            M[f'{name}_L{L}_svr'], T[f'{name}_L{L}_svr'] = cv_fit(lambda: SVR(C=SVR_C[name], epsilon=0.1), X, Xte, denoise)
    M['ft_whisper_lora'], T['ft_whisper_lora'] = ft2, ft2_te
    print(f'members({"denoised" if denoise else "raw"}) best single {min(rmse(y, v) for v in M.values()):.4f}', flush=True)
    return pd.DataFrame(M), pd.DataFrame(T)


def stack(M, T, nonlinear=False):
    Mv, Tv = M.values, T.values; nest = np.zeros(len(y))
    mk = lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_depth=3, min_samples_leaf=20, l2_regularization=1.0)  # noqa: E731
    for trn, val in SPLITS[:5]:
        if nonlinear:
            p = mk().fit(Mv[trn], y[trn]).predict(Mv[val])
        else:
            w_, _ = nnls(Mv[trn], y[trn]); w_ /= w_.sum(); a_, b_ = np.polyfit(Mv[trn] @ w_, y[trn], 1); p = a_ * (Mv[val] @ w_) + b_
        nest[val] = np.clip((1 - SMOOTH) * p + SMOOTH * gmean(p, twins_tr[val]), 1, 5)
    if nonlinear:
        pte = mk().fit(Mv, y).predict(Tv)
    else:
        w, _ = nnls(Mv, y); w /= w.sum(); a, b = np.polyfit(Mv @ w, y, 1); pte = a * (Tv @ w) + b
    pte = np.clip((1 - SMOOTH) * pte + SMOOTH * gmean(pte, twins_te), 1, 5)
    return nest, np.clip(0.8 * pte + 0.2 * ft_te, 1, 5)


def report(name, nest):
    d = pd.DataFrame({'y': y, 'r': nest - y}); lo = d[d.y == 2.0].r
    print(f'{name:14s} nested rmse {rmse(y, nest):.4f}  pearson {pearsonr(y, nest)[0]:.4f}  | true-2.0 bias {lo.mean():+.3f} rmse {np.sqrt((lo ** 2).mean()):.3f}', flush=True)


M_raw, T_raw = members(False)
M_den, T_den = members(True)
subs = {}
n0, p0 = stack(M_raw, T_raw); report('S0 reference', n0); subs['S0_reference'] = p0
mu = p0.mean()
subs['S1_scale0.9'] = np.clip(mu + 0.9 * (p0 - mu), 1, 5); subs['S2_scale1.1'] = np.clip(mu + 1.1 * (p0 - mu), 1, 5)
n3, p3 = stack(M_den, T_den); report('S3 denoised', n3); subs['S3_denoised'] = p3
n4, p4 = stack(M_raw, T_raw, True); report('S4 nonlinear', n4); subs['S4_nonlinear'] = p4
n5, p5 = stack(M_den, T_den, True); report('S5 den+nonlin', n5); subs['S5_den_nonlin'] = p5
for k, p in subs.items():
    pd.DataFrame({'filename': test.filename, 'label': p}).to_csv(os.path.join(OUT, f'sub_{k}.csv'), index=False)
print(pd.DataFrame(subs).corr().round(4).to_string())
print(pd.DataFrame(subs).describe().round(3).T[['mean', 'std', 'min', 'max']].to_string())

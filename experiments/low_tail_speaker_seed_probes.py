# SHL Hiring Assessment 2026 -- stage 23: low-tail fix, speaker regress-out, more seeds (CPU, ~45 min)
"""Follow-ups to the v20 residual analysis on the adopted recipe (denoised targets, x1.1 spread = LB 0.3485).
  (1) true-2.0 clips carry 28 % of the squared error with bias +0.6; the symmetric x1.1 spread stretched both
      tails equally. Probes: asymmetric spread (low tail stretched more) and a "low clip" classifier whose
      probability enters the final linear calibration (fully nested, so CV can judge it).
  (2) twin residual corr 0.57 -> errors are speaker-shaped. Probe: residualise the audio blocks on the top-k
      principal components of the speaker vector (WavLM-L layer-4 mean) before fitting the members.
  (3) variance: members averaged over 6 fold seeds instead of 3.
Variants (all on denoised targets, NNLS + linear cal + twin smoothing + 0.2 v3-FT):
  R   reference (= v20 S3)                       R_x1.1 must reproduce out20/sub_S3_denoised_x1.1
  A1..A3 asymmetric spread of R: (lo, hi) = (1.2, 1.05) (1.3, 1.0) (1.2, 1.1)
  C   low-tail classifier in the calibration, then x1.1 and (1.2, 1.05)
  K8 / K32  speaker PCs regressed out of the audio blocks, then x1.1
  S6  six fold seeds, then x1.1
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings('ignore')
OUT = '/kaggle/working'
SEEDS3, SEEDS6 = [42, 101, 2024], [42, 101, 2024, 7, 777, 31337]
GROUP_TH, TWIN_TH, SMOOTH, LOW = 0.30, 0.20, 0.5, 2.0
QL = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
BLOCKS = {'wavlmL_m': [21, 20], 'wavlmL_s': [21, 22], 'whisper_m': [32, 30], 'whisper_s': [32, 31],
          'roberta': [20, 21], 'qwen': [17, 18], 'qwen7_whisper': [18, 16], 'qwen7_ctc': [18, 20], 'roberta_ctc': [19, 18],
          'q7p_ctc': [20, 22], 'q7p_whisper': [14, 16]}
AUDIO = ('wavlmL_m', 'wavlmL_s', 'whisper_m', 'whisper_s')
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
splits_for = lambda seeds: [s for seed in seeds for s in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(y, ybin, groups)]  # noqa: E731
SPLITS = splits_for(SEEDS3)
gmean = lambda v, g: pd.Series(v).groupby(g).transform('mean').values  # noqa: E731
ft2 = pd.read_csv(find('oof_ft2.csv')).set_index('filename').loc[train.filename[~noise_tr]].oof.values
ft2_te = pd.read_csv(find('test_ft2.csv')).pred.values; ft_te = pd.read_csv(find('test_ft.csv')).pred.values
print('kept', len(y), 'groups', len(set(groups)), 'twin clusters', len(set(twins_tr)), 'low (<=2.0)', int((y <= LOW).sum()), flush=True)

# speaker subspace: PCs of the (unit-norm, centred) speaker vector over train+test rows (unsupervised, label-free)
ROWS = KEEP | IS_TE
Zs = {k: PCA(k, random_state=0).fit_transform(V[ROWS]) for k in (8, 32)}


def regress_out(X_all, Z):
    """remove the part of X (rows = train-kept then test, in ROWS order) explained linearly by the speaker PCs"""
    Z1 = np.c_[Z, np.ones(len(Z))]
    beta, *_ = np.linalg.lstsq(Z1, X_all, rcond=None)
    return X_all - Z1 @ beta


def cv_fit(make, X, Xte, splits):
    oof = np.zeros(len(y)); cnt = np.zeros(len(y)); pte = 0
    for trn, val in splits:
        yt = gmean(y[trn], twins_tr[trn])            # denoised (speaker-twin-mean) targets, within fold
        sc = StandardScaler().fit(X[trn]); m = make().fit(sc.transform(X[trn]), yt)
        oof[val] += m.predict(sc.transform(X[val])); cnt[val] += 1; pte = pte + m.predict(sc.transform(Xte))
    return oof / cnt, pte / len(splits)


def members(splits=SPLITS, speaker_k=0, tag=''):
    M, T = {}, {}
    for name, layers in BLOCKS.items():
        for L in layers:
            X = np.array(E[name][KEEP, col(name, L)], dtype=np.float32); Xte = np.array(E[name][IS_TE, col(name, L)], dtype=np.float32)
            if speaker_k and name in AUDIO:
                R = regress_out(np.r_[X, Xte].astype(np.float64), Zs[speaker_k]).astype(np.float32)
                X, Xte = R[:len(X)], R[len(X):]
            best = min(((rmse(y, o), o, p) for a in ALPHAS[name] for o, p in [cv_fit(lambda a=a: Ridge(alpha=a), X, Xte, splits)]), key=lambda t: t[0])
            M[f'{name}_L{L}_ridge'], T[f'{name}_L{L}_ridge'] = best[1], best[2]
            M[f'{name}_L{L}_svr'], T[f'{name}_L{L}_svr'] = cv_fit(lambda: SVR(C=SVR_C[name], epsilon=0.1), X, Xte, splits)
    M['ft_whisper_lora'], T['ft_whisper_lora'] = ft2, ft2_te
    aud = {k: rmse(y, v) for k, v in M.items() if k.startswith(AUDIO)}
    print(f'members{tag}: best single {min(rmse(y, v) for v in M.values()):.4f}  best audio {min(aud.values()):.4f} ({min(aud, key=aud.get)})', flush=True)
    return pd.DataFrame(M), pd.DataFrame(T)


def stack(M, T, low_clf=False):
    """NNLS + linear calibration (+ optional nested low-clip probability in the calibration), twin smoothing, +0.2 v3-FT"""
    Mv, Tv = M.values, T.values; nest = np.zeros(len(y)); islow = (y <= LOW).astype(int)

    def fit(trn):
        w_, _ = nnls(Mv[trn], y[trn]); w_ /= w_.sum(); p_ = Mv[trn] @ w_
        if low_clf:
            sc_ = StandardScaler().fit(Mv[trn]); lr_ = LogisticRegression(C=0.1, max_iter=2000).fit(sc_.transform(Mv[trn]), islow[trn])
            F_ = np.c_[p_, lr_.predict_proba(sc_.transform(Mv[trn]))[:, 1], np.ones(len(trn))]
            beta_, *_ = np.linalg.lstsq(F_, y[trn], rcond=None)
            return lambda Q: np.c_[Q @ w_, lr_.predict_proba(sc_.transform(Q))[:, 1], np.ones(len(Q))] @ beta_, beta_
        a_, b_ = np.polyfit(p_, y[trn], 1)
        return lambda Q: a_ * (Q @ w_) + b_, (a_, b_)

    for trn, val in SPLITS[:5]:
        f, _ = fit(trn); p = f(Mv[val])
        nest[val] = np.clip((1 - SMOOTH) * p + SMOOTH * gmean(p, twins_tr[val]), 1, 5)
    f, beta = fit(np.arange(len(y))); pte = f(Tv)
    pte = np.clip((1 - SMOOTH) * pte + SMOOTH * gmean(pte, twins_te), 1, 5)
    if low_clf:
        print(f'   low-clf calibration: y = {beta[0]:.3f}*p + {beta[1]:+.3f}*P(low) + {beta[2]:.3f}', flush=True)
    return nest, np.clip(0.8 * pte + 0.2 * ft_te, 1, 5)


def spread(p, lo, hi):
    mu = p.mean(); d = p - mu
    return np.clip(mu + np.where(d < 0, lo, hi) * d, 1, 5)


def report(name, nest):
    d = pd.DataFrame({'y': y, 'r': nest - y}); lo = d[d.y == 2.0].r; hi = d[d.y >= 4.5].r
    print(f'{name:12s} nested rmse {rmse(y, nest):.4f}  pearson {pearsonr(y, nest)[0]:.4f}  | true-2.0 bias {lo.mean():+.3f} rmse {np.sqrt((lo ** 2).mean()):.3f}'
          f' | >=4.5 bias {hi.mean():+.3f}', flush=True)


subs = {}
M_r, T_r = members(tag='(ref)')
n, p = stack(M_r, T_r); report('R', n); subs['R'] = p; subs['R_x1.1'] = spread(p, 1.1, 1.1)
ref = pd.read_csv(find('sub_S3_denoised.csv')).label.values
print(f'reproduction check vs v20 S3_denoised: max |diff| {np.abs(subs["R"] - ref).max():.4f}', flush=True)
for tag, lo, hi in [('A1', 1.2, 1.05), ('A2', 1.3, 1.0), ('A3', 1.2, 1.1)]:
    subs[f'{tag}_lo{lo}_hi{hi}'] = spread(p, lo, hi)
# nested-CV view of the asymmetric spread (CV scores against noisy labels, so this is for the record, not the decision)
mu_n = n.mean()
for lo, hi in [(1.1, 1.1), (1.2, 1.05), (1.3, 1.0), (1.2, 1.1)]:
    print(f'   nested rmse of spread({lo},{hi}) on R: {rmse(y, np.clip(mu_n + np.where(n < mu_n, lo, hi) * (n - mu_n), 1, 5)):.4f}', flush=True)

n, p = stack(M_r, T_r, low_clf=True); report('C lowclf', n); subs['C_lowclf'] = p
subs['C_lowclf_x1.1'] = spread(p, 1.1, 1.1); subs['C_lowclf_lo1.2_hi1.05'] = spread(p, 1.2, 1.05)

for k in (8, 32):
    M_k, T_k = members(speaker_k=k, tag=f'(speaker-k{k})')
    n, p = stack(M_k, T_k); report(f'K{k}', n); subs[f'K{k}'] = p; subs[f'K{k}_x1.1'] = spread(p, 1.1, 1.1)

M_6, T_6 = members(splits=splits_for(SEEDS6), tag='(6 seeds)')
n, p = stack(M_6, T_6); report('S6', n); subs['S6'] = p; subs['S6_x1.1'] = spread(p, 1.1, 1.1)

for k, v in subs.items():
    pd.DataFrame({'filename': test.filename, 'label': v}).to_csv(os.path.join(OUT, f'sub_{k}.csv'), index=False)
M_r.to_csv(os.path.join(OUT, 'members_oof_ref.csv'), index=False); T_r.to_csv(os.path.join(OUT, 'members_test_ref.csv'), index=False)
D = pd.DataFrame(subs)
print(D.corr().round(4).loc[:, ['R_x1.1']].T.to_string())
print(D.describe().round(3).T[['mean', 'std', 'min', 'max']].to_string())

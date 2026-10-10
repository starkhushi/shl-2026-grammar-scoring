# SHL Hiring Assessment 2026 -- stage 13: full stack = v5 members + segment-level audio + Qwen-7B/CTC text
"""Speaker-grouped CV identical to v5/v11 (same clustering, same seeds). Members:

  v5 block      : WavLM-L / Whisper-enc / RoBERTa-L / Qwen-1.5B at the layers v5 chose
  segment block : ridge + SVR trained on 10-s SEGMENT rows (label = clip label),
                  clip prediction = mean over its segments  [v12 seg_*.npy]
  text7 block   : Qwen2.5-7B hidden states on Whisper text and on the verbatim CTC
                  text; RoBERTa-L on CTC text. Top-2 layers per family by grouped CV.

Every block is reported alone and the stack is reported with/without each block,
so a block is kept only if the nested grouped CV improves.
MODE / SMOOTH come from the v11 experiment ('base' | 'denoise', twin-smoothing weight).
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings('ignore')
OUT = '/kaggle/working'
SEEDS = [42, 101, 2024]
GROUP_TH, TWIN_TH = 0.30, 0.20
MODE = 'base'          # v11: speaker-mean targets (+0.004) and speaker-averaged rows (+0.010) both hurt
SMOOTH = 0.5           # v11: twin smoothing 0.5 -> grouped CV 0.5213 -> 0.5157
V5 = {'wavlmL_m': [(21, 100), (20, 100)], 'wavlmL_s': [(21, 100), (22, 100)],
      'whisper_m': [(32, 1000), (30, 1000)], 'whisper_s': [(32, 1000), (31, 1000)],
      'roberta': [(20, 3000), (21, 3000)], 'qwen': [(17, 3000), (18, 3000)]}
SVR_C = {'wavlmL_m': 10, 'wavlmL_s': 10, 'whisper_m': 10, 'whisper_s': 10, 'roberta': 1, 'qwen': 1,
         'seg_wavlm': 10, 'seg_whisper': 10, 'qwen7_whisper': 1, 'qwen7_ctc': 1, 'roberta_ctc': 1}
WL = [19, 20, 21, 22, 23]
WHL = [28, 29, 30, 31, 32]
QL = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]


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
for p in ['emb_roberta.npy', 'seg_wavlm.npy']:
    assert (pd.read_csv(os.path.join(os.path.dirname(find(p)), 'index.csv')).filename.values == meta.filename.values).all(), p
EMB = {'wavlmL_m': np.load(find('emb_wavlmL_mean.npy'), mmap_mode='r'),
       'wavlmL_s': np.load(find('emb_wavlmL_std.npy'), mmap_mode='r'),
       'whisper_m': np.load(find('emb_whisper_mean.npy'), mmap_mode='r'),
       'whisper_s': np.load(find('emb_whisper_std.npy'), mmap_mode='r'),
       'roberta': np.load(find('emb_roberta.npy'), mmap_mode='r'),
       'qwen': np.load(find('emb_qwen.npy'), mmap_mode='r'),
       'qwen7_whisper': np.load(find('emb_qwen7_whisper.npy'), mmap_mode='r'),
       'qwen7_ctc': np.load(find('emb_qwen7_ctc.npy'), mmap_mode='r'),
       'roberta_ctc': np.load(find('emb_roberta_ctc.npy'), mmap_mode='r')}
SEG = {'seg_wavlm': np.load(find('seg_wavlm.npy'), mmap_mode='r'),
       'seg_whisper': np.load(find('seg_whisper.npy'), mmap_mode='r')}
NSEG = np.load(find('seg_nseg.npy'))
for k, v in {**EMB, **SEG}.items():
    print(k, v.shape)


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
KI = np.where(keep)[0]; TI = np.where(is_te)[0]
print('kept', keep.sum(), 'test', is_te.sum(), 'nseg dist', np.bincount(NSEG).tolist())

V = np.array(EMB['wavlmL_m'][:, 4]).astype(np.float64)
V -= V[keep].mean(0); V /= np.linalg.norm(V, axis=1, keepdims=True)
cl = lambda th, m: AgglomerativeClustering(n_clusters=None, distance_threshold=th, metric='cosine',  # noqa: E731
                                           linkage='average').fit_predict(V[m])
groups, twins_tr, twins_te = cl(GROUP_TH, keep), cl(TWIN_TH, keep), cl(TWIN_TH, is_te)
ybin = np.round(y * 2).astype(int)
SPLITS = [s for seed in SEEDS
          for s in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(y, ybin, groups)]


def gmean(vals, g):
    return pd.Series(vals).groupby(g).transform('mean').values


def target(trn):
    return gmean(y[trn], twins_tr[trn]) if MODE == 'denoise' else y[trn]


# ------------------------------------------------------- clip-level members
def cv_clip(make, X, Xte):
    oof = np.zeros(len(y)); cnt = np.zeros(len(y)); pte = 0
    for trn, val in SPLITS:
        sc = StandardScaler().fit(X[trn])
        m = make().fit(sc.transform(X[trn]), target(trn))
        oof[val] += m.predict(sc.transform(X[val])); cnt[val] += 1
        pte = pte + m.predict(sc.transform(Xte))
    return oof / cnt, pte / len(SPLITS)


# ---------------------------------------------------- segment-level members
def seg_rows(E, li, rows):
    """Stack valid segments of the given clips -> (X, owner index into rows)."""
    Xs, own = [], []
    for j, i in enumerate(rows):
        n = NSEG[i]
        if n == 0:
            continue
        Xs.append(np.asarray(E[i, :n, li], dtype=np.float32)); own.extend([j] * n)
    return np.concatenate(Xs), np.array(own)


def clip_mean(pred, own, n):
    out = np.full(n, np.nan)
    s = pd.Series(pred).groupby(own).mean()
    out[s.index.values] = s.values
    return np.where(np.isnan(out), np.nanmean(out), out)


def cv_seg(make, E, li):
    Xte, ote = seg_rows(E, li, TI)
    oof = np.zeros(len(y)); cnt = np.zeros(len(y)); pte = 0
    for trn, val in SPLITS:
        Xtr, otr = seg_rows(E, li, KI[trn]); ytr = target(trn)[otr]
        Xva, ova = seg_rows(E, li, KI[val])
        sc = StandardScaler().fit(Xtr)
        m = make().fit(sc.transform(Xtr), ytr)
        oof[val] += clip_mean(m.predict(sc.transform(Xva)), ova, len(val)); cnt[val] += 1
        pte = pte + clip_mean(m.predict(sc.transform(Xte)), ote, len(TI))
    return oof / cnt, pte / len(SPLITS)


M, T = {}, {}
# v5 block
for name, specs in V5.items():
    for L, alpha in specs:
        X = np.array(EMB[name][keep, L], dtype=np.float32); Xte = np.array(EMB[name][is_te, L], dtype=np.float32)
        M[f'{name}_L{L}_ridge'], T[f'{name}_L{L}_ridge'] = cv_clip(lambda a=alpha: Ridge(alpha=a), X, Xte)
        M[f'{name}_L{L}_svr'], T[f'{name}_L{L}_svr'] = cv_clip(lambda C=SVR_C[name]: SVR(C=C, epsilon=0.1), X, Xte)
print('v5 block done; best', min(rmse(y, v) for v in M.values()).__round__(4), flush=True)

# text7 block: layer sweep (ridge) then ridge+svr on the top-2
for name in ['qwen7_whisper', 'qwen7_ctc', 'roberta_ctc']:
    E = EMB[name]; res = []
    for li in range(E.shape[1]):
        X = np.array(E[keep, li], dtype=np.float32)
        best = min((rmse(y, cv_clip(lambda a=a: Ridge(alpha=a), X, X[:1])[0]), a) for a in [300, 3000, 30000])
        res.append((best[0], li, best[1]))
    res.sort()
    lab = QL if name.startswith('qwen7') else list(range(E.shape[1]))
    print(f'{name}: layer sweep best {[(lab[li], round(r, 4)) for r, li, _ in res[:3]]}', flush=True)
    for r, li, alpha in res[:2]:
        X = np.array(E[keep, li], dtype=np.float32); Xte = np.array(E[is_te, li], dtype=np.float32)
        M[f'{name}_L{lab[li]}_ridge'], T[f'{name}_L{lab[li]}_ridge'] = cv_clip(lambda a=alpha: Ridge(alpha=a), X, Xte)
        M[f'{name}_L{lab[li]}_svr'], T[f'{name}_L{lab[li]}_svr'] = cv_clip(lambda C=SVR_C[name]: SVR(C=C, epsilon=0.1), X, Xte)

# segment block
for name, lab in [('seg_wavlm', WL), ('seg_whisper', WHL)]:
    E = SEG[name]; res = []
    for li in range(E.shape[2]):
        best = min((rmse(y, cv_seg(lambda a=a: Ridge(alpha=a), E, li)[0]), a) for a in [100, 1000])
        res.append((best[0], li, best[1]))
    res.sort()
    print(f'{name}: segment-ridge sweep {[(lab[li], round(r, 4)) for r, li, _ in res]}', flush=True)
    for r, li, alpha in res[:2]:
        M[f'{name}_L{lab[li]}_ridge'], T[f'{name}_L{lab[li]}_ridge'] = cv_seg(lambda a=alpha: Ridge(alpha=a), E, li)
        M[f'{name}_L{lab[li]}_svr'], T[f'{name}_L{lab[li]}_svr'] = cv_seg(lambda C=SVR_C[name]: SVR(C=C, epsilon=0.1), E, li)
M, T = pd.DataFrame(M), pd.DataFrame(T)
M.to_csv(os.path.join(OUT, 'members_oof.csv'), index=False); T.to_csv(os.path.join(OUT, 'members_test.csv'), index=False)
print('\nmember grouped-CV rmse:'); print(M.apply(lambda c: rmse(y, c)).sort_values().round(4).to_string())


# ------------------------------------------------------- stacks & ablation
def stack(cols):
    Mv, Tv = M[cols].values, T[cols].values
    nest = np.zeros(len(y))
    for trn, val in SPLITS[:5]:
        w_, _ = nnls(Mv[trn], y[trn]); w_ /= w_.sum()
        a_, b_ = np.polyfit(Mv[trn] @ w_, y[trn], 1)
        p = a_ * (Mv[val] @ w_) + b_
        if SMOOTH:
            p = (1 - SMOOTH) * p + SMOOTH * gmean(p, twins_tr[val])
        nest[val] = np.clip(p, 1, 5)
    w, _ = nnls(Mv, y); w /= w.sum()
    a, b = np.polyfit(Mv @ w, y, 1)
    pte = a * (Tv @ w) + b
    if SMOOTH:
        pte = (1 - SMOOTH) * pte + SMOOTH * gmean(pte, twins_te)
    return rmse(y, nest), pearsonr(y, nest)[0], np.clip(pte, 1, 5), nest, pd.Series(w, cols)


blk = lambda *pre: [c for c in M.columns if c.split('_L')[0] in pre]  # noqa: E731
v5c = blk(*V5.keys()); segc = blk('seg_wavlm', 'seg_whisper'); t7c = blk('qwen7_whisper', 'qwen7_ctc', 'roberta_ctc')
cands = {'v5 only': v5c, 'v5+seg': v5c + segc, 'v5+text7': v5c + t7c, 'all': v5c + segc + t7c,
         'seg+text7 (no v5)': segc + t7c}
res = {k: stack(c) for k, c in cands.items()}
print()
for k, (r, pr, *_) in res.items():
    print(f'{k:20s} nested rmse {r:.4f}  pearson {pr:.4f}')
best = min(res, key=lambda k: res[k][0])
r, pr, pte, nest, w = res[best]
print(f'\nBEST: {best}  ({r:.4f})'); print(w[w > 0].sort_values(ascending=False).round(3).to_string())
sub = te_csv[['filename']].merge(pd.DataFrame({'filename': meta.filename.values[is_te], 'label': pte}), on='filename', how='left')
assert sub.label.notna().all() and len(sub) == len(te_csv)
sub.to_csv(os.path.join(OUT, 'submission.csv'), index=False)
pd.DataFrame({'filename': meta.filename.values[keep], 'label': y, 'group': groups, 'oof_nested': nest}).to_csv(os.path.join(OUT, 'oof.csv'), index=False)
print(sub.label.describe().round(3).to_string())

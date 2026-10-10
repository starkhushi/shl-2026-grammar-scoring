# SHL Hiring Assessment 2026 -- stage 15: v13 members + v14 blocks, ablation under speaker-grouped CV (CPU, ~10 min)
"""Reuses v13's saved out-of-fold / test member predictions (same folds) and adds
the v14 blocks, each swept for its best layers then ridge+SVR:

  q7p      : Qwen-7B prompt-pooled (last token) on whisper / ctc / verbatim text
  q7verb   : Qwen-7B mean-pooled on the disfluency-prompted verbatim transcript
  w2vbert  : w2v-BERT 2.0 per-layer mean / std audio embeddings

Ablation: v13 alone, +each block, all. Twin smoothing 0.5 (v11). The winner is
written to submission.csv; a 0.8/0.2 blend with the WavLM fine-tune (v3) is
written to submission_ft.csv.
"""
import glob
import os
import warnings

import numpy as np
import pandas as pd
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
GROUP_TH, TWIN_TH, SMOOTH = 0.30, 0.20, 0.5
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
v13 = os.path.dirname(find('members_oof.csv'))
M13 = pd.read_csv(os.path.join(v13, 'members_oof.csv')); T13 = pd.read_csv(os.path.join(v13, 'members_test.csv'))
oof13 = pd.read_csv(os.path.join(v13, 'oof.csv'))
y = oof13.label.values; groups = oof13.group.values
meta = pd.read_csv(os.path.join(os.path.dirname(find('emb_wavlmL_mean.npy')), 'index.csv'))
keep = ((meta.split == 'train') & meta.filename.isin(oof13.filename)).values   # test reuses train filenames
assert (meta.filename.values[keep] == oof13.filename.values).all()
is_te = (meta.split == 'test').values
assert (meta.filename.values[is_te] == te_csv.filename.values).all()
v14 = os.path.dirname(find('emb_w2vbert_mean.npy'))
assert (pd.read_csv(os.path.join(v14, 'index.csv')).filename.values == meta.filename.values).all()
EMB = {'q7p_whisper': np.load(find('emb_q7p_whisper.npy'), mmap_mode='r'),
       'q7p_ctc': np.load(find('emb_q7p_ctc.npy'), mmap_mode='r'),
       'q7p_verbatim': np.load(find('emb_q7p_verbatim.npy'), mmap_mode='r'),
       'qwen7_verbatim': np.load(find('emb_qwen7_verbatim.npy'), mmap_mode='r'),
       'w2vbert_m': np.load(find('emb_w2vbert_mean.npy'), mmap_mode='r'),
       'w2vbert_s': np.load(find('emb_w2vbert_std.npy'), mmap_mode='r')}
LAB = {k: (QL if k.startswith('q') else list(range(25))) for k in EMB}
SVR_C = {k: (1 if k.startswith('q') else 10) for k in EMB}
for k, v in EMB.items():
    print(k, v.shape, 'nan', int(np.isnan(np.asarray(v[:, 0], dtype=np.float32)).sum()))

V = np.array(np.load(find('emb_wavlmL_mean.npy'), mmap_mode='r')[:, 4]).astype(np.float64)
V -= V[keep].mean(0); V /= np.linalg.norm(V, axis=1, keepdims=True)
cl = lambda th, m: AgglomerativeClustering(n_clusters=None, distance_threshold=th, metric='cosine',  # noqa: E731
                                           linkage='average').fit_predict(V[m])
g_check, twins_tr, twins_te = cl(GROUP_TH, keep), cl(TWIN_TH, keep), cl(TWIN_TH, is_te)
assert len(set(g_check)) == len(set(groups)), 'speaker clustering changed'
ybin = np.round(y * 2).astype(int)
SPLITS = [s for seed in SEEDS
          for s in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(y, ybin, groups)]


def gmean(vals, g):
    return pd.Series(vals).groupby(g).transform('mean').values


def cv_clip(make, X, Xte):
    oof = np.zeros(len(y)); cnt = np.zeros(len(y)); pte = 0
    for trn, val in SPLITS:
        sc = StandardScaler().fit(X[trn])
        m = make().fit(sc.transform(X[trn]), y[trn])
        oof[val] += m.predict(sc.transform(X[val])); cnt[val] += 1
        pte = pte + m.predict(sc.transform(Xte))
    return oof / cnt, pte / len(SPLITS)


M, T = {}, {}
for name, E in EMB.items():
    res = []
    for li in range(E.shape[1]):
        X = np.nan_to_num(np.array(E[keep, li], dtype=np.float32))
        best = min((rmse(y, cv_clip(lambda a=a: Ridge(alpha=a), X, X[:1])[0]), a)
                   for a in ([300, 3000, 30000] if name.startswith('q') else [30, 300, 3000]))
        res.append((best[0], li, best[1]))
    res.sort()
    print(f'{name:15s} sweep best {[(LAB[name][li], round(r, 4)) for r, li, _ in res[:3]]}', flush=True)
    for r, li, alpha in res[:2]:
        X = np.nan_to_num(np.array(E[keep, li], dtype=np.float32)); Xte = np.nan_to_num(np.array(E[is_te, li], dtype=np.float32))
        k = f'{name}_L{LAB[name][li]}'
        M[k + '_ridge'], T[k + '_ridge'] = cv_clip(lambda a=alpha: Ridge(alpha=a), X, Xte)
        M[k + '_svr'], T[k + '_svr'] = cv_clip(lambda C=SVR_C[name]: SVR(C=C, epsilon=0.1), X, Xte)
M, T = pd.DataFrame(M), pd.DataFrame(T)
print('\nnew member grouped-CV rmse:'); print(M.apply(lambda c: rmse(y, c)).sort_values().round(4).to_string())
MA, TA = pd.concat([M13, M], axis=1), pd.concat([T13, T], axis=1)
MA.to_csv(os.path.join(OUT, 'members_oof.csv'), index=False); TA.to_csv(os.path.join(OUT, 'members_test.csv'), index=False)


def stack(cols):
    Mv, Tv = MA[cols].values, TA[cols].values
    nest = np.zeros(len(y))
    for trn, val in SPLITS[:5]:
        w_, _ = nnls(Mv[trn], y[trn]); w_ /= w_.sum()
        a_, b_ = np.polyfit(Mv[trn] @ w_, y[trn], 1)
        p = a_ * (Mv[val] @ w_) + b_
        p = (1 - SMOOTH) * p + SMOOTH * gmean(p, twins_tr[val])
        nest[val] = np.clip(p, 1, 5)
    w, _ = nnls(Mv, y); w /= w.sum()
    a, b = np.polyfit(Mv @ w, y, 1)
    pte = a * (Tv @ w) + b
    pte = (1 - SMOOTH) * pte + SMOOTH * gmean(pte, twins_te)
    return rmse(y, nest), pearsonr(y, nest)[0], np.clip(pte, 1, 5), nest, pd.Series(w, cols)


blk = lambda *pre: [c for c in MA.columns if c.split('_L')[0] in pre]  # noqa: E731
v13c = [c for c in M13.columns if c.split('_L')[0] not in ('seg_wavlm', 'seg_whisper')]   # v13's kept set
q7p = blk('q7p_whisper', 'q7p_ctc', 'q7p_verbatim'); q7v = blk('qwen7_verbatim'); w2v = blk('w2vbert_m', 'w2vbert_s')
cands = {'v13': v13c, 'v13+q7p': v13c + q7p, 'v13+q7verb': v13c + q7v, 'v13+w2vbert': v13c + w2v,
         'v13+q7p+q7verb': v13c + q7p + q7v, 'all': v13c + q7p + q7v + w2v}
res = {k: stack(c) for k, c in cands.items()}
print()
for k, (r, pr, *_) in res.items():
    print(f'{k:18s} nested rmse {r:.4f}  pearson {pr:.4f}')
best = min(res, key=lambda k: res[k][0])
r, pr, pte, nest, w = res[best]
print(f'\nBEST: {best}  ({r:.4f}, v13 {res["v13"][0]:.4f})'); print(w[w > 0].sort_values(ascending=False).round(3).to_string())
sub = pd.DataFrame({'filename': te_csv.filename, 'label': pte})
sub.to_csv(os.path.join(OUT, 'submission.csv'), index=False)
ft = pd.read_csv(find('test_ft.csv'))
assert (ft.filename.values == te_csv.filename.values).all()
pd.DataFrame({'filename': te_csv.filename, 'label': np.clip(0.8 * pte + 0.2 * ft.pred.values, 1, 5)}).to_csv(os.path.join(OUT, 'submission_ft.csv'), index=False)
pd.DataFrame({'filename': oof13.filename, 'label': y, 'group': groups, 'oof_nested': nest}).to_csv(os.path.join(OUT, 'oof.csv'), index=False)
print(sub.label.describe().round(3).to_string())

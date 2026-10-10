# SHL Hiring Assessment 2026 -- stage 17: final stack with the Whisper-LoRA fine-tune as an honest member (CPU, ~2 min)
"""v16's folds are StratifiedGroupKFold(seed 42) over the same speaker clusters the
stacks use == SPLITS[0:5], so its OOF predictions are leak-free and can be an NNLS
member. Variants written as submissions:

  A  v15 best set (v13 + q7p)                       + 0.2 * v3-FT  (current LB best 0.3518, reference)
  B  v15 best set + v16-FT as NNLS member            (fully nested)
  C  B                                               + 0.2 * v3-FT
  D  0.8 * v15 best set + 0.2 * mean(v3-FT, v16-FT)
"""
import glob
import os
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import pearsonr
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings('ignore')
OUT = '/kaggle/working'
GROUP_TH, TWIN_TH, SMOOTH = 0.30, 0.20, 0.5


def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(f'{pattern} not found under /kaggle/input')
    return hits[0]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


tr_csv = find('Dataset_Final/train.csv'); root = os.path.dirname(tr_csv)
te_csv = pd.read_csv(os.path.join(root, 'test.csv'))
v15 = os.path.dirname(find('members_oof.csv'))
M = pd.read_csv(os.path.join(v15, 'members_oof.csv')); T = pd.read_csv(os.path.join(v15, 'members_test.csv'))
oof15 = pd.read_csv(os.path.join(v15, 'oof.csv')); y = oof15.label.values; groups = oof15.group.values
ft2 = pd.read_csv(find('oof_ft2.csv')).set_index('filename').loc[oof15.filename]
ft2_te = pd.read_csv(find('test_ft2.csv')); ft_te = pd.read_csv(find('test_ft.csv'))
assert (ft2_te.filename.values == te_csv.filename.values).all() and (ft_te.filename.values == te_csv.filename.values).all()
print('v16 FT OOF rmse', round(rmse(y, ft2.oof.values), 4))

meta = pd.read_csv(os.path.join(os.path.dirname(find('emb_wavlmL_mean.npy')), 'index.csv'))
keep = ((meta.split == 'train') & meta.filename.isin(oof15.filename)).values; is_te = (meta.split == 'test').values
V = np.array(np.load(find('emb_wavlmL_mean.npy'), mmap_mode='r')[:, 4]).astype(np.float64)
V -= V[keep].mean(0); V /= np.linalg.norm(V, axis=1, keepdims=True)
cl = lambda th, m: AgglomerativeClustering(n_clusters=None, distance_threshold=th, metric='cosine', linkage='average').fit_predict(V[m])  # noqa: E731
twins_tr, twins_te = cl(TWIN_TH, keep), cl(TWIN_TH, is_te)
ybin = np.round(y * 2).astype(int)
SPLITS = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(y, ybin, groups))
# sanity: v16 used exactly these folds -> its OOF must be leak-free w.r.t. them (val rows of fold k came from fold k's model)
gmean = lambda v, g: pd.Series(v).groupby(g).transform('mean').values  # noqa: E731


def stack(cols, Mx, Tx):
    Mv, Tv = Mx[cols].values, Tx[cols].values
    nest = np.zeros(len(y))
    for trn, val in SPLITS:
        w_, _ = nnls(Mv[trn], y[trn]); w_ /= w_.sum(); a_, b_ = np.polyfit(Mv[trn] @ w_, y[trn], 1)
        p = a_ * (Mv[val] @ w_) + b_
        nest[val] = np.clip((1 - SMOOTH) * p + SMOOTH * gmean(p, twins_tr[val]), 1, 5)
    w, _ = nnls(Mv, y); w /= w.sum(); a, b = np.polyfit(Mv @ w, y, 1)
    pte = a * (Tv @ w) + b
    pte = np.clip((1 - SMOOTH) * pte + SMOOTH * gmean(pte, twins_te), 1, 5)
    return rmse(y, nest), pearsonr(y, nest)[0], pte, pd.Series(w, cols)


base = [c for c in M.columns if c.split('_L')[0] not in ('seg_wavlm', 'seg_whisper', 'qwen7_verbatim', 'w2vbert_m', 'w2vbert_s', 'q7p_verbatim')]
base = [c for c in base if not c.startswith('q7p_verbatim')]
M2, T2 = M.copy(), T.copy(); M2['ft_whisper_lora'] = ft2.oof.values; T2['ft_whisper_lora'] = ft2_te.pred.values
rA, pA, teA, wA = stack(base, M, T)
rB, pB, teB, wB = stack(base + ['ft_whisper_lora'], M2, T2)
print(f'A  v15 set                     nested rmse {rA:.4f} pearson {pA:.4f}')
print(f'B  v15 set + v16 FT member     nested rmse {rB:.4f} pearson {pB:.4f}   FT weight {wB["ft_whisper_lora"]:.3f}')
print(wB[wB > 0].sort_values(ascending=False).round(3).to_string())
subs = {'A_v15_ft3': np.clip(0.8 * teA + 0.2 * ft_te.pred.values, 1, 5),
        'B_v15_ft16member': teB,
        'C_B_plus_ft3': np.clip(0.8 * teB + 0.2 * ft_te.pred.values, 1, 5),
        'D_v15_ft3ft16': np.clip(0.8 * teA + 0.2 * (0.5 * ft_te.pred.values + 0.5 * ft2_te.pred.values), 1, 5)}
for k, p in subs.items():
    pd.DataFrame({'filename': te_csv.filename, 'label': p}).to_csv(os.path.join(OUT, f'sub_{k}.csv'), index=False)
print(pd.DataFrame(subs).corr().round(4))

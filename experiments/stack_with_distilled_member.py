# SHL Hiring Assessment 2026 -- stage 24: add the distilled Whisper fine-tune (v21) to the final stack (CPU, ~3 min)
"""v21 = Whisper-large-v3 encoder LoRA trained on 0.5*label + 0.5*ensemble-OOF targets plus the 216 test
clips as soft pseudo-labels; grouped OOF 0.5301 (v16: 0.5784, v18 7-epoch rerun: 0.5811). Its folds are
StratifiedGroupKFold(seed 42) over the same speaker clusters = SPLITS[0:5], so it can be an NNLS member.
Caveat printed for the record: the distillation targets of fold-k's training clips were produced by a teacher
that saw fold-k labels, so v21's OOF (and any nested CV that includes it) is mildly optimistic. The LB decides.

Members = v23 reference set (denoised targets, incl. v16 FT) from members_*_ref.csv. Variants, all with
NNLS + linear cal + twin smoothing, then x1.1 spread:
  D0  reference + 0.2 v3-FT                         (= LB 0.3485, reproduction check)
  D1  + v21 as NNLS member, + 0.2 v3-FT
  D2  + v21 + v18 as NNLS members, + 0.2 v3-FT
  D3  reference, outer blend 0.8*stack + 0.1 v3-FT + 0.1 v21
  D4  reference, outer blend 0.8*stack + 0.2 v21   (v21 replaces v3-FT)
  D5  D1 without the outer v3-FT blend (pure nested stack)
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
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings('ignore')
OUT = '/kaggle/working'
GROUP_TH, TWIN_TH, SMOOTH, SPREAD = 0.30, 0.20, 0.5, 1.1


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


def is_noise(p):
    x, _ = sf.read(p, dtype='float32'); fr = x[:len(x) // 400 * 400].reshape(-1, 400)
    rms = np.sqrt((fr ** 2).mean(1) + 1e-12)
    return abs(float(np.abs(x).max()) - 0.95) < 1e-3 and float((rms > max(rms.max() * .02, 1e-4)).mean()) > 0.999


noise_tr = np.array([is_noise(os.path.join(root, 'train', f)) for f in train.filename])
IS_TR = (idx.split == 'train').values; IS_TE = ~IS_TR
KEEP = IS_TR.copy(); KEEP[IS_TR] = ~noise_tr
fn = train.filename[~noise_tr].values
y = train.label.values[~noise_tr].astype(float)
V = np.array(np.load(find('emb_wavlmL_mean.npy'), mmap_mode='r')[:, 4], dtype=np.float64); V -= V[KEEP].mean(0); V /= np.linalg.norm(V, axis=1, keepdims=True)
cl = lambda th, m: AgglomerativeClustering(n_clusters=None, distance_threshold=th, metric='cosine', linkage='average').fit_predict(V[m])  # noqa: E731
groups, twins_tr, twins_te = cl(GROUP_TH, KEEP), cl(TWIN_TH, KEEP), cl(TWIN_TH, IS_TE)
ybin = np.round(y * 2).astype(int)
SPLITS = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(y, ybin, groups))
gmean = lambda v, g: pd.Series(v).groupby(g).transform('mean').values  # noqa: E731

M = pd.read_csv(find('members_oof_ref.csv')); T = pd.read_csv(find('members_test_ref.csv'))
assert len(M) == len(y) and len(T) == len(test)
ft_te = pd.read_csv(find('test_ft.csv')).pred.values                                   # v3 WavLM FT (outer blend only)
ft4 = pd.read_csv(find('oof_ft4.csv')).set_index('filename').loc[fn].oof.values; ft4_te = pd.read_csv(find('test_ft4.csv')).pred.values
ft3 = pd.read_csv(find('oof_ft3.csv')).set_index('filename').loc[fn].oof.values; ft3_te = pd.read_csv(find('test_ft3.csv')).pred.values
print(f'kept {len(y)}  members {M.shape[1]}  | OOF rmse v16 {rmse(y, M.ft_whisper_lora):.4f}  v18 {rmse(y, ft3):.4f}  v21 {rmse(y, ft4):.4f}'
      f'  | corr(v21, v16 oof) {np.corrcoef(ft4, M.ft_whisper_lora)[0, 1]:.3f}', flush=True)


def stack(Mx, Tx):
    Mv, Tv = Mx.values, Tx.values; nest = np.zeros(len(y))
    for trn, val in SPLITS:
        w_, _ = nnls(Mv[trn], y[trn]); w_ /= w_.sum(); a_, b_ = np.polyfit(Mv[trn] @ w_, y[trn], 1); p = a_ * (Mv[val] @ w_) + b_
        nest[val] = np.clip((1 - SMOOTH) * p + SMOOTH * gmean(p, twins_tr[val]), 1, 5)
    w, _ = nnls(Mv, y); w /= w.sum(); a, b = np.polyfit(Mv @ w, y, 1); pte = a * (Tv @ w) + b
    pte = np.clip((1 - SMOOTH) * pte + SMOOTH * gmean(pte, twins_te), 1, 5)
    return nest, pte, pd.Series(w, Mx.columns)


def spread(p, k=SPREAD):
    mu = p.mean(); return np.clip(mu + k * (p - mu), 1, 5)


def report(name, nest, w=None):
    extra = '' if w is None else '  weights: ' + ', '.join(f'{k}={v:.3f}' for k, v in w[w > 0.02].sort_values(ascending=False).items())
    print(f'{name:4s} nested rmse {rmse(y, nest):.4f}  pearson {pearsonr(y, nest)[0]:.4f}{extra}', flush=True)


subs = {}
n0, p0, w0 = stack(M, T); report('D0', n0, w0)
subs['D0_ref'] = spread(np.clip(0.8 * p0 + 0.2 * ft_te, 1, 5))
ref = pd.read_csv(find('sub_R_x1.1.csv')).label.values
print(f'reproduction check vs v23 R_x1.1: max |diff| {np.abs(subs["D0_ref"] - ref).max():.4f}', flush=True)

M1, T1 = M.assign(ft_distill=ft4), T.assign(ft_distill=ft4_te)
n1, p1, w1 = stack(M1, T1); report('D1', n1, w1)
subs['D1_v21member'] = spread(np.clip(0.8 * p1 + 0.2 * ft_te, 1, 5))
subs['D5_v21member_noouter'] = spread(p1)

M2, T2 = M1.assign(ft_7ep=ft3), T1.assign(ft_7ep=ft3_te)
n2, p2, w2 = stack(M2, T2); report('D2', n2, w2)
subs['D2_v21_v18members'] = spread(np.clip(0.8 * p2 + 0.2 * ft_te, 1, 5))

subs['D3_outer_half'] = spread(np.clip(0.8 * p0 + 0.1 * ft_te + 0.1 * ft4_te, 1, 5))
subs['D4_outer_v21'] = spread(np.clip(0.8 * p0 + 0.2 * ft4_te, 1, 5))
# nested view of the outer blends (v3-FT has no honest OOF, so only the v21 outer blend can be scored)
report('D4n', np.clip(0.8 * n0 + 0.2 * ft4, 1, 5))

for k, v in subs.items():
    pd.DataFrame({'filename': test.filename, 'label': v}).to_csv(os.path.join(OUT, f'sub_{k}.csv'), index=False)
pd.DataFrame({'filename': fn, 'label': y, 'nest_D0': n0, 'nest_D1': n1, 'nest_D2': n2}).to_csv(os.path.join(OUT, 'nested_oof.csv'), index=False)
D = pd.DataFrame(subs)
print(D.corr().round(4).loc[['D0_ref']].to_string())
print(D.describe().round(3).T[['mean', 'std', 'min', 'max']].to_string())

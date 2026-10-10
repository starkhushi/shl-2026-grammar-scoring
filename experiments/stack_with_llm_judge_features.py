# SHL Hiring Assessment 2026 -- stage 9: v5 stack + zero-training grammar features (CPU, minutes)
"""Reuses v5's saved out-of-fold / test predictions (members_oof.csv,
members_test.csv) instead of recomputing 170 layer sweeps, rebuilds the SAME
speaker-grouped folds from v5's oof.csv (same groups, same seeds), fits new
members on the stage-8 grader features, and re-stacks.

Each new block is reported on its own and as an add-on to v5, so it is kept
only if the grouped nested CV improves.
"""
import glob
import os
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings('ignore')
OUT = '/kaggle/working'
SEEDS = [42, 101, 2024]


def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(f'{pattern} not found under /kaggle/input')
    return hits[0]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# ------------------------------------------------------------------ data
v5_dir = os.path.dirname(find('members_oof.csv'))
M5 = pd.read_csv(os.path.join(v5_dir, 'members_oof.csv'))
T5 = pd.read_csv(os.path.join(v5_dir, 'members_test.csv'))
oof5 = pd.read_csv(os.path.join(v5_dir, 'oof.csv'))          # filename, label, group (kept train rows)
y = oof5.label.values
groups = oof5.group.values
te_csv = pd.read_csv(find('Dataset_Final/test.csv'))

J = pd.read_csv(find('judge.csv'))
Jtr = J[J.split == 'train'].set_index('filename').loc[oof5.filename]
Jte = J[J.split == 'test'].reset_index(drop=True)
assert (Jte.filename.values == te_csv.filename.values).all()
FEATS = [c for c in J.columns if c not in ('filename', 'split')]
print('grader features:', FEATS)
for c in FEATS:
    print(f'  {c:14s} r={np.corrcoef(Jtr[c].values, y)[0, 1]:+.3f}')

ybin = np.round(y * 2).astype(int)
SPLITS = [s for seed in SEEDS
          for s in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(y, ybin, groups)]


def cv_oof(make, X, Xte):
    oof = np.zeros(len(y)); cnt = np.zeros(len(y)); pte = 0
    for trn, val in SPLITS:
        sc = StandardScaler().fit(X[trn])
        m = make().fit(sc.transform(X[trn]), y[trn])
        oof[val] += m.predict(sc.transform(X[val])); cnt[val] += 1
        pte = pte + m.predict(sc.transform(Xte))
    return oof / cnt, pte / len(SPLITS)


def stack(M, T, label):
    """Nested NNLS + linear calibration (honest), then final refit on all rows."""
    nest = np.zeros(len(y))
    for trn, val in SPLITS[:5]:
        w_, _ = nnls(M.values[trn], y[trn]); w_ /= w_.sum()
        a_, b_ = np.polyfit(M.values[trn] @ w_, y[trn], 1)
        nest[val] = np.clip(a_ * (M.values[val] @ w_) + b_, 1, 5)
    w, _ = nnls(M.values, y); w /= w.sum()
    a, b = np.polyfit(M.values @ w, y, 1)
    print(f'{label:28s} nested rmse {rmse(y, nest):.4f}  pearson {pearsonr(y, nest)[0]:.4f}')
    return nest, w, a, b


# ---------------------------------------------------------- new members
blocks = {
    'llm': ['llm_score', 'llm_p1', 'llm_p2', 'llm_p3', 'llm_p4', 'llm_p5', 'llm_errors', 'llm_err_rate'],
    'cola': ['cola_mean', 'cola_min', 'cola_bad', 'cola_whole', 'n_sent', 'sent_len'],
}
blocks['judge_all'] = blocks['llm'] + blocks['cola']
new_oof, new_te = {}, {}
for name, cols in blocks.items():
    X, Xte = Jtr[cols].values.astype(float), Jte[cols].values.astype(float)
    for a in [1, 10, 100]:
        o, p = cv_oof(lambda: Ridge(alpha=a), X, Xte)
        if f'{name}_ridge' not in new_oof or rmse(y, o) < rmse(y, new_oof[f'{name}_ridge']):
            new_oof[f'{name}_ridge'], new_te[f'{name}_ridge'] = o, p
    o, p = cv_oof(lambda: SVR(C=1, epsilon=0.1), X, Xte)
    new_oof[f'{name}_svr'], new_te[f'{name}_svr'] = o, p
    print(f'{name:10s} ridge {rmse(y, new_oof[name + "_ridge"]):.4f}  svr {rmse(y, o):.4f}', flush=True)

# --------------------------------------------- compare stacks, pick the best
print()
base = stack(M5, T5, 'v5 (reproduced)')
cands = {'v5': (M5, T5)}
for add in ['llm', 'cola', 'judge_all']:
    ks = [add + '_ridge', add + '_svr']
    M = pd.concat([M5, pd.DataFrame({k: new_oof[k] for k in ks})], axis=1)
    T = pd.concat([T5, pd.DataFrame({k: new_te[k] for k in ks})], axis=1)
    cands[f'v5+{add}'] = (M, T)
res = {k: stack(M, T, k) for k, (M, T) in cands.items()}
best = min(res, key=lambda k: rmse(y, res[k][0]))
nest, w, a, b = res[best]
M, T = cands[best]
print(f'\nBEST: {best}')
print(pd.Series(w, M.columns)[w > 0].sort_values(ascending=False).round(3).to_string())

pte = np.clip(a * (T.values @ w) + b, 1, 5)
sub = pd.DataFrame({'filename': te_csv.filename, 'label': pte})
sub.to_csv(os.path.join(OUT, 'submission.csv'), index=False)
cal_train = np.clip(a * (M.values @ w) + b, 1, 5)
pd.DataFrame({'filename': oof5.filename, 'label': y, 'group': groups,
              'oof_nested': nest, 'fit': cal_train}).to_csv(os.path.join(OUT, 'oof.csv'), index=False)
M.to_csv(os.path.join(OUT, 'members_oof.csv'), index=False)
T.to_csv(os.path.join(OUT, 'members_test.csv'), index=False)
print(f'train (in-fit) rmse {rmse(y, cal_train):.4f} | grouped nested CV rmse {rmse(y, nest):.4f}')
print(sub.label.describe().round(3).to_string())

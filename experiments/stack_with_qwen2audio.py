# SHL Hiring Assessment 2026 -- stage 22: final recipe + Qwen2-Audio features (CPU, ~20 min)
"""Takes the v20 recipe (members fit on speaker-twin-mean targets, NNLS, linear cal,
twin smoothing, +0.2 WavLM-FT, spread 1.1) and tests the Qwen2-Audio-7B features from
the Colab run (dataset shl26-colab-v19):
    q2a_mean : mean-pooled hidden states, 8 layers   -> layer sweep, ridge + SVR on top-2
    q2a_last : last-token states, 8 layers          -> same
    q2a_grade: zero-shot 1-5 grade (+ its 5 probs)   -> ridge
Ablation: final / +q2a_mean / +q2a_last / +q2a_grade / all. Grouped nested CV decides;
submissions are written for the reference and for the best variant (both with spread 1.1).
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
GROUP_TH, TWIN_TH, SMOOTH, SPREAD = 0.30, 0.20, 0.5, 1.1
QL = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
Q2L = [8, 12, 16, 20, 22, 24, 28, 32]
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
      ('q7p_ctc', 'emb_q7p_ctc.npy'), ('q7p_whisper', 'emb_q7p_whisper.npy'),
      ('q2a_mean', 'emb_q2a_mean.npy'), ('q2a_last', 'emb_q2a_last.npy')]}
q2a_dir = os.path.dirname(find('emb_q2a_mean.npy'))
assert (pd.read_csv(os.path.join(q2a_dir, 'index.csv')).filename.values == idx.filename.values).all()
grade = pd.read_csv(os.path.join(q2a_dir, 'q2a_grade.csv')); assert (grade.filename.values == idx.filename.values).all()
col = lambda name, L: (QL.index(L) if name.startswith(('qwen7', 'q7p')) else (Q2L.index(L) if name.startswith('q2a') else L))  # noqa: E731


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
ft2 = pd.read_csv(find('oof_ft2.csv')).set_index('filename').loc[train.filename[~noise_tr]].oof.values
ft2_te = pd.read_csv(find('test_ft2.csv')).pred.values; ft_te = pd.read_csv(find('test_ft.csv')).pred.values
print('kept', len(y), 'groups', len(set(groups)), flush=True)


def cv_fit(make, X, Xte):
    oof = np.zeros(len(y)); cnt = np.zeros(len(y)); pte = 0
    for trn, val in SPLITS:
        yt = gmean(y[trn], twins_tr[trn])
        sc = StandardScaler().fit(X[trn]); m = make().fit(sc.transform(X[trn]), yt)
        oof[val] += m.predict(sc.transform(X[val])); cnt[val] += 1; pte = pte + m.predict(sc.transform(Xte))
    return oof / cnt, pte / len(SPLITS)


def block(name, L, alphas, C):
    X = np.nan_to_num(np.array(E[name][KEEP, col(name, L)], dtype=np.float32)); Xte = np.nan_to_num(np.array(E[name][IS_TE, col(name, L)], dtype=np.float32))
    best = min(((rmse(y, o), o, p, a) for a in alphas for o, p in [cv_fit(lambda a=a: Ridge(alpha=a), X, Xte)]), key=lambda t: t[0])
    o, p = cv_fit(lambda: SVR(C=C, epsilon=0.1), X, Xte)
    return best, (rmse(y, o), o, p)


M, T = {}, {}
for name, layers in BLOCKS.items():
    for L in layers:
        r, s_ = block(name, L, ALPHAS[name], SVR_C[name])
        M[f'{name}_L{L}_ridge'], T[f'{name}_L{L}_ridge'] = r[1], r[2]; M[f'{name}_L{L}_svr'], T[f'{name}_L{L}_svr'] = s_[1], s_[2]
M['ft_whisper_lora'], T['ft_whisper_lora'] = ft2, ft2_te
print('base members done; best single', round(min(rmse(y, v) for v in M.values()), 4), flush=True)

# --- Qwen2-Audio blocks: layer sweep then ridge+SVR on the top-2 layers
for name in ['q2a_mean', 'q2a_last']:
    res = []
    for L in Q2L:
        X = np.nan_to_num(np.array(E[name][KEEP, col(name, L)], dtype=np.float32))
        r = min(rmse(y, cv_fit(lambda a=a: Ridge(alpha=a), X, X[:1])[0]) for a in [300, 3000, 30000])
        res.append((r, L))
    res.sort(); print(f'{name} sweep: {[(L, round(r, 4)) for r, L in res]}', flush=True)
    for r, L in res[:2]:
        rr, ss = block(name, L, [300, 3000, 30000], 1)
        M[f'{name}_L{L}_ridge'], T[f'{name}_L{L}_ridge'] = rr[1], rr[2]; M[f'{name}_L{L}_svr'], T[f'{name}_L{L}_svr'] = ss[1], ss[2]
        print(f'  {name} L{L}: ridge {rr[0]:.4f} svr {ss[0]:.4f}', flush=True)
Gf = grade[['q2a_grade', 'q2a_p1', 'q2a_p2', 'q2a_p3', 'q2a_p4', 'q2a_p5']].values.astype(np.float32)
o, p = cv_fit(lambda: Ridge(alpha=10), Gf[KEEP], Gf[IS_TE]); M['q2a_grade_ridge'], T['q2a_grade_ridge'] = o, p
print(f'q2a_grade ridge {rmse(y, o):.4f}  (raw r with label {pearsonr(grade.q2a_grade.values[KEEP], y)[0]:.3f})', flush=True)
M, T = pd.DataFrame(M), pd.DataFrame(T)
M.to_csv(os.path.join(OUT, 'members_oof.csv'), index=False); T.to_csv(os.path.join(OUT, 'members_test.csv'), index=False)


def stack(cols):
    Mv, Tv = M[cols].values, T[cols].values; nest = np.zeros(len(y))
    for trn, val in SPLITS[:5]:
        w_, _ = nnls(Mv[trn], y[trn]); w_ /= w_.sum(); a_, b_ = np.polyfit(Mv[trn] @ w_, y[trn], 1); p = a_ * (Mv[val] @ w_) + b_
        nest[val] = np.clip((1 - SMOOTH) * p + SMOOTH * gmean(p, twins_tr[val]), 1, 5)
    w, _ = nnls(Mv, y); w /= w.sum(); a, b = np.polyfit(Mv @ w, y, 1); pte = a * (Tv @ w) + b
    pte = np.clip((1 - SMOOTH) * pte + SMOOTH * gmean(pte, twins_te), 1, 5)
    pred = 0.8 * pte + 0.2 * ft_te; pred = np.clip(pred.mean() + SPREAD * (pred - pred.mean()), 1, 5)
    return rmse(y, nest), pearsonr(y, nest)[0], pred, pd.Series(w, cols)


base = [c for c in M.columns if not c.startswith('q2a')]
qm = [c for c in M.columns if c.startswith('q2a_mean')]; qlst = [c for c in M.columns if c.startswith('q2a_last')]; qg = ['q2a_grade_ridge']
cands = {'final (reference)': base, '+q2a_mean': base + qm, '+q2a_last': base + qlst, '+q2a_grade': base + qg, '+all q2a': base + qm + qlst + qg}
res = {k: stack(c) for k, c in cands.items()}
print()
for k, (r, pr, *_) in res.items():
    print(f'{k:20s} nested rmse {r:.4f}  pearson {pr:.4f}')
best = min(res, key=lambda k: res[k][0])
print(f'\nBEST: {best}'); w = res[best][3]; print(w[w > 0].sort_values(ascending=False).round(3).to_string())
pd.DataFrame({'filename': test.filename, 'label': res['final (reference)'][2]}).to_csv(os.path.join(OUT, 'sub_reference.csv'), index=False)
pd.DataFrame({'filename': test.filename, 'label': res[best][2]}).to_csv(os.path.join(OUT, 'sub_best.csv'), index=False)
pd.DataFrame({'filename': test.filename, 'label': res['+all q2a'][2]}).to_csv(os.path.join(OUT, 'sub_all_q2a.csv'), index=False)
print('corr best vs reference', np.corrcoef(res[best][2], res['final (reference)'][2])[0, 1].round(4))

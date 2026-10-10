# SHL Hiring Assessment 2026 -- stage 5: audio + text stack with SPEAKER-GROUPED CV (CPU)
"""Same ladder as v2 (per-layer ridge/SVR -> NNLS stack -> linear calibration),
with two changes driven by the data:

  * Speaker-grouped folds. Train repeats speakers (53 % of clips have a near-
    identical-voice twin, label diff 0.25) while test speakers are all new
    (0 % have a twin in train). Random folds therefore reward memorising voices.
    Speakers are found by average-linkage cosine clustering of WavLM-large
    layer-4 embeddings (threshold 0.30 -> ~370 groups); folds are
    StratifiedGroupKFold(5) x 3 seeds.
  * Text track: RoBERTa-large and Qwen2.5-1.5B layer embeddings of Whisper-large-v3
    transcripts, plus ASR log-prob / word-rate. Text can't recognise voices.

Inputs: stage-1b (audio embeddings) + stage-4 (text) kernel outputs.
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
GROUP_TH = 0.30


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
a_dir = os.path.dirname(find('emb_wavlmL_mean.npy'))
t_dir = os.path.dirname(find('emb_roberta.npy'))
meta = pd.read_csv(os.path.join(a_dir, 'index.csv'))
assert (pd.read_csv(os.path.join(t_dir, 'index.csv')).filename.values == meta.filename.values).all()
meta = meta.merge(pd.read_csv(tr_csv), on='filename', how='left')
meta.loc[meta.split == 'test', 'label'] = np.nan
text = pd.read_csv(os.path.join(t_dir, 'text.csv'))
assert (text.filename.values == meta.filename.values).all()


def sig(p):
    x, _ = sf.read(p, dtype='float32')
    fr = x[:len(x) // 400 * 400].reshape(-1, 400)
    rms = np.sqrt((fr ** 2).mean(1) + 1e-12)
    return float(np.abs(x).max()), float((rms > max(rms.max() * 0.02, 1e-4)).mean()), len(x) / 16000


pv = np.array([sig(os.path.join(root, s, f)) for s, f in zip(meta.split, meta.filename)])
meta['peak'], meta['voiced_frac'], meta['dur'] = pv[:, 0], pv[:, 1], pv[:, 2]
noise = ((meta.peak - 0.95).abs() < 1e-3) & (meta.voiced_frac > 0.999)
is_tr = (meta.split == 'train').values
is_te = ~is_tr
keep = is_tr & ~noise.values
y = meta.label.values[keep]
print('noise train', int((noise & is_tr).sum()), 'test', int((noise & is_te).sum()), '| kept', keep.sum())

EMB = {'wavlmL_m': np.load(find('emb_wavlmL_mean.npy'), mmap_mode='r'),
       'wavlmL_s': np.load(find('emb_wavlmL_std.npy'), mmap_mode='r'),
       'whisper_m': np.load(find('emb_whisper_mean.npy'), mmap_mode='r'),
       'whisper_s': np.load(find('emb_whisper_std.npy'), mmap_mode='r'),
       'roberta': np.load(find('emb_roberta.npy'), mmap_mode='r'),
       'qwen': np.load(find('emb_qwen.npy'), mmap_mode='r')}
for k, v in EMB.items():
    print(k, v.shape)

# ------------------------------------------------------------ speaker groups
Xs = np.array(EMB['wavlmL_m'][keep, 4])
Xs -= Xs.mean(0)
Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
groups = AgglomerativeClustering(n_clusters=None, distance_threshold=GROUP_TH, metric='cosine',
                                 linkage='average').fit_predict(Xs)
print('speaker groups', len(set(groups)), 'max size', pd.Series(groups).value_counts().max())
ybin = np.round(y * 2).astype(int)
SPLITS = [s for seed in SEEDS
          for s in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(Xs, ybin, groups)]


def cv_oof(make, X, Xte=None):
    oof = np.zeros(len(y)); cnt = np.zeros(len(y)); pte = 0
    for trn, val in SPLITS:
        sc = StandardScaler().fit(X[trn])
        m = make().fit(sc.transform(X[trn]), y[trn])
        oof[val] += m.predict(sc.transform(X[val])); cnt[val] += 1
        if Xte is not None:
            pte = pte + m.predict(sc.transform(Xte))
    return oof / cnt, (None if Xte is None else pte / len(SPLITS))


ridge = lambda a: (lambda: Ridge(alpha=a))                       # noqa: E731
svr = lambda C: (lambda: SVR(C=C, epsilon=0.1, gamma='scale'))   # noqa: E731

# ---------------------------------------------------- 1. layer sweep (ridge)
rows = []
for name, E in EMB.items():
    for L in range(E.shape[1]):
        X = np.array(E[keep, L])
        res = [(rmse(y, cv_oof(ridge(a), X)[0]), a) for a in [30, 300, 3000, 30000]]
        r, a = min(res)
        rows.append(dict(enc=name, layer=L, rmse=r, alpha=a))
    g = pd.DataFrame([r for r in rows if r['enc'] == name])
    print(f'{name:10s} best L{int(g.loc[g.rmse.idxmin(), "layer"])} rmse {g.rmse.min():.4f}', flush=True)
sweep = pd.DataFrame(rows)
sweep.to_csv(os.path.join(OUT, 'layer_sweep.csv'), index=False)

# ------------------------------------- 2. members: top-2 layers, ridge + SVR
members, tests = {}, {}
for name, g in sweep.groupby('enc'):
    for _, r in g.nsmallest(2, 'rmse').iterrows():
        L = int(r.layer)
        X = np.array(EMB[name][keep, L]); Xte = np.array(EMB[name][is_te, L])
        k = f'{name}_L{L}'
        members[k + '_ridge'], tests[k + '_ridge'] = cv_oof(ridge(r.alpha), X, Xte)
        best = min(((rmse(y, o), o, p, C) for C in [0.3, 1, 3, 10]
                    for o, p in [cv_oof(svr(C), X, Xte)]), key=lambda t: t[0])
        members[k + '_svr'], tests[k + '_svr'] = best[1], best[2]
        print(f'{k}: ridge {rmse(y, members[k + "_ridge"]):.4f}  svr(C={best[3]}) {best[0]:.4f}', flush=True)

# small handcrafted text block: ASR confidence + speaking rate
H = pd.DataFrame({'lp': text.asr_logprob.fillna(text.asr_logprob.mean()),
                  'nw': text.n_words, 'rate': text.n_words / meta.dur,
                  'lnw': np.log1p(text.n_words)}).values
members['hand_ridge'], tests['hand_ridge'] = cv_oof(ridge(10), H[keep], H[is_te])
print(f'hand: ridge {rmse(y, members["hand_ridge"]):.4f}')

M = pd.DataFrame(members); T = pd.DataFrame(tests)
M.to_csv(os.path.join(OUT, 'members_oof.csv'), index=False); T.to_csv(os.path.join(OUT, 'members_test.csv'), index=False)

# ------------------------------------------------ 3. NNLS stack + calibration
# nested: weights and calibration fitted on 4/5 of the OOF rows, scored on the rest,
# so the reported stack RMSE is honest; final weights refit on all rows.
nest = np.zeros(len(y))
for trn, val in SPLITS[:5]:
    w_, _ = nnls(M.values[trn], y[trn]); w_ /= w_.sum()
    a_, b_ = np.polyfit(M.values[trn] @ w_, y[trn], 1)
    nest[val] = np.clip(a_ * (M.values[val] @ w_) + b_, 1, 5)
w, _ = nnls(M.values, y); w /= w.sum()
stack = M.values @ w
a, b = np.polyfit(stack, y, 1)
cal = np.clip(a * stack + b, 1, 5)
print('\nstack weights:\n', pd.Series(w, M.columns)[w > 0].sort_values(ascending=False).round(3).to_string())
print(f'audio-only best member {M[[c for c in M if c.split("_")[0] in ("wavlmL", "whisper")]].apply(lambda c: rmse(y, c)).min():.4f}')
print(f'text-only best member  {M[[c for c in M if c.split("_")[0] in ("roberta", "qwen", "hand")]].apply(lambda c: rmse(y, c)).min():.4f}')
print(f'GROUPED stack: nested rmse {rmse(y, nest):.4f} pearson {pearsonr(y, nest)[0]:.4f} | in-fit cal a={a:.3f} b={b:.3f} rmse {rmse(y, cal):.4f}')

# ------------------------------------------------------------- 4. submission
pte = np.clip(a * (T.values @ w) + b, 1, 5)
sub = pd.DataFrame({'filename': meta.filename.values[is_te], 'label': pte})
sub.loc[noise.values[is_te], 'label'] = 0.0
sub = te_csv[['filename']].merge(sub, on='filename', how='left')
assert sub.label.notna().all() and len(sub) == len(te_csv)
sub.to_csv(os.path.join(OUT, 'submission.csv'), index=False)
pd.DataFrame({'filename': meta.filename.values[keep], 'label': y, 'group': groups,
              'oof': cal, 'oof_nested': nest}).to_csv(os.path.join(OUT, 'oof.csv'), index=False)
print(sub.label.describe().round(3).to_string())

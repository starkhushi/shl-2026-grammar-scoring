# SHL Hiring Assessment 2026 -- stage 16: LoRA fine-tune of the Whisper-large-v3 encoder, speaker-grouped folds (GPU, ~4 h)
"""Lessons from v3 (WavLM-large full fine-tune, random folds, OOF 0.585) applied:
  * backbone = Whisper-large-v3 encoder, our best frozen single (grouped CV 0.570)
  * LoRA (r=16) on the top 16 of 32 layers only + mean-pool head; everything else frozen
  * StratifiedGroupKFold over the same speaker clusters as v5/v13 -> epoch selection
    cannot be fooled by voice twins
  * 20-s random crops in training; inference averages all 30-s windows of the clip
  * OOF / test predictions are written after EVERY fold (a 12-h kill keeps the folds done)
Outputs: oof_ft2.csv, test_ft2.csv, submission.csv, fold_*.npy
"""
import glob
import math
import os
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import StratifiedGroupKFold
from transformers import WhisperFeatureExtractor, WhisperModel, get_cosine_schedule_with_warmup


class LoRALinear(nn.Module):
    """y = W x + (alpha/r) * B A x ; W frozen, A ~ N(0, 1/r), B = 0 (peft-free, no version games)."""

    def __init__(self, base: nn.Linear, r=16, alpha=32, dropout=0.1):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(r))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.t() @ self.B.t()) * self.scale


def add_lora(encoder, layers, names=('q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2')):
    n = 0
    for li in layers:
        layer = encoder.layers[li]
        for name in names:
            parent = layer.self_attn if name in ('q_proj', 'k_proj', 'v_proj', 'out_proj') else layer
            setattr(parent, name, LoRALinear(getattr(parent, name)))
            n += 1
    return n

SR = 16000
CROP_S, WIN_S = 20, 30
OUT = '/kaggle/working'
DEV = 'cuda'
EPOCHS, BS, ACC = 4, 4, 2
LR_LORA, LR_HEAD = 1e-4, 1e-3
LORA_LAYERS = list(range(16, 32))
SEED, GROUP_TH = 42, 0.30
torch.manual_seed(SEED); np.random.seed(SEED)
t0 = time.time()
print(torch.cuda.get_device_name(0), flush=True)


def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(f'{pattern} not found under /kaggle/input')
    return hits[0]


def log(m):
    print(f'[{time.time() - t0:6.0f}s] {m}', flush=True)


# ------------------------------------------------------------------ data
tr_csv = find('Dataset_Final/train.csv')
root = os.path.dirname(tr_csv)
tr = pd.read_csv(tr_csv); te = pd.read_csv(os.path.join(root, 'test.csv'))


def load(p):
    x, sr = sf.read(p, dtype='float32')
    if x.ndim > 1:
        x = x.mean(1)
    assert sr == SR
    return x


def is_noise(x):
    fr = x[:len(x) // 400 * 400].reshape(-1, 400)
    rms = np.sqrt((fr ** 2).mean(1) + 1e-12)
    return abs(float(np.abs(x).max()) - 0.95) < 1e-3 and float((rms > max(rms.max() * 0.02, 1e-4)).mean()) > 0.999


A_tr = [load(os.path.join(root, 'train', f)) for f in tr.filename]
A_te = [load(os.path.join(root, 'test', f)) for f in te.filename]
noise_tr = np.array([is_noise(x) for x in A_tr]); noise_te = np.array([is_noise(x) for x in A_te])
keep = np.where(~noise_tr)[0]
y_all = tr.label.values.astype(np.float32)
Y_MEAN = float(y_all[keep].mean())
log(f'audio loaded; noise train {noise_tr.sum()} test {noise_te.sum()}; kept {len(keep)}')

# speaker groups: same recipe as v5/v13 (WavLM-large layer-4 cosine clustering) -> same folds family
idx1b = pd.read_csv(os.path.join(os.path.dirname(find('emb_wavlmL_mean.npy')), 'index.csv'))
E4 = np.array(np.load(find('emb_wavlmL_mean.npy'), mmap_mode='r')[:, 4]).astype(np.float64)
pos = {(s, f): i for i, (s, f) in enumerate(zip(idx1b.split, idx1b.filename))}
V = E4[[pos[('train', f)] for f in tr.filename[keep]]]
V -= V.mean(0); V /= np.linalg.norm(V, axis=1, keepdims=True)
groups = AgglomerativeClustering(n_clusters=None, distance_threshold=GROUP_TH, metric='cosine',
                                 linkage='average').fit_predict(V)
ybin = np.round(y_all[keep] * 2).astype(int)
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(V, ybin, groups))
log(f'speaker groups {len(set(groups))}')

fe = WhisperFeatureExtractor.from_pretrained('openai/whisper-large-v3')


def feats(segs):
    """list of waveforms (<= 30 s) -> (B,128,3000) fp16 mel, (B,) valid frame counts."""
    f = fe(segs, sampling_rate=SR, return_tensors='pt').input_features.to(DEV, torch.float16)
    n = torch.tensor([max(int(math.ceil(len(s) / SR * 50)), 1) for s in segs], device=DEV)
    return f, n


def rand_crop(x):
    n = CROP_S * SR
    if len(x) <= n:
        return x
    s = np.random.randint(0, len(x) - n + 1)
    return x[s:s + n]


# ----------------------------------------------------------------- model
class Reg(nn.Module):
    def __init__(self):
        super().__init__()
        enc = WhisperModel.from_pretrained('openai/whisper-large-v3', torch_dtype=torch.float32).encoder
        for p in enc.parameters():
            p.requires_grad = False
        enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        add_lora(enc, LORA_LAYERS)
        self.enc = enc
        d = enc.config.d_model
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.2), nn.Linear(d, 1))

    def forward(self, mel, nvalid):
        h = self.enc(input_features=mel).last_hidden_state                 # (B,1500,D)
        ar = torch.arange(h.shape[1], device=h.device)[None]
        m = (ar < nvalid[:, None]).unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(pooled.float()).squeeze(-1) + Y_MEAN


@torch.no_grad()
def predict(model, xs):
    model.eval(); out = []
    for x in xs:
        wins = [x[s:s + WIN_S * SR] for s in range(0, len(x), WIN_S * SR)]
        wins = [w for w in wins if len(w) >= SR // 2] or [x]
        mel, n = feats(wins)
        with torch.autocast('cuda', dtype=torch.float16):
            p = model(mel, n).float()
        w = torch.tensor([len(v) for v in wins], device=DEV, dtype=torch.float32)
        out.append(float((p * w).sum() / w.sum()))
    return np.array(out)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# ------------------------------------------------------------------ CV
oof = np.full(len(tr), np.nan, dtype=np.float32)
pte = np.zeros(len(te), dtype=np.float32)
done = 0
for fold, (tri, vai) in enumerate(folds):
    tri, vai = keep[tri], keep[vai]
    model = Reg().to(DEV)
    ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([{'params': [p for p in model.enc.parameters() if p.requires_grad], 'lr': LR_LORA},
                             {'params': model.head.parameters(), 'lr': LR_HEAD}], weight_decay=0.01)
    steps = EPOCHS * math.ceil(len(tri) / (BS * ACC))
    sch = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.cuda.amp.GradScaler()
    best = (1e9, None, None, -1)
    log(f'fold {fold}: train {len(tri)} val {len(vai)} trainable params {ntrain / 1e6:.1f}M')
    for ep in range(EPOCHS):
        model.train(); perm = np.random.permutation(tri); tot = 0.0; nb = 0; opt.zero_grad()
        for bi in range(0, len(perm), BS):
            ids = perm[bi:bi + BS]
            mel, n = feats([rand_crop(A_tr[i]) for i in ids])
            yb = torch.tensor(y_all[ids], device=DEV)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = nn.functional.mse_loss(model(mel, n).float(), yb) / ACC
            scaler.scale(loss).backward(); tot += float(loss) * ACC; nb += 1
            if nb % ACC == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        pv = predict(model, [A_tr[i] for i in vai]); r = rmse(y_all[vai], pv)
        log(f'fold {fold} ep {ep}  train_mse {tot / nb:.4f}  val_rmse {r:.4f}')
        if r < best[0]:
            best = (r, pv, predict(model, A_te), ep)
    oof[vai] = best[1]; pte += best[2] / len(folds); done += 1
    log(f'== fold {fold} best ep {best[3]} rmse {best[0]:.4f}')
    np.save(os.path.join(OUT, f'fold_{fold}.npy'), np.r_[best[1], best[2]])
    m = ~np.isnan(oof)
    pd.DataFrame({'filename': tr.filename[m], 'label': y_all[m], 'oof': oof[m]}).to_csv(os.path.join(OUT, 'oof_ft2.csv'), index=False)
    pd.DataFrame({'filename': te.filename, 'pred': np.clip(pte * len(folds) / done, 1, 5)}).to_csv(os.path.join(OUT, 'test_ft2.csv'), index=False)
    del model, opt; torch.cuda.empty_cache()

m = ~np.isnan(oof)
log(f'GROUPED OOF rmse {rmse(y_all[m], oof[m]):.4f}  pearson {np.corrcoef(y_all[m], oof[m])[0, 1]:.4f} (n={m.sum()})')
pte = np.where(noise_te, 0.0, np.clip(pte, 1, 5))
pd.DataFrame({'filename': te.filename, 'label': pte}).to_csv(os.path.join(OUT, 'submission.csv'), index=False)
log('done')

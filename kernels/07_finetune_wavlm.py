# SHL Hiring Assessment 2026 -- stage 3: fine-tune WavLM-large for grammar-score regression (GPU, internet)
"""5-fold fine-tune of microsoft/wavlm-large with a mean-pool regression head.

Design choices (small data: 732 real-speech clips):
  * the 37 synthetic-noise clips (label 0) are dropped -- the test set has none
  * CNN feature extractor + lower FREEZE_LAYERS transformer layers are frozen;
    only the upper layers (where the grammar signal lives) are trained
  * training sees random CROP_S-second crops (augmentation + memory); inference
    averages the prediction over all CROP_S windows of the clip
  * MSE on the raw 1-5 scale, small LR, cosine schedule, best-epoch by fold RMSE
  * KFold(5, shuffle, seed 42) over the kept rows in file order -- same contract
    stage 2 can stack against (oof.csv carries filename)

Outputs: oof_ft.csv (filename, label, oof), test_ft.csv (filename, pred), log.
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
from sklearn.model_selection import KFold
from transformers import AutoModel, get_cosine_schedule_with_warmup

SR = 16000
CROP_S = 20
OUT = '/kaggle/working'
DEV = 'cuda'
MODEL = 'microsoft/wavlm-large'
FREEZE_LAYERS = 12          # of 24 transformer layers
EPOCHS = 8
BS = 4
ACC = 2                     # grad accumulation -> effective batch 8
LR_ENC, LR_HEAD = 2e-5, 1e-3
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
print(torch.cuda.get_device_name(0), flush=True)


def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(f'{pattern} not found under /kaggle/input')
    return hits[0]


# ------------------------------------------------------------------ data
tr_csv = find('Dataset_Final/train.csv')
root = os.path.dirname(tr_csv)
tr = pd.read_csv(tr_csv)
te = pd.read_csv(os.path.join(root, 'test.csv'))


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


print('loading audio', flush=True)
A_tr = [load(os.path.join(root, 'train', f)) for f in tr.filename]
A_te = [load(os.path.join(root, 'test', f)) for f in te.filename]
noise_tr = np.array([is_noise(x) for x in A_tr])
noise_te = np.array([is_noise(x) for x in A_te])
print('noise clips train', noise_tr.sum(), 'test', noise_te.sum(), flush=True)
keep = np.where(~noise_tr)[0]
y_all = tr.label.values.astype(np.float32)
Y_MEAN = float(y_all[keep].mean())


def norm(x):
    return (x - x.mean()) / (x.std() + 1e-5)


def rand_crop(x):
    n = CROP_S * SR
    if len(x) <= n:
        return x
    s = np.random.randint(0, len(x) - n + 1)
    return x[s:s + n]


def batchify(xs):
    n = max(len(x) for x in xs)
    arr = np.zeros((len(xs), n), dtype=np.float32)
    mask = np.zeros((len(xs), n), dtype=np.int64)
    for i, x in enumerate(xs):
        arr[i, :len(x)] = norm(x); mask[i, :len(x)] = 1
    return torch.from_numpy(arr).to(DEV), torch.from_numpy(mask).to(DEV)


# ----------------------------------------------------------------- model
class Reg(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL)
        self.enc.feature_extractor._freeze_parameters()
        for layer in self.enc.encoder.layers[:FREEZE_LAYERS]:
            for p in layer.parameters():
                p.requires_grad = False
        self.enc.config.layerdrop = 0.0
        self.enc.gradient_checkpointing_enable()
        d = self.enc.config.hidden_size
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.2), nn.Linear(d, 1))

    def forward(self, wav, mask):
        h = self.enc(wav, attention_mask=mask).last_hidden_state            # (B,T,D)
        fm = self.enc._get_feature_vector_attention_mask(h.shape[1], mask)  # (B,T)
        fm = fm.unsqueeze(-1).to(h.dtype)
        pooled = (h * fm).sum(1) / fm.sum(1).clamp(min=1)
        return self.head(pooled).squeeze(-1) + Y_MEAN


@torch.no_grad()
def predict(model, xs):
    """Average over CROP_S windows of each clip."""
    model.eval()
    out = []
    for x in xs:
        wins = [x[s:s + CROP_S * SR] for s in range(0, len(x), CROP_S * SR)]
        wins = [w for w in wins if len(w) >= SR // 2] or [x]
        wav, mask = batchify(wins)
        with torch.autocast('cuda', dtype=torch.float16):
            p = model(wav, mask).float()
        w = torch.tensor([len(v) for v in wins], device=DEV, dtype=torch.float32)
        out.append(float((p * w).sum() / w.sum()))
    return np.array(out)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# ------------------------------------------------------------------ CV
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.full(len(tr), np.nan, dtype=np.float32)
pte = np.zeros(len(te), dtype=np.float32)
t0 = time.time()
for fold, (tri, vai) in enumerate(kf.split(keep)):
    tri, vai = keep[tri], keep[vai]
    model = Reg().to(DEV)
    params = [{'params': [p for p in model.enc.parameters() if p.requires_grad], 'lr': LR_ENC},
              {'params': model.head.parameters(), 'lr': LR_HEAD}]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    steps = EPOCHS * math.ceil(len(tri) / (BS * ACC))
    sch = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.cuda.amp.GradScaler()
    best = (1e9, None, -1)
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(tri)
        tot, nb = 0.0, 0
        opt.zero_grad()
        for bi in range(0, len(perm), BS):
            ids = perm[bi:bi + BS]
            wav, mask = batchify([rand_crop(A_tr[i]) for i in ids])
            yb = torch.tensor(y_all[ids], device=DEV)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = nn.functional.mse_loss(model(wav, mask).float(), yb) / ACC
            scaler.scale(loss).backward()
            tot += float(loss) * ACC; nb += 1
            if nb % ACC == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        pv = predict(model, [A_tr[i] for i in vai])
        r = rmse(y_all[vai], pv)
        print(f'fold {fold} ep {ep}  train_mse {tot / nb:.4f}  val_rmse {r:.4f}  {time.time() - t0:.0f}s', flush=True)
        if r < best[0]:
            best = (r, pv, ep)
            pt = predict(model, A_te)
    oof[vai] = best[1]
    pte += pt / kf.get_n_splits()
    print(f'== fold {fold} best ep {best[2]} rmse {best[0]:.4f}', flush=True)
    del model, opt; torch.cuda.empty_cache()

m = ~np.isnan(oof)
print(f'OOF rmse {rmse(y_all[m], oof[m]):.4f}  pearson {np.corrcoef(y_all[m], oof[m])[0, 1]:.4f}  (n={m.sum()})')
pd.DataFrame({'filename': tr.filename[m], 'label': y_all[m], 'oof': oof[m]}).to_csv(os.path.join(OUT, 'oof_ft.csv'), index=False)
pte = np.where(noise_te, 0.0, np.clip(pte, 1, 5))
pd.DataFrame({'filename': te.filename, 'pred': pte}).to_csv(os.path.join(OUT, 'test_ft.csv'), index=False)
pd.DataFrame({'filename': te.filename, 'label': pte}).to_csv(os.path.join(OUT, 'submission.csv'), index=False)
print('done', f'{time.time() - t0:.0f}s')

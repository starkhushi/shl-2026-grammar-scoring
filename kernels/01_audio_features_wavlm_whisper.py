# SHL Hiring Assessment 2026 -- stage 1b: large SSL / Whisper-encoder embeddings (GPU, internet)
"""Per-layer time-pooled hidden states for the encoders that reportedly carry the
most grammar-score signal (upper layers of WavLM-large and the Whisper-large-v3
encoder). Saved to /kaggle/working, row order = index.csv:

    emb_wavlmL_mean.npy   (N, 25, 1024)   microsoft/wavlm-large, mean over time
    emb_wavlmL_std.npy    (N, 25, 1024)   same, std over time
    emb_whisper_mean.npy  (N, 33, 1280)   openai/whisper-large-v3 encoder, mean over valid frames
    emb_whisper_std.npy   (N, 33, 1280)
    index.csv             (filename, split)

Clips are <= 60 s; each is processed as 30 s windows and pooled with weights
proportional to window length so a 45 s clip is not dominated by its tail.
"""
import glob
import os
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import AutoModel, WhisperFeatureExtractor, WhisperModel

SR = 16000
WIN = 30 * SR
OUT = '/kaggle/working'
DEV = 'cuda'
print(torch.cuda.get_device_name(0), flush=True)


def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(f'{pattern} not found under /kaggle/input')
    return hits[0]


tr_csv = find('Dataset_Final/train.csv')
root = os.path.dirname(tr_csv)
tr = pd.read_csv(tr_csv)
te = pd.read_csv(os.path.join(root, 'test.csv'))
tr['split'] = 'train'
te['split'] = 'test'
idx = pd.concat([tr[['filename', 'split']], te[['filename', 'split']]], ignore_index=True)
idx['path'] = [os.path.join(root, s, f) for s, f in zip(idx.split, idx.filename)]


def load(p):
    x, sr = sf.read(p, dtype='float32')
    if x.ndim > 1:
        x = x.mean(1)
    assert sr == SR
    return x


def windows(x):
    for s in range(0, len(x), WIN):
        seg = x[s:s + WIN]
        if len(seg) >= SR // 2:
            yield seg


def pool(parts):
    """parts: list of (weight, mean[L,D], std[L,D]) -> weighted mean / std."""
    w = np.array([p[0] for p in parts], dtype=np.float64)
    w /= w.sum()
    m = sum(wi * p[1] for wi, p in zip(w, parts))
    s = sum(wi * p[2] for wi, p in zip(w, parts))
    return m.astype(np.float32), s.astype(np.float32)


# ------------------------------------------------------------- WavLM-large
wavlm = AutoModel.from_pretrained('microsoft/wavlm-large', torch_dtype=torch.float16).to(DEV).eval()


@torch.no_grad()
def wavlm_embed(x):
    parts = []
    for seg in windows(x):
        inp = torch.from_numpy(seg)[None].to(DEV, torch.float16)
        inp = (inp - inp.mean()) / (inp.std() + 1e-5)
        hs = wavlm(inp, output_hidden_states=True).hidden_states          # 25 x (1,T,1024)
        H = torch.stack([h[0].float() for h in hs])                        # (25,T,1024)
        parts.append((len(seg), H.mean(1).cpu().numpy(), H.std(1).cpu().numpy()))
    return pool(parts)


# ---------------------------------------------------- Whisper-large-v3 encoder
fe = WhisperFeatureExtractor.from_pretrained('openai/whisper-large-v3')
wenc = WhisperModel.from_pretrained('openai/whisper-large-v3', torch_dtype=torch.float16).encoder.to(DEV).eval()


@torch.no_grad()
def whisper_embed(x):
    parts = []
    for seg in windows(x):
        feats = fe(seg, sampling_rate=SR, return_tensors='pt').input_features.to(DEV, torch.float16)
        hs = wenc(feats, output_hidden_states=True).hidden_states          # 33 x (1,1500,1280)
        nvalid = max(int(np.ceil(len(seg) / SR * 50)), 1)                  # 50 frames / s, pad is silence
        H = torch.stack([h[0, :nvalid].float() for h in hs])
        parts.append((len(seg), H.mean(1).cpu().numpy(), H.std(1).cpu().numpy()))
    return pool(parts)


t0 = time.time()
WM, WS, HM, HS = [], [], [], []
for i, r in idx.iterrows():
    x = load(r.path)
    m, s = wavlm_embed(x)
    WM.append(m); WS.append(s)
    m, s = whisper_embed(x)
    HM.append(m); HS.append(s)
    if i % 100 == 0:
        print(f'{i}/{len(idx)} {time.time() - t0:.0f}s', flush=True)

np.save(os.path.join(OUT, 'emb_wavlmL_mean.npy'), np.stack(WM))
np.save(os.path.join(OUT, 'emb_wavlmL_std.npy'), np.stack(WS))
np.save(os.path.join(OUT, 'emb_whisper_mean.npy'), np.stack(HM))
np.save(os.path.join(OUT, 'emb_whisper_std.npy'), np.stack(HS))
idx[['filename', 'split']].to_csv(os.path.join(OUT, 'index.csv'), index=False)
print('done', np.stack(WM).shape, np.stack(HM).shape, f'{time.time() - t0:.0f}s', flush=True)

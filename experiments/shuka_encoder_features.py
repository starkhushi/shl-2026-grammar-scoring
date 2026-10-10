# SHL Hiring Assessment 2026 -- stage 6: Sarvam Shuka v1 audio encoder ("Saaras v1") embeddings (GPU, internet)
"""Shuka v1 = Saaras v1 audio encoder (Whisper-large-v2 architecture, 32 layers,
d=1280) + projector + Llama-3-8B. Only the encoder is needed: all of its
weights ('audio_tower.*') live in shard 1, so the 17.5 GB checkpoint is never
fully downloaded and the LLM is never loaded.

Also reports how far the Saaras weights moved from stock openai/whisper-large-v2:
if Shuka kept the encoder frozen (Ultravox default) the features are just
Whisper-v2 and add nothing Sarvam-specific.

Outputs (row order = index.csv, same as stage 1b):
    emb_shuka_mean.npy, emb_shuka_std.npy  (N, 33, 1280)
"""
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperModel
from transformers.models.whisper.modeling_whisper import WhisperEncoder

SR = 16000
WIN = 30 * SR
OUT = '/kaggle/working'
DEV = 'cuda'
REPO = 'sarvamai/shuka_v1'
print(torch.cuda.get_device_name(0), flush=True)


def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(f'{pattern} not found under /kaggle/input')
    return hits[0]


# ------------------------------------------------------- load Saaras encoder
t0 = time.time()
cfg = json.load(open(hf_hub_download(REPO, 'config.json')))
wcfg = WhisperConfig(**{k: v for k, v in cfg['audio_config'].items() if not k.startswith('_')})
shard = hf_hub_download(REPO, 'model-00001-of-00004.safetensors')
sd = {}
with safe_open(shard, 'pt') as f:
    for k in f.keys():
        if k.startswith('audio_tower.'):
            sd[k[len('audio_tower.'):]] = f.get_tensor(k)
print('audio_tower tensors', len(sd), f'{time.time() - t0:.0f}s', flush=True)
enc = WhisperEncoder(wcfg)
missing, unexpected = enc.load_state_dict(sd, strict=False)
print('missing', missing, 'unexpected', unexpected, flush=True)

# how different is it from stock whisper-large-v2?
ref = WhisperModel.from_pretrained('openai/whisper-large-v2', torch_dtype=torch.float32).encoder.state_dict()
diffs = {k: float((sd[k].float() - ref[k].float()).abs().max()) for k in sd if k in ref}
rel = np.mean([float((sd[k].float() - ref[k].float()).norm() / (ref[k].float().norm() + 1e-8)) for k in sd if k in ref])
print(f'vs whisper-large-v2: max|diff| {max(diffs.values()):.3e}  mean relative diff {rel:.3e}  '
      f'identical tensors {sum(v == 0 for v in diffs.values())}/{len(diffs)}', flush=True)
FROZEN = max(diffs.values()) == 0
print('SAARAS == stock Whisper-v2 (frozen)?', FROZEN, flush=True)
del ref
enc = enc.half().to(DEV).eval()
fe = WhisperFeatureExtractor.from_pretrained('openai/whisper-large-v2')

# --------------------------------------------------------------- data
tr_csv = find('Dataset_Final/train.csv')
root = os.path.dirname(tr_csv)
tr = pd.read_csv(tr_csv)
te = pd.read_csv(os.path.join(root, 'test.csv'))
tr['split'] = 'train'
te['split'] = 'test'
idx = pd.concat([tr[['filename', 'split']], te[['filename', 'split']]], ignore_index=True)


@torch.no_grad()
def embed(x):
    parts = []
    for a in range(0, len(x), WIN):
        seg = x[a:a + WIN]
        if len(seg) < SR // 2:
            continue
        feats = fe(seg, sampling_rate=SR, return_tensors='pt').input_features.to(DEV, torch.float16)
        hs = enc(feats, output_hidden_states=True).hidden_states            # 33 x (1,1500,1280)
        nvalid = max(int(np.ceil(len(seg) / SR * 50)), 1)
        H = torch.stack([h[0, :nvalid].float() for h in hs])
        parts.append((len(seg), H.mean(1).cpu().numpy(), H.std(1).cpu().numpy()))
    w = np.array([p[0] for p in parts], dtype=np.float64); w /= w.sum()
    return (sum(wi * p[1] for wi, p in zip(w, parts)).astype(np.float32),
            sum(wi * p[2] for wi, p in zip(w, parts)).astype(np.float32))


if FROZEN:
    print('encoder is stock Whisper-large-v2 -> nothing Sarvam-specific to add; skipping extraction')
else:
    M, S = [], []
    for i, (s, f) in enumerate(zip(idx.split, idx.filename)):
        x, sr = sf.read(os.path.join(root, s, f), dtype='float32')
        if x.ndim > 1:
            x = x.mean(1)
        m, sd_ = embed(x)
        M.append(m); S.append(sd_)
        if i % 100 == 0:
            print(f'{i}/{len(idx)} {time.time() - t0:.0f}s', flush=True)
    np.save(os.path.join(OUT, 'emb_shuka_mean.npy'), np.stack(M))
    np.save(os.path.join(OUT, 'emb_shuka_std.npy'), np.stack(S))
idx[['filename', 'split']].to_csv(os.path.join(OUT, 'index.csv'), index=False)
print('done', f'{time.time() - t0:.0f}s')

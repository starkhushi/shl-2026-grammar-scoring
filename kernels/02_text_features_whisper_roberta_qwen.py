# SHL Hiring Assessment 2026 -- stage 4: transcripts + text embeddings (GPU, internet)
"""Text track. Text features cannot recognise a speaker's voice, so they carry
over to the all-new test speakers better than audio embeddings do.

  1. Whisper-large-v3 transcripts: each clip is cut into 30 s windows, windows
     are batched (no word timestamps -- that mode made v1 take >3 h), and the
     mean token log-probability is kept as an ASR-confidence feature.
  2. Mean-pooled hidden states of every layer of
        roberta-large           -> emb_roberta.npy  (N, 25, 1024)
        Qwen/Qwen2.5-1.5B       -> emb_qwen.npy     (N, 29, 1536)

Outputs (row order = index.csv, same as stage 1b): text.csv, emb_*.npy, index.csv
"""
import glob
import os
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import (AutoModel, AutoTokenizer, WhisperForConditionalGeneration,
                          WhisperProcessor)

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

# ---------------------------------------------------------- 1. transcripts
t0 = time.time()
proc = WhisperProcessor.from_pretrained('openai/whisper-large-v3')
asr = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v3',
                                                      torch_dtype=torch.float16).to(DEV).eval()

wins = []                                   # (row, window index, samples)
for i, (s, f) in enumerate(zip(idx.split, idx.filename)):
    x, sr = sf.read(os.path.join(root, s, f), dtype='float32')
    if x.ndim > 1:
        x = x.mean(1)
    assert sr == SR
    for k, a in enumerate(range(0, len(x), WIN)):
        seg = x[a:a + WIN]
        if len(seg) >= SR:                  # skip sub-second tails
            wins.append((i, k, seg))
print('windows', len(wins), flush=True)

texts = {}
lps = {}
BS = 8
SPECIAL = set(proc.tokenizer.all_special_ids)
with torch.no_grad():
    for b in range(0, len(wins), BS):
        chunk = wins[b:b + BS]
        feats = proc([w[2] for w in chunk], sampling_rate=SR,
                     return_tensors='pt').input_features.to(DEV, torch.float16)
        seqs = asr.generate(feats, language='en', task='transcribe', max_new_tokens=220)
        dec = proc.batch_decode(seqs, skip_special_tokens=True)
        # ASR confidence: one teacher-forced pass over the generated tokens
        # (asking generate() for scores keeps every cache and OOMs on a T4)
        logits = asr(input_features=feats, decoder_input_ids=seqs[:, :-1]).logits.float()
        lp = torch.log_softmax(logits, -1).gather(-1, seqs[:, 1:, None])[..., 0].cpu().numpy()
        tok = seqs[:, 1:].cpu().numpy()
        for (i, k, _), t, l, tk in zip(chunk, dec, lp, tok):
            texts.setdefault(i, {})[k] = t.strip()
            lps.setdefault(i, []).extend(l[[x not in SPECIAL for x in tk]].tolist())
        if b % (BS * 20) == 0:
            print(f'asr {b}/{len(wins)} {time.time() - t0:.0f}s | {dec[0][:90]!r}', flush=True)

idx['text'] = [' '.join(texts.get(i, {})[k] for k in sorted(texts.get(i, {}))) for i in range(len(idx))]
idx['asr_logprob'] = [float(np.mean(lps[i])) if lps.get(i) else np.nan for i in range(len(idx))]
idx['n_words'] = idx.text.str.split().str.len().fillna(0)
del asr
torch.cuda.empty_cache()
print('asr done', f'{time.time() - t0:.0f}s', 'empty:', int((idx.text.str.len() == 0).sum()), flush=True)


# -------------------------------------------------------- 2. text embeddings
@torch.no_grad()
def embed_all(name, bs, max_len):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModel.from_pretrained(name, torch_dtype=torch.float16).to(DEV).eval()
    out = []
    texts_ = [t if t else '.' for t in idx.text]
    for b in range(0, len(texts_), bs):
        enc = tok(texts_[b:b + bs], padding=True, truncation=True, max_length=max_len,
                  return_tensors='pt').to(DEV)
        hs = m(**enc, output_hidden_states=True).hidden_states
        mask = enc['attention_mask'].unsqueeze(-1).float()
        H = torch.stack([(h.float() * mask).sum(1) / mask.sum(1) for h in hs], 1)  # (B,L,D)
        out.append(H.cpu().numpy().astype(np.float32))
    del m
    torch.cuda.empty_cache()
    E = np.concatenate(out)
    print(name, E.shape, f'{time.time() - t0:.0f}s', flush=True)
    return E


np.save(os.path.join(OUT, 'emb_roberta.npy'), embed_all('roberta-large', 32, 512))
np.save(os.path.join(OUT, 'emb_qwen.npy'), embed_all('Qwen/Qwen2.5-1.5B', 16, 512))

idx[['filename', 'split']].to_csv(os.path.join(OUT, 'index.csv'), index=False)
idx.drop(columns=[]).to_csv(os.path.join(OUT, 'text.csv'), index=False)
print(idx.groupby('split')[['n_words', 'asr_logprob']].describe().T.round(3).to_string())
print('done', f'{time.time() - t0:.0f}s')

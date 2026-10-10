# SHL Hiring Assessment 2026 -- stage 12: three new feature families in ONE GPU pass (~45 min T4)
"""Everything a top-5 push still lacks, extracted together so the GPU is used once:

  A. Segment-level audio embeddings. Upper layers of WavLM-large (19-23) and the
     Whisper-large-v3 encoder (28-32), mean-pooled per 10-second segment instead of
     per clip -> up to 6 rows per clip (732 -> ~4000 training rows).
        seg_wavlm.npy   (N, 6, 5, 1024) float16, NaN-padded     seg_nseg.npy (N,)
        seg_whisper.npy (N, 6, 5, 1280) float16
  B. Verbatim CTC transcripts (wav2vec2-large-960h-lv60-self, greedy, no LM).
     Whisper silently repairs grammar; a CTC model writes what was said.
        text_ctc.csv (filename, split, ctc)
  C. Qwen2.5-7B-Instruct hidden states (4-bit) on BOTH transcripts, plus
     RoBERTa-large on the CTC transcript.
        emb_qwen7_whisper.npy / emb_qwen7_ctc.npy (N, 10, 3584) float16, layers QL
        emb_roberta_ctc.npy (N, 25, 1024) float16
Row order = index.csv (train.csv order then test.csv order, same as stage 1b/4).
"""
import glob
import os
import subprocess
import time

subprocess.run('pip install -q -U bitsandbytes', shell=True, check=False)

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402
import soundfile as sf                               # noqa: E402
import torch                                         # noqa: E402
from transformers import (AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer,  # noqa: E402
                          BitsAndBytesConfig, Wav2Vec2ForCTC, WhisperFeatureExtractor, WhisperModel)

SR = 16000
WIN = 30 * SR
SEG_FRAMES = 500                 # 10 s at 50 frames/s (both WavLM and Whisper encoder)
MAX_SEG = 6
WL = [19, 20, 21, 22, 23]        # WavLM-large hidden_states indices
WHL = [28, 29, 30, 31, 32]       # Whisper encoder hidden_states indices
QL = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
OUT = '/kaggle/working'
DEV = 'cuda'
t0 = time.time()
print(torch.cuda.get_device_name(0), flush=True)


def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(f'{pattern} not found under /kaggle/input')
    return hits[0]


def log(msg):
    print(f'[{time.time() - t0:5.0f}s] {msg}', flush=True)


tr_csv = find('Dataset_Final/train.csv')
root = os.path.dirname(tr_csv)
tr = pd.read_csv(tr_csv); te = pd.read_csv(os.path.join(root, 'test.csv'))
tr['split'] = 'train'; te['split'] = 'test'
idx = pd.concat([tr[['filename', 'split']], te[['filename', 'split']]], ignore_index=True)
idx['path'] = [os.path.join(root, s, f) for s, f in zip(idx.split, idx.filename)]
wtext = pd.read_csv(find('text.csv'))
assert (wtext.filename.values == idx.filename.values).all()
N = len(idx)


def load(p):
    x, sr = sf.read(p, dtype='float32')
    if x.ndim > 1:
        x = x.mean(1)
    assert sr == SR
    return x


def windows(x):
    for a in range(0, len(x), WIN):
        seg = x[a:a + WIN]
        if len(seg) >= SR // 2:
            yield seg


def seg_pool(H_windows):
    """H_windows: list of (L, T, D) valid-frame tensors per 30 s window -> (MAX_SEG, L, D) NaN-padded."""
    segs = []
    for H in H_windows:
        for k in range(0, H.shape[1], SEG_FRAMES):
            part = H[:, k:k + SEG_FRAMES]
            if part.shape[1] >= 100:                       # >= 2 s
                segs.append(part.mean(1))
    out = np.full((MAX_SEG, H_windows[0].shape[0], H_windows[0].shape[2]), np.nan, np.float16)
    for i, s in enumerate(segs[:MAX_SEG]):
        out[i] = s.float().cpu().numpy().astype(np.float16)
    return out, min(len(segs), MAX_SEG)


# ------------------------------------------------- A+B: audio pass (3 models)
wavlm = AutoModel.from_pretrained('microsoft/wavlm-large', torch_dtype=torch.float16).to(DEV).eval()
fe = WhisperFeatureExtractor.from_pretrained('openai/whisper-large-v3')
wenc = WhisperModel.from_pretrained('openai/whisper-large-v3', torch_dtype=torch.float16).encoder.to(DEV).eval()
ctc_proc = AutoProcessor.from_pretrained('facebook/wav2vec2-large-960h-lv60-self')
ctc = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-large-960h-lv60-self', torch_dtype=torch.float16).to(DEV).eval()
log('audio models loaded')

SW = np.full((N, MAX_SEG, len(WL), 1024), np.nan, np.float16)
SH = np.full((N, MAX_SEG, len(WHL), 1280), np.nan, np.float16)
nseg = np.zeros(N, int)
ctc_text = []
with torch.no_grad():
    for i, p in enumerate(idx.path):
        x = load(p)
        Hw, Hh, txt = [], [], []
        for seg in windows(x):
            inp = torch.from_numpy(seg)[None].to(DEV, torch.float16)
            inp = (inp - inp.mean()) / (inp.std() + 1e-5)
            hs = wavlm(inp, output_hidden_states=True).hidden_states
            Hw.append(torch.stack([hs[l][0] for l in WL]))                       # (5,T,1024)
            feats = fe(seg, sampling_rate=SR, return_tensors='pt').input_features.to(DEV, torch.float16)
            hs = wenc(feats, output_hidden_states=True).hidden_states
            nvalid = max(int(np.ceil(len(seg) / SR * 50)), 1)
            Hh.append(torch.stack([hs[l][0, :nvalid] for l in WHL]))             # (5,T,1280)
            iv = ctc_proc(seg, sampling_rate=SR, return_tensors='pt').input_values.to(DEV, torch.float16)
            txt.append(ctc_proc.batch_decode(ctc(iv).logits.argmax(-1))[0])
        SW[i], nseg[i] = seg_pool(Hw)
        SH[i], _ = seg_pool(Hh)
        ctc_text.append(' '.join(txt).lower().strip())
        if i % 100 == 0:
            log(f'audio {i}/{N} nseg={nseg[i]} | {ctc_text[-1][:80]!r}')
np.save(os.path.join(OUT, 'seg_wavlm.npy'), SW)
np.save(os.path.join(OUT, 'seg_whisper.npy'), SH)
np.save(os.path.join(OUT, 'seg_nseg.npy'), nseg)
idx['ctc'] = ctc_text
idx[['filename', 'split', 'ctc']].to_csv(os.path.join(OUT, 'text_ctc.csv'), index=False)
del wavlm, wenc, ctc
torch.cuda.empty_cache()
log(f'audio pass done; nseg distribution {np.bincount(nseg).tolist()}')


# ----------------------------------------------------- C: text embeddings
@torch.no_grad()
def embed_texts(model, tok, texts, layers, bs, max_len=512):
    out = []
    for b in range(0, len(texts), bs):
        enc = tok([t if t else '.' for t in texts[b:b + bs]], padding=True, truncation=True,
                  max_length=max_len, return_tensors='pt').to(DEV)
        hs = model(**enc, output_hidden_states=True).hidden_states
        mask = enc['attention_mask'].unsqueeze(-1).float()
        H = torch.stack([(hs[l].float() * mask).sum(1) / mask.sum(1) for l in layers], 1)
        out.append(H.cpu().numpy().astype(np.float16))
    return np.concatenate(out)


rtok = AutoTokenizer.from_pretrained('roberta-large')
rob = AutoModel.from_pretrained('roberta-large', torch_dtype=torch.float16).to(DEV).eval()
np.save(os.path.join(OUT, 'emb_roberta_ctc.npy'), embed_texts(rob, rtok, ctc_text, list(range(25)), 32))
del rob
torch.cuda.empty_cache()
log('roberta on ctc done')

QN = 'Qwen/Qwen2.5-7B-Instruct'
qtok = AutoTokenizer.from_pretrained(QN)
qtok.padding_side = 'right'
qwen = AutoModelForCausalLM.from_pretrained(
    QN, device_map='cuda:0',
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                           bnb_4bit_quant_type='nf4')).eval()
log('qwen7 loaded')
np.save(os.path.join(OUT, 'emb_qwen7_whisper.npy'),
        embed_texts(qwen.model, qtok, wtext.text.fillna('').tolist(), QL, 8))
log('qwen7 on whisper text done')
np.save(os.path.join(OUT, 'emb_qwen7_ctc.npy'), embed_texts(qwen.model, qtok, ctc_text, QL, 8))
log('qwen7 on ctc text done')

idx[['filename', 'split']].to_csv(os.path.join(OUT, 'index.csv'), index=False)
log('ALL DONE')

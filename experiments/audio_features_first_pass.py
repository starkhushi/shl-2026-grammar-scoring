# SHL Hiring Assessment 2026 -- stage 1: transcripts + audio features (GPU, internet)
"""Extract everything downstream models need, once, and save to /kaggle/working:

    transcripts.csv : filename, split, label, whisper text, verbatim CTC text,
                      fluency / pause features from Whisper word timestamps,
                      raw-signal energy / voicing features
    emb_wavlm.npy   : (N, layers, 768)  mean-pooled hidden states, microsoft/wavlm-base-plus
    emb_hubert.npy  : (N, layers, 1024) mean-pooled hidden states, facebook/hubert-large-ls960-ft
    index.csv       : row order for the .npy files (filename, split)

Two transcripts on purpose: Whisper silently "repairs" grammar and drops
fillers; a CTC model without an LM keeps the speaker's actual errors.
"""
import glob
import os
import re
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch

SR = 16000
OUT = '/kaggle/working'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', DEV, torch.cuda.get_device_name(0) if DEV == 'cuda' else '', flush=True)


def find(pattern):
    """Locate an input file regardless of mount layout (glob, never hardcode)."""
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(f'{pattern} not found under /kaggle/input')
    return hits[0]


# ---------------------------------------------------------------- data index
tr_csv = find('Dataset_Final/train.csv')
root = os.path.dirname(tr_csv)
tr = pd.read_csv(tr_csv)
te = pd.read_csv(os.path.join(root, 'test.csv'))
tr['split'] = 'train'
te['split'] = 'test'
te['label'] = np.nan
idx = pd.concat([tr, te], ignore_index=True)
idx['path'] = [os.path.join(root, s, f) for s, f in zip(idx.split, idx.filename)]
print(idx.split.value_counts().to_dict(), flush=True)


def load(p):
    x, sr = sf.read(p, dtype='float32')
    if x.ndim > 1:
        x = x.mean(1)
    assert sr == SR, (p, sr)
    return x


# ------------------------------------------------------- raw signal features
def signal_feats(x):
    fr = x[:len(x) // 400 * 400].reshape(-1, 400)        # 25 ms frames
    rms = np.sqrt((fr ** 2).mean(1) + 1e-12)
    thr = max(rms.max() * 0.02, 1e-4)
    return dict(dur=len(x) / SR, rms_mean=float(rms.mean()), rms_std=float(rms.std()),
                peak=float(np.abs(x).max()), voiced_frac=float((rms > thr).mean()),
                nonzero_frac=float((np.abs(x) > 1e-4).mean()))


# ------------------------------------------------------------- Whisper turbo
from transformers import pipeline  # noqa: E402

t0 = time.time()
asr = pipeline('automatic-speech-recognition', model='openai/whisper-large-v3-turbo',
               torch_dtype=torch.float16, device=0 if DEV == 'cuda' else -1,
               chunk_length_s=30, batch_size=8)
FILLERS = {'um', 'uh', 'umm', 'uhh', 'hmm', 'mm', 'er', 'ah', 'eh', 'mhm'}


def whisper_one(x):
    out = asr({'raw': x, 'sampling_rate': SR}, return_timestamps='word',
              generate_kwargs={'language': 'en', 'task': 'transcribe'})
    text = out['text'].strip()
    chunks = [c for c in out.get('chunks', []) if c.get('timestamp') and c['timestamp'][0] is not None]
    starts = np.array([c['timestamp'][0] for c in chunks], dtype=float)
    ends = np.array([c['timestamp'][1] if c['timestamp'][1] is not None else c['timestamp'][0]
                     for c in chunks], dtype=float)
    words = [re.sub(r'[^a-z\']', '', c['text'].lower()) for c in chunks]
    words = [w for w in words if w]
    dur = len(x) / SR
    f = dict(w_nwords=len(words), w_nchars=len(text))
    if len(starts) >= 2:
        gaps = np.clip(starts[1:] - ends[:-1], 0, None)
        f.update(w_pause_cnt=int((gaps > 0.5).sum()), w_pause_long=int((gaps > 1.5).sum()),
                 w_pause_mean=float(gaps.mean()), w_pause_max=float(gaps.max()),
                 w_pause_total=float(gaps.sum()),
                 w_speech_span=float(ends[-1] - starts[0]),
                 w_first_word=float(starts[0]), w_last_word=float(dur - ends[-1]))
    else:
        f.update(w_pause_cnt=0, w_pause_long=0, w_pause_mean=0.0, w_pause_max=0.0,
                 w_pause_total=0.0, w_speech_span=0.0, w_first_word=dur, w_last_word=dur)
    f['w_rate'] = len(words) / max(dur, 1e-3)
    f['w_rate_speech'] = len(words) / max(f['w_speech_span'], 1e-3)
    f['w_filler'] = sum(w in FILLERS for w in words)
    f['w_repeat'] = sum(a == b for a, b in zip(words, words[1:]))
    f['w_uniq_ratio'] = len(set(words)) / max(len(words), 1)
    f['w_mean_wlen'] = float(np.mean([len(w) for w in words])) if words else 0.0
    return text, f


# ------------------------------------------------------------ wav2vec2 CTC
from transformers import AutoProcessor, Wav2Vec2ForCTC, AutoModel  # noqa: E402

ctc_proc = AutoProcessor.from_pretrained('facebook/wav2vec2-large-960h-lv60-self')
ctc = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-large-960h-lv60-self',
                                     torch_dtype=torch.float16).to(DEV).eval()


@torch.no_grad()
def ctc_one(x):
    # 30 s windows keep memory flat; clips are <= 60 s so at most two windows
    outs = []
    for s in range(0, len(x), 30 * SR):
        seg = x[s:s + 30 * SR]
        if len(seg) < SR // 2:
            continue
        inp = ctc_proc(seg, sampling_rate=SR, return_tensors='pt').input_values.to(DEV, torch.float16)
        ids = ctc(inp).logits.argmax(-1)
        outs.append(ctc_proc.batch_decode(ids)[0])
    return ' '.join(outs).lower().strip()


# ------------------------------------------------------- SSL embeddings
def make_enc(name):
    m = AutoModel.from_pretrained(name, torch_dtype=torch.float16).to(DEV).eval()
    return m


wavlm = make_enc('microsoft/wavlm-base-plus')
hubert = make_enc('facebook/hubert-large-ls960-ft')


@torch.no_grad()
def embed(model, x):
    """Mean over time of every hidden layer -> (layers, dim). 30 s windows, averaged."""
    acc = None
    n = 0
    for s in range(0, len(x), 30 * SR):
        seg = x[s:s + 30 * SR]
        if len(seg) < SR // 2:
            continue
        inp = torch.from_numpy(seg)[None].to(DEV, torch.float16)
        inp = (inp - inp.mean()) / (inp.std() + 1e-5)
        hs = model(inp, output_hidden_states=True).hidden_states
        h = torch.stack([t.mean(1)[0] for t in hs]).float().cpu().numpy()
        w = len(seg)
        acc = h * w if acc is None else acc + h * w
        n += w
    return acc / max(n, 1)


# ------------------------------------------------------------------- run
rows, E_w, E_h = [], [], []
for i, r in idx.iterrows():
    x = load(r.path)
    f = dict(filename=r.filename, split=r.split, label=r.label)
    f.update(signal_feats(x))
    try:
        f['whisper'], wf = whisper_one(x)
    except Exception as e:                      # never lose the whole run to one clip
        print('whisper fail', r.filename, e, flush=True)
        f['whisper'], wf = '', {}
    f.update(wf)
    try:
        f['ctc'] = ctc_one(x)
    except Exception as e:
        print('ctc fail', r.filename, e, flush=True)
        f['ctc'] = ''
    E_w.append(embed(wavlm, x))
    E_h.append(embed(hubert, x))
    rows.append(f)
    if i % 50 == 0:
        print(f'{i}/{len(idx)}  {time.time() - t0:.0f}s  {r.filename}  lbl={r.label}  '
              f'words={f.get("w_nwords")}  | {f["whisper"][:80]!r}', flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'transcripts.csv'), index=False)
np.save(os.path.join(OUT, 'emb_wavlm.npy'), np.stack(E_w).astype(np.float32))
np.save(os.path.join(OUT, 'emb_hubert.npy'), np.stack(E_h).astype(np.float32))
idx[['filename', 'split']].to_csv(os.path.join(OUT, 'index.csv'), index=False)
print('done', df.shape, f'{time.time() - t0:.0f}s', flush=True)
print(df.groupby('split')[['w_nwords', 'w_rate', 'w_pause_cnt', 'voiced_frac']].describe().T.round(3).to_string())

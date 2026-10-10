# SHL Hiring Assessment 2026 -- stage 19: Qwen2-Audio-7B-Instruct listens to the clip (GPU, ~1 h)
"""An audio-language model that takes the waveform directly (Whisper-large-v3-style
audio encoder -> Qwen-7B LLM). It is the one source of information not yet in the
stack: audio and language are processed jointly, zero training. For every 30-s window
the model is asked the rubric question; we keep
    emb_q2a_mean.npy  (N, 8, 4096)  mean over all tokens of hidden layers QL
    emb_q2a_last.npy  (N, 8, 4096)  last-token state (the position that would answer)
    q2a_grade.csv     expected value over the next-token probabilities of "1".."5"
Windows are averaged per clip weighted by length. Row order = index.csv (stage 1b).
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
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2AudioForConditionalGeneration  # noqa: E402

SR = 16000
WIN = 30 * SR
QL = [8, 12, 16, 20, 22, 24, 28, 32]
OUT = '/kaggle/working'
DEV = 'cuda'
t0 = time.time()
print(torch.cuda.get_device_name(0), flush=True)


def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(f'{pattern} not found under /kaggle/input')
    return hits[0]


def log(m):
    print(f'[{time.time() - t0:5.0f}s] {m}', flush=True)


tr_csv = find('Dataset_Final/train.csv'); root = os.path.dirname(tr_csv)
tr = pd.read_csv(tr_csv); te = pd.read_csv(os.path.join(root, 'test.csv'))
tr['split'] = 'train'; te['split'] = 'test'
idx = pd.concat([tr[['filename', 'split']], te[['filename', 'split']]], ignore_index=True)
idx['path'] = [os.path.join(root, s, f) for s, f in zip(idx.split, idx.filename)]
N = len(idx)

MODEL = 'Qwen/Qwen2-Audio-7B-Instruct'
proc = AutoProcessor.from_pretrained(MODEL)
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    MODEL, device_map='cuda:0',
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                           bnb_4bit_quant_type='nf4')).eval()
log('model loaded')
QUESTION = ('Listen to this spoken English answer. Rate ONLY the speaker\'s grammar on a 1-5 scale '
            '(1 = limited control of simple structures, 3 = decent but with errors, 5 = accurate complex grammar). '
            'Reply with a single digit.')
conv = [{'role': 'user', 'content': [{'type': 'audio', 'audio_url': 'clip.wav'}, {'type': 'text', 'text': QUESTION}]}]
prompt = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
tok = proc.tokenizer
DIG = [tok.encode(d, add_special_tokens=False)[0] for d in '12345']


def load(p):
    x, sr = sf.read(p, dtype='float32')
    if x.ndim > 1:
        x = x.mean(1)
    assert sr == SR
    return x


@torch.no_grad()
def one_window(seg):
    inputs = proc(text=prompt, audio=[seg], sampling_rate=SR, return_tensors='pt').to(DEV)
    out = model(**inputs, output_hidden_states=True)
    hs = out.hidden_states                                            # 33 x (1, T, 4096)
    mean = torch.stack([hs[l][0].float().mean(0) for l in QL]).cpu().numpy().astype(np.float16)
    last = torch.stack([hs[l][0, -1].float() for l in QL]).cpu().numpy().astype(np.float16)
    p = torch.softmax(out.logits[0, -1, DIG].float(), -1).cpu().numpy()
    return mean, last, p


MEAN, LAST, GR = [], [], []
for i, p in enumerate(idx.path):
    x = load(p)
    parts = []
    for a in range(0, len(x), WIN):
        seg = x[a:a + WIN]
        if len(seg) >= SR:
            try:
                m, l, pr = one_window(seg)
                parts.append((len(seg), m, l, pr))
            except Exception as e:                                    # never lose the run to one clip
                print('fail', idx.filename[i], e, flush=True)
    if not parts:
        parts = [(1, np.zeros((len(QL), 4096), np.float16), np.zeros((len(QL), 4096), np.float16), np.full(5, .2))]
    w = np.array([q[0] for q in parts], dtype=np.float64); w /= w.sum()
    MEAN.append(sum(wi * q[1].astype(np.float32) for wi, q in zip(w, parts)).astype(np.float16))
    LAST.append(sum(wi * q[2].astype(np.float32) for wi, q in zip(w, parts)).astype(np.float16))
    GR.append(sum(wi * q[3] for wi, q in zip(w, parts)))
    if i % 100 == 0:
        log(f'{i}/{N}  grade p={np.round(GR[-1], 2)}')
np.save(os.path.join(OUT, 'emb_q2a_mean.npy'), np.stack(MEAN))
np.save(os.path.join(OUT, 'emb_q2a_last.npy'), np.stack(LAST))
G = pd.DataFrame(np.stack(GR), columns=[f'q2a_p{k}' for k in range(1, 6)])
G['q2a_grade'] = G.values @ np.arange(1, 6)
G.insert(0, 'split', idx.split); G.insert(0, 'filename', idx.filename)
G.to_csv(os.path.join(OUT, 'q2a_grade.csv'), index=False)
idx[['filename', 'split']].to_csv(os.path.join(OUT, 'index.csv'), index=False)
m = G[G.split == 'train'].merge(tr[['filename', 'label']], on='filename'); m = m[m.label > 0]
log(f'zero-shot audio-LLM grade: r with label = {np.corrcoef(m.q2a_grade, m.label)[0, 1]:.3f}')
log('ALL DONE')

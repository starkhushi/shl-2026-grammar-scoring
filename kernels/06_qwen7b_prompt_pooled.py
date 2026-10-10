# SHL Hiring Assessment 2026 -- stage 14: one GPU pass (~50 min) along the axis that worked in v13
"""v13 showed the biggest gain since v5 came from a bigger LLM reading a VERBATIM
transcript. This pass pushes that axis three ways, plus one untried audio encoder:

  A. Verbatim Whisper transcript: Whisper-large-v3 with a disfluent prompt so it
     keeps fillers / repetitions / false starts but spells words correctly
     (the CTC transcript keeps errors but misspells heavily).   -> text_verbatim.csv
  B. Qwen2.5-7B-Instruct PROMPT-POOLED embeddings: instead of mean-pooling every
     token, the model is asked "the grammar quality of this speaker, in one word,
     is:" and the LAST-token hidden state is taken (PromptEOL-style), on all three
     transcripts (whisper / ctc / verbatim).                     -> emb_q7p_*.npy (N,10,3584)
     Plus plain mean-pooled Qwen-7B on the verbatim text.         -> emb_qwen7_verbatim.npy
  C. w2v-BERT 2.0 (facebook/w2v-bert-2.0, 4.5M h pretraining, different objective
     from WavLM/Whisper) per-layer mean/std.                      -> emb_w2vbert_{mean,std}.npy (N,25,1024)
Row order = index.csv (same as every earlier stage).
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
from transformers import (AutoFeatureExtractor, AutoModel, AutoModelForCausalLM, AutoTokenizer,  # noqa: E402
                          BitsAndBytesConfig, WhisperForConditionalGeneration, WhisperProcessor)

SR = 16000
WIN = 30 * SR
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
N = len(idx)
wtext = pd.read_csv(find('text.csv')); ctext = pd.read_csv(find('text_ctc.csv'))
assert (wtext.filename.values == idx.filename.values).all() and (ctext.filename.values == idx.filename.values).all()
whisper_txt = wtext.text.fillna('').tolist()
ctc_txt = ctext.ctc.fillna('').tolist()


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


AUDIO = [load(p) for p in idx.path]
log('audio loaded')

# ------------------------------------------------ A. verbatim Whisper transcript
proc = WhisperProcessor.from_pretrained('openai/whisper-large-v3')
asr = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v3', torch_dtype=torch.float16).to(DEV).eval()
PROMPT = "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking. So, so I I went to the the market."
prompt_ids = proc.get_prompt_ids(PROMPT, return_tensors='pt').to(DEV)
wins = [(i, k, seg) for i, x in enumerate(AUDIO) for k, seg in enumerate(windows(x)) if len(seg) >= SR]
texts = {}
BS = 8
with torch.no_grad():
    for b in range(0, len(wins), BS):
        chunk = wins[b:b + BS]
        feats = proc([w[2] for w in chunk], sampling_rate=SR, return_tensors='pt').input_features.to(DEV, torch.float16)
        seqs = asr.generate(feats, prompt_ids=prompt_ids, language='en', task='transcribe', max_new_tokens=220)
        dec = proc.batch_decode(seqs, skip_special_tokens=True)
        for (i, k, _), t in zip(chunk, dec):
            t = t.strip()
            if t.startswith(PROMPT):                 # some versions echo the prompt
                t = t[len(PROMPT):].strip()
            texts.setdefault(i, {})[k] = t
        if b % (BS * 40) == 0:
            log(f'asr {b}/{len(wins)} | {dec[0][:90]!r}')
verb_txt = [' '.join(texts.get(i, {})[k] for k in sorted(texts.get(i, {}))) for i in range(N)]
idx['verbatim'] = verb_txt
idx[['filename', 'split', 'verbatim']].to_csv(os.path.join(OUT, 'text_verbatim.csv'), index=False)
del asr
torch.cuda.empty_cache()
log(f'verbatim ASR done; empty {sum(1 for t in verb_txt if not t)}; sample: {verb_txt[0][:150]!r}')

# ------------------------------------------- B. Qwen-7B prompt-pooled + mean-pooled
QN = 'Qwen/Qwen2.5-7B-Instruct'
qtok = AutoTokenizer.from_pretrained(QN)
qwen = AutoModelForCausalLM.from_pretrained(
    QN, device_map='cuda:0',
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                           bnb_4bit_quant_type='nf4')).eval()
log('qwen7 loaded')
EOL = 'Transcript of a one-minute spoken answer (automatic speech recognition, ignore punctuation and spelling):\n"{t}"\nThe grammar quality of this speaker, in one word, is: "'


@torch.no_grad()
def q7_last(texts_, bs=4):
    qtok.padding_side = 'left'
    out = []
    for b in range(0, len(texts_), bs):
        ps = [EOL.format(t=(t or '(silence)')[:3000]) for t in texts_[b:b + bs]]
        enc = qtok(ps, padding=True, truncation=True, max_length=1024, return_tensors='pt').to(DEV)
        hs = qwen.model(**enc, output_hidden_states=True).hidden_states
        out.append(torch.stack([hs[l][:, -1].float() for l in QL], 1).cpu().numpy().astype(np.float16))
    return np.concatenate(out)


@torch.no_grad()
def q7_mean(texts_, bs=8):
    qtok.padding_side = 'right'
    out = []
    for b in range(0, len(texts_), bs):
        enc = qtok([t if t else '.' for t in texts_[b:b + bs]], padding=True, truncation=True,
                   max_length=512, return_tensors='pt').to(DEV)
        hs = qwen.model(**enc, output_hidden_states=True).hidden_states
        mask = enc['attention_mask'].unsqueeze(-1).float()
        out.append(torch.stack([(hs[l].float() * mask).sum(1) / mask.sum(1) for l in QL], 1).cpu().numpy().astype(np.float16))
    return np.concatenate(out)


for name, txt in [('whisper', whisper_txt), ('ctc', ctc_txt), ('verbatim', verb_txt)]:
    np.save(os.path.join(OUT, f'emb_q7p_{name}.npy'), q7_last(txt))
    log(f'qwen7 prompt-pooled on {name} done')
np.save(os.path.join(OUT, 'emb_qwen7_verbatim.npy'), q7_mean(verb_txt))
log('qwen7 mean-pooled on verbatim done')
del qwen
torch.cuda.empty_cache()

# ------------------------------------------------------------ C. w2v-BERT 2.0
wfe = AutoFeatureExtractor.from_pretrained('facebook/w2v-bert-2.0')
w2v = AutoModel.from_pretrained('facebook/w2v-bert-2.0', torch_dtype=torch.float16).to(DEV).eval()
M, S = [], []
with torch.no_grad():
    for i, x in enumerate(AUDIO):
        parts = []
        for seg in windows(x):
            inp = wfe(seg, sampling_rate=SR, return_tensors='pt')
            hs = w2v(input_features=inp.input_features.to(DEV, torch.float16),
                     attention_mask=inp.attention_mask.to(DEV), output_hidden_states=True).hidden_states
            H = torch.stack([h[0].float() for h in hs])                       # (25,T,1024)
            parts.append((len(seg), H.mean(1).cpu().numpy(), H.std(1).cpu().numpy()))
        w = np.array([p[0] for p in parts], dtype=np.float64); w /= w.sum()
        M.append(sum(wi * p[1] for wi, p in zip(w, parts)).astype(np.float16))
        S.append(sum(wi * p[2] for wi, p in zip(w, parts)).astype(np.float16))
        if i % 200 == 0:
            log(f'w2v-bert {i}/{N}')
np.save(os.path.join(OUT, 'emb_w2vbert_mean.npy'), np.stack(M))
np.save(os.path.join(OUT, 'emb_w2vbert_std.npy'), np.stack(S))
idx[['filename', 'split']].to_csv(os.path.join(OUT, 'index.csv'), index=False)
log('ALL DONE')

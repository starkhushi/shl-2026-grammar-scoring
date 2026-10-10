# SHL Hiring Assessment 2026 -- stage 8: zero-training grammar features from transcripts (GPU, internet)
"""Pure inference with ready-made open-source models on the stage-4 Whisper
transcripts -- nothing is trained here.

  1. CoLA acceptability (textattack/roberta-base-CoLA): P(grammatical) per
     sentence -> mean / min / share below 0.5 / whole-text probability.
  2. LLM rubric grader (Qwen2.5-7B-Instruct, 4-bit): the official SHL grammar
     rubric + transcript, then the next-token probabilities of "1".."5" give an
     expected score (continuous, one forward pass, no sampling). A second prompt
     asks for the number of grammatical errors (expected value over "0".."9").
     Zero-shot: no training labels go into any prompt, so CV stays clean.

Output: judge.csv (row order = index.csv of stage 4)
"""
import glob
import os
import re
import subprocess
import time

subprocess.run('pip install -q -U bitsandbytes', shell=True, check=False)

import numpy as np
import pandas as pd
import torch
from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification,
                          AutoTokenizer, BitsAndBytesConfig)

OUT = '/kaggle/working'
DEV = 'cuda'
t0 = time.time()
print(torch.cuda.get_device_name(0), flush=True)


def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(f'{pattern} not found under /kaggle/input')
    return hits[0]


text = pd.read_csv(find('text.csv'))
texts = text.text.fillna('').astype(str).tolist()
print('transcripts', len(texts), flush=True)
feats = pd.DataFrame({'filename': text.filename, 'split': text.split})


# ------------------------------------------------------------------ 1. CoLA
def sentences(t):
    s = [x.strip() for x in re.split(r'(?<=[.!?])\s+', t) if len(x.strip().split()) >= 2]
    return s or [t or '.']


tok = AutoTokenizer.from_pretrained('textattack/roberta-base-CoLA')
cola = AutoModelForSequenceClassification.from_pretrained('textattack/roberta-base-CoLA').to(DEV).eval()


@torch.no_grad()
def p_ok(batch):
    enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors='pt').to(DEV)
    return torch.softmax(cola(**enc).logits.float(), -1)[:, 1].cpu().numpy()   # label 1 = acceptable


rows = []
for t in texts:
    ss = sentences(t)
    p = np.concatenate([p_ok(ss[i:i + 64]) for i in range(0, len(ss), 64)])
    rows.append(dict(cola_mean=p.mean(), cola_min=p.min(), cola_bad=(p < 0.5).mean(),
                     cola_whole=float(p_ok([t or '.'])[0]), n_sent=len(ss),
                     sent_len=np.mean([len(s.split()) for s in ss])))
feats = pd.concat([feats, pd.DataFrame(rows)], axis=1)
del cola
torch.cuda.empty_cache()
print('cola done', f'{time.time() - t0:.0f}s', flush=True)

# -------------------------------------------------------------- 2. LLM judge
LLM = 'Qwen/Qwen2.5-7B-Instruct'
ltok = AutoTokenizer.from_pretrained(LLM)
llm = AutoModelForCausalLM.from_pretrained(
    LLM, device_map='cuda:0',
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                           bnb_4bit_quant_type='nf4')).eval()
print('llm loaded', f'{time.time() - t0:.0f}s', flush=True)

RUBRIC = """Grammar Score rubric for a spoken English response:
1 - struggles with sentence structure and syntax; limited control of simple grammatical structures and memorized patterns.
2 - limited understanding of sentence structure; uses simple structures but consistently makes basic grammatical mistakes; may leave sentences incomplete.
3 - decent grasp of sentence structure but makes errors in grammatical structure, or decent grammar but errors in syntax and structure.
4 - strong understanding of sentence structure and syntax; good control of grammar; occasional minor errors that do not cause misunderstanding.
5 - high grammatical accuracy and adept control of complex grammar; seldom makes noticeable mistakes; handles complex structures well."""

P_SCORE = (RUBRIC + "\n\nThe transcript below is automatic speech recognition of a ~1 minute spoken answer "
           "(punctuation was added by the recognizer, so ignore punctuation and spelling). "
           "Judge only the speaker's grammar.\n\nTranscript:\n\"\"\"{t}\"\"\"\n\n"
           "Reply with a single digit from 1 to 5.")
P_ERR = ("Count the grammatical errors (wrong tense, agreement, articles, prepositions, word order, "
         "incomplete sentences) made by the speaker in this transcript of spoken English. "
         "Ignore punctuation and spelling.\n\nTranscript:\n\"\"\"{t}\"\"\"\n\n"
         "Reply with a single digit from 0 to 9 (9 means 9 or more).")


def digit_ids(ds):
    ids = [ltok.encode(d, add_special_tokens=False) for d in ds]
    assert all(len(i) == 1 for i in ids), ids
    return [i[0] for i in ids]


ID15 = digit_ids('12345')
ID09 = digit_ids('0123456789')


@torch.no_grad()
def judge(prompt, ids, vals, bs=4):
    out = []
    for b in range(0, len(texts), bs):
        msgs = [ltok.apply_chat_template([{'role': 'user', 'content': prompt.format(t=t or '(silence)')}],
                                         tokenize=False, add_generation_prompt=True)
                for t in texts[b:b + bs]]
        ltok.padding_side = 'left'
        enc = ltok(msgs, return_tensors='pt', padding=True).to(DEV)
        logits = llm(**enc).logits[:, -1, :].float()
        p = torch.softmax(logits[:, ids], -1).cpu().numpy()
        out.append(p)
        if b % 200 == 0:
            print(f'  {b}/{len(texts)} {time.time() - t0:.0f}s', flush=True)
    P = np.concatenate(out)
    return P @ np.array(vals, dtype=float), P


ev, P = judge(P_SCORE, ID15, [1, 2, 3, 4, 5])
feats['llm_score'] = ev
for k in range(5):
    feats[f'llm_p{k + 1}'] = P[:, k]
ev, _ = judge(P_ERR, ID09, list(range(10)))
feats['llm_errors'] = ev
feats['llm_err_rate'] = ev / np.maximum(text.n_words.values, 1) * 100
print('llm done', f'{time.time() - t0:.0f}s', flush=True)

feats.to_csv(os.path.join(OUT, 'judge.csv'), index=False)
tr = pd.read_csv(find('Dataset_Final/train.csv'))
m = feats[feats.split == 'train'].merge(tr, on='filename')
m = m[m.label > 0]
print('\ncorrelation with label (train, real speech):')
print(m.drop(columns=['filename', 'split']).corr()['label'].drop('label').round(3).sort_values().to_string())
print('done', f'{time.time() - t0:.0f}s')

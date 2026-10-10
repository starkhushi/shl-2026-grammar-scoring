# Grammar scoring from speech — SHL Hiring Assessment 2026

Code for my entry to the [SHL Hiring Assessment 2026](https://www.kaggle.com/competitions/shl-hiring-assessment-2026)
on Kaggle. Each clip is 45–60 seconds of spontaneous spoken English and gets a grammar score on SHL's 1–5 MOS rubric;
the metric is RMSE.

**Public leaderboard: 0.3485 RMSE** (a constant prediction scores 1.005). Speaker-grouped cross-validation on the
732 real speech clips: 0.512 RMSE / 0.864 Pearson.

The full write-up, with the data checks, validation design, ablations and plots, is the notebook in
[`notebook/`](notebook/shl-2026-grammar-scoring-engine.ipynb). It is also published on Kaggle:
<https://www.kaggle.com/code/starkhushi/shl-2026-grammar-scoring-engine>.

## What the model is

Ridge and SVR regressors on frozen features, stacked with non-negative least squares:

```
wav ─┬─ WavLM-large (frozen)              layers 20/21, mean & std over time ─┐
     ├─ Whisper-large-v3 encoder (frozen)  layers 30–32                        ├─ per (model, layer): scaler → Ridge and SVR
     ├─ Whisper transcript ─┬─ RoBERTa-large / Qwen2.5-1.5B / Qwen2.5-7B ─────┤   (36 members, 15 speaker-grouped fits each)
     └─ CTC transcript ─────┴─ RoBERTa-large / Qwen2.5-7B ─────────────────────┤
     + Whisper-encoder LoRA fine-tune (same speaker folds) ─────────────────────┘
                                                                                │
        NNLS on out-of-fold predictions → linear calibration → twin smoothing → 0.8·stack + 0.2·WavLM fine-tune → spread ×1.1
```

Most of the score came from looking at the data rather than from the models:

- 37 training clips labelled 0 are synthetic white noise, not speech. I drop them and keep a detector as a guard.
- The training set repeats speakers (about half the clips have a same-voice twin) while every test speaker is new.
  Random K-fold rewards voice recognition, so all validation is grouped by speaker clusters.
- Two clips of the same speaker differ by about 0.47 in label (std), which works out to roughly 0.33 RMSE of rater
  noise per clip. The test labels are evidently cleaner, so I fit the members to speaker-mean targets and undo some of
  the shrinkage at the end. Both choices made CV slightly worse and the leaderboard better; they are the only two
  decisions I took on the public board.
- Whisper quietly repairs grammar while transcribing, so I also use a CTC transcript (wav2vec2, no language model)
  that keeps the repetitions and broken sentences the rubric penalises. Qwen2.5-7B reading that transcript is the
  best text member.

Things that did not help, all under the same grouped folds: a second speech encoder (Shuka), w2v-BERT 2.0,
segment-level training rows, LLM rubric grades and CoLA as stack inputs, Qwen2-Audio features, a non-linear stacker,
speaker-direction removal from the audio features, extra fold seeds, longer fine-tunes and a distilled fine-tune
(better CV, flat leaderboard — the write-up explains why). The scripts for those are kept in `experiments/`.

## Repository layout

```
notebook/    the report notebook: refits every model from the extracted features, prints CV and train RMSE, writes submission.csv
kernels/     the Kaggle GPU notebooks that extract the features the report uses (numbered in the order they run)
experiments/ everything I tried and dropped, kept for the record
submission/  the prediction file I submitted (public LB 0.3485)
```

### `kernels/` — feature extraction (run on Kaggle, T4)

| file | Kaggle kernel | what it produces |
|---|---|---|
| `01_audio_features_wavlm_whisper.py` | `shl26-v1b-feats` | WavLM-large (25 layers) and Whisper-large-v3 encoder (33 layers): per-layer mean and std over time |
| `02_text_features_whisper_roberta_qwen.py` | `shl26-v4-text` | Whisper transcripts; RoBERTa-large and Qwen2.5-1.5B layer states |
| `03_layer_sweep.py` | `shl26-v5-model` | ridge on every layer of every encoder under grouped CV |
| `04_judge_features_cola_qwen.py` | `shl26-v8-judge` | CoLA acceptability and a zero-shot Qwen2.5-7B rubric grade (sanity-check features) |
| `05_ctc_transcripts_qwen7b.py` | `shl26-v12-feats2` | wav2vec2 CTC transcripts; Qwen2.5-7B (4-bit) layer states on both transcripts; RoBERTa-large on the CTC text |
| `06_qwen7b_prompt_pooled.py` | `shl26-v14-feats3` | Qwen2.5-7B last-token states after a grammar question |
| `07_finetune_wavlm.py` | `shl26-v3-ft-wavlm` | end-to-end WavLM-large regressor (the 20 % blend) |
| `08_finetune_whisper_lora.py` | `shl26-v16-ft-whisper` | Whisper-large-v3 encoder + LoRA on the speaker-grouped folds (a stack member) |

## Reproducing

Everything runs on Kaggle; nothing needs a local GPU.

1. Create a notebook from each script in `kernels/`, in order, with the competition data attached and the GPU on.
   Scripts `05`–`08` also need the output of `01` attached (they reuse its row index and the speaker clustering).
2. Create a notebook from `notebook/shl-2026-grammar-scoring-engine.ipynb`, attach the competition data and the
   outputs of the eight kernels, and run it on CPU (about 15 minutes). It prints the grouped-CV and training RMSE and
   writes `submission.csv`.

The scripts locate their inputs by glob under `/kaggle/input`, so the exact mount paths don't matter. Every `.npy`
shares one row order (`index.csv`), and the speaker clustering is deterministic, so the folds are identical in every
notebook. No test label is used anywhere; the only test-side information is the test clips' own audio.

Python packages: see `requirements.txt` (standard Kaggle image plus `peft`-free LoRA — the fine-tune implements its
own LoRA layer because `peft` conflicted with the image's `torchao` at the time).

## Data

The competition data is not included (competition rules: competition use only). Get it from the Kaggle page.

## Licence

MIT.

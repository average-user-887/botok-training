# Training speed on Kaggle T4 (derived from the logs in logs/kaggle)

Hardware: Kaggle free T4, 14.56 GiB, fp16 + GradScaler AMP (no bf16), internet off, 12 h session cap.
Model: LFM3-1.7B-Base, LoRA on all-linear, gradient checkpointing, 1 row per forward (batch 1), grad accumulation, AdamW lr 2e-4, cosine with 5% warmup.
Token counts below are estimated (chars/3.6) because no tokenizer was available when this was written; treat as +-15%.

| run | rows | LoRA r | ACCUM | MAX_LEN | steps | train wall | s/step | s/row | est tok/row | est tok/s | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| deriver v1 | 297 (4 epochs = 1188 rows) | 32 | 8 | 6000 | 152 | ~1040 s | 6.9 | 0.86 | ~790 | ~900 | short windows, loss .0263 at end |
| deriver v2a run1 | 1061 of 1115 (<=7000 tok) | 32 | 8 | 7000 | 5 done of 399 | ~110 s | ~22 (steps 2-5) | ~2.8 | ~2450 | ~880 | OOM at step ~6 |
| pruner | 7963 (1.66 epochs seen) | 16 | 16 | 6000 | 827 of 996 | ~37700 s (budget hit) | 45.7 | 2.86 | not measured (private) | - | eval 1036 s on 500 rows |

Fixed overheads (not training): model load from the attached Kaggle dataset took 5394 s (v1), 1468 s (v2a), 717 s (pruner). Budget 30-90 minutes for load, plus 15-20 min for dev eval/saving. Adapter is saved every epoch.

Failure and fix: v2a run1 died with CUDA OOM (tried to allocate 2.58 GiB) at ~step 5 because full-sequence logits for a ~7k-token row in fp16/fp32 are huge. Fix: `model(..., logits_to_keep=len(completion)+1)` so only completion positions get logits (see code/train_deriver_v2.py).

## Planning numbers for a 12 h session
- Usable training time = 12 h - load (~0.4 h worst case 1.5 h) - eval/save (~0.4 h) ~= 10.4 h = 37,400 s (the script's TIME_BUDGET_S default).
- Measured rate ~880 tok/s on 2.4k-token rows; a conservative planning rate for 7k-token rows (attention + checkpointing recompute) is ~210-400 tok/s. Use 210 tok/s to be safe.
- Tokens per session  = 37,400 s x rate:  at 210 tok/s = 7.9 M tokens;  at 880 tok/s = 33 M tokens.
- Dataset tokens per epoch = rows x mean tokens/row (prompt ~450 static + windows + gold JSON; all tokens are forward/backward even though loss is on completion only).
- epochs = floor(tokens_per_session / tokens_per_epoch);  plan 3 epochs per part:  rows_per_part = tokens_per_session / (3 x mean tokens/row).
- Example: mean 2.4k tokens/row at 210 tok/s -> ~3,300 rows total across 3 epochs -> ~1,100 rows per part; at 880 tok/s ~4x more.
- Current v2 set: 1115 rows, ~2.6 M tokens/epoch -> 3 epochs ~ 7.9 M tokens, i.e. about 10.4 h at 210 tok/s or ~2.5 h at 880 tok/s.
- Rows over MAX_LEN 7000 tokens are skipped; keep long windows <= ~5.5k message tokens.
- Dot target: a synthetic set of ~1,000-1,500 additional rows (mean ~2.4k tokens, ~3-4 M tokens) per part keeps one part within one session.

## Extra throughput data from other Kaggle kernels (logs/kaggle)
Different models/tasks, so not directly comparable; useful as order-of-magnitude checks.

| kernel | GPU | model / method | measured rate | notes |
|---|---|---|---|---|
| memreader-bounded-sft-pilot | Kaggle GPU (4-bit NF4, fp16 compute) | 4B model, LoRA SFT, 48 steps, 2 epochs, 23 cases | ~24.4-24.6 s/step; train_runtime 1179 s (0.326 samples/s) | baseline eval (23 generations) 1068 s, candidate eval ~1010 s; total run 2910 s |
| kev-lab-native-calibration-and-pilot | Kaggle GPU, 1 device | ~4B hybrid model, LoRA + pointer head, 421 requests, 1 epoch | ~26.1-26.5 s/record (steady 26.0) | early local (Apple MPS) runs of the same recipe: 0.33-0.37 s/rec on a 0.6B model; local CUDA runs 0.09 s/rec |
| qwen-27b / glm TPU kernels | TPU (8 chips) | inference benchmarks | no training throughput | GLM run stopped at its 60 min test limit during weight loading; no tok/s logged |

## One T4 versus two
Kaggle's "T4 x2" runtime provides TWO T4 GPUs, but our current training script is a single process on cuda:0, so every rate above is per single T4 and the second GPU is idle. A 2-GPU data-parallel run (e.g. accelerate/DDP, one process per GPU, each holding a full copy of base + LoRA, ~7 GiB for r=32 at 7k tokens) could roughly double throughput (about 1.8-2x after gradient sync). That would double the tokens per 12 h part (e.g. ~15.8 M instead of ~7.9 M at the 210 tok/s planning rate). Not yet tried; the per-session load times and the 3-epoch-per-part plan above assume one GPU.

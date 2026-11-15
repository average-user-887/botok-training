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

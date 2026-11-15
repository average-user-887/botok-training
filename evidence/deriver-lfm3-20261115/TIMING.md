# TIMING

All MEASURED, single T4 (cuda:0), probe kernel, 2026-11-15. Raw: `raw/probe_summary.json` (phases), `raw/step_log.jsonl` (steps), `raw/probe_kernel_log.txt`.

## Phases
| Phase | Seconds | Note |
|---|---|---|
| setup_imports | 36.4 | |
| hash_inputs | 1.4 | |
| tokenizer load + tokenisation | 9.0 | 1190 rows |
| model_load | 766.8 | includes the cold read of the 3.46 GB weights from the dataset mount; cold read not separable |
| hash_weights (page cache warm) | 9.5 | NOT a cold-read time |
| peft_wrap_to_gpu | 1.8 | |
| eval_baseline | 512.7 | 7 dev rows, adapter disabled, every row ran to the 1800-token cap |
| training | 315.6 | 12 optimizer steps (8 effective, see RELIABILITY) |
| eval_candidate | 10.4 | same 7 dev rows |
| checkpoint_save | 0.275 | adapter + progress.json |
| resume_check | errored | CUDA OOM, no timing |
Total wall 1692.9 s (~28 min). Export/merge: NOT measured.

## Steps (accum 8 rows/step; "rows" = rows actually trained, the rest OOM-skipped)
| Step | s | rows | OOM-skipped | nonpad tok | supervised tok | loss | grad_norm | cuda0 max alloc MiB |
|---|---|---|---|---|---|---|---|---|
| 1 | 28.9 | 7 | 1 | 15483 | 3241 | 1.890 | null (scaler skip) | 13547 |
| 2 | 35.4 | 7 | 1 | 21341 | 4567 | 1.303 | null (scaler skip) | 13954 |
| 3 | 24.1 | 5 | 3 | 7946 | 1082 | 1.190 | null (scaler skip) | 13954 |
| 4 | 17.9 | 7 | 1 | 10907 | 1527 | 2.135 | null (scaler skip) | 13954 |
| 5 | 22.1 | 8 | 0 | 16012 | 2829 | 2.016 | 8.65 | 13954 |
| 6 | 22.2 | 7 | 1 | 13628 | 2256 | 1.274 | 6.97 | 13954 |
| 7 | 28.7 | 6 | 2 | 13890 | 2701 | 0.768 | 3.40 | 13954 |
| 8 | 24.8 | 8 | 0 | 17996 | 3088 | 0.801 | 1.00 | 13954 |
| 9 | 22.2 | 8 | 0 | 15868 | 2406 | 0.711 | 1.34 | 14307 |
| 10 | 26.5 | 8 | 0 | 19543 | 3666 | 0.556 | 1.76 | 14307 |
| 11 | 36.9 | 8 | 0 | 25265 | 4388 | 0.791 | 0.83 | 14307 |
| 12 | 26.0 | 7 | 1 | 16461 | 3328 | 0.572 | 0.72 | 14307 |

## Steady state (steps 3-12, i.e. after the two warm-up steps; 10 steps, 251.3 s)
- 25.13 s/step mean (median 25.4, range 17.9-36.9).
- 626.8 non-padding tokens/s; 108.5 supervised (completion) tokens/s; 0.287 trained rows/s (72 rows trained + 8 skipped in these steps, so 0.32 rows/s counting skipped attempts).
- All 12 steps: 315.6 s, 26.3 s/step, 615.8 nonpad tok/s, 111.2 supervised tok/s.
- Step time tracks tokens in the step (step 11: 25265 tok, 36.9 s; step 4: 10907 tok, 17.9 s), so s/step depends on which rows land in a step. 12 steps over a length-biased 96-row sample is a small sample: treat the numbers as +/- 15%.
- "Nonpad" counts prompt + completion tokens of trained rows (batch 1, so no padding in practice). Supervised counts completion tokens only. Skipped rows are excluded from both.

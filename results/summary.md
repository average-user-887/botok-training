# Evaluation summaries (aggregate numbers only)

## Gate for a specialist model going into production (deriver)
schema-valid >= 99%, hard-fail <= 1%, F1 >= 0.70 (embedding match cos >= 0.80), hallucination <= 5%, p95 latency <= 60 s.
No tested model passes. Best untuned small model by F1: gemma4-e2b (0.29, but 41-47% schema-valid).

## Tuned LFM3-1.7B deriver (LoRA r=32, v1 = 330 synthetic windows, 4 epochs, 152 steps)
| eval set | n | valid | hard-fail | recall | precision | F1 | halluc | p50 lat | VRAM |
|---|---|---|---|---|---|---|---|---|---|
| held-out synthetic dev (Jaccard match, Kaggle) | 33 | 1.00 | - | 0.557 | 0.562 | - | - | - | - |
| real private fixtures (embedding match, P2000, Q4_K_M) | 160 | 0.98 | 0.04 | 0.03 | 0.20 | 0.15 | 0.01 | 2.1 s | not recorded (model Q4_K_M ~ 4.2 GiB incl. KV, fits 5 GiB) |
Reading: format is learned (valid .98) but recall is near zero on long real windows (distribution mismatch: v1 windows were short, ~790 tokens; real windows are long). v2 data (1115 rows, up to 7k tokens) is the fix being trained.
Base LFM3-1.7B without tuning: 61% schema-valid, F1 0.08 (see leaderboard).

## Tuned LFM3-1.7B pruner (LoRA r=16, 7963 train rows, 827 of 996 planned steps, time budget hit)
Task: given GOAL + numbered tool output lines, emit `KEEP 3-9,12` / `KEEP none` / `KEEP all`.
| eval | n | recall (micro) | precision (micro) | removed token share | malformed | p50 | p95 |
|---|---|---|---|---|---|---|---|
| Kaggle dev, fp16 LoRA | 500 | 0.548 | 0.556 | 0.73 | 0.038 | - | - |
| P2000 dev, Q4_K_M GGUF | 500 | 0.595 | 0.532 | 0.70 | 0.052 | 2.3 s | 5.0 s |
| P2000 sealed test, Q4_K_M GGUF | 500 | 0.421 | 0.606 | 0.73 | 0.010 | 1.4 s | 5.1 s |
Malformed output counts as keep-all. Base LFM3 (no tuning) on the same dev/test: malformed 100% (never produces the KEEP format), so recall 1.0 / precision 0.21-0.29 is an artifact.

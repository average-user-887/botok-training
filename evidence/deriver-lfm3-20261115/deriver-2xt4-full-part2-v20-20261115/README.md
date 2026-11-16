# Deriver LoRA (LFM3-1.7B-Base) - 2xT4 DDP Full Run Part 2 (v20) Evidence Packet, 2026-11-16

This directory contains sanitized training receipts, GPU telemetry, decile milestone logs, dev metrics, and held-out GGUF evaluation results for Full Run Part 2 (`<kaggle-user>/p2000-deriver-v15-full-part2`), executed on 2x Tesla T4 GPUs (`cuda:0` and `cuda:1`, `world_size=2`) via PyTorch `torchrun` DDP.

## Executive Summary

- **Kernel Reference**: `<kaggle-user>/p2000-deriver-v15-full-part2`
- **Kernel Status**: `KernelWorkerStatus.COMPLETE` (post-training NCCL timeout fix verified: `dist.destroy_process_group()` before Rank 0 final 24-row eval; `ddp_timeout=3600`)
- **DDP Verification**: `true` (`PER_RANK_STEPS_AND_ALLOCATIONS_OBSERVED`, `steps: [147, 147]`, `rows_processed: [588, 588]`, `known_distinct_uuids: true`)
- **Dataset**: `honcho-deriver-v20-training-only` (`392` train rows, `1,132,018` tokens/epoch, `420,575` supervised tokens/epoch, `max_length=5600`)
- **Init & Hyperparameters**: initialized from Part 1 `checkpoint-pct-80`, `role=deriver`, `lora_r=16`, `lora_alpha=32`, `target_modules=attn-mlp`, `lr=1e-4` (0.5x adjustment), `epochs=3`, `effective_batch=8`, `microbatch=1`
- **Optimizer Steps**: `147 / 147` planned steps completed (`remaining_optimizer_steps = 0`, `stop_requested = false`)
- **P95 Step Latency**: `21.17 s/step` (`21.166 s`)
- **Elapsed Time**: `2,880.88 s` training (`474.19 s` across 4 milestone probes; `3,875.38 s` total script runtime)
- **Peak VRAM Allocated**: GPU 0: `12,146.03 MiB`, GPU 1: `12,146.00 MiB` (`12,502 / 12,504 MiB` PyTorch reserved)
- **Peak VRAM (nvidia-smi)**: GPU 0: `12,845 MiB`, GPU 1: `12,743 MiB` (`100%` peak util on both GPUs across `3,910` samples)
- **OOM / NaN Grad Skips**: `0` OOM skips, `0` NaN grad skips
- **Selected Checkpoint**: final `adapter/` (step `147` / 100%, `dev_f1 = 0.2145`, `dev_schema_valid = 0.7500`, `non_empty_pred_rate = 0.3750`, `empty_fp_rate = 0.0`)
- **GGUF LoRA Export & Held-Out Eval (`honcho-deriver-v14-evaluation-only`, 4 cases, CPU Q4_K_M + LoRA, `n_predict=1200`)**:
  - `schema_valid = 0.5000` (2/4 EOS; 2 medium rows hit `n_predict=1200` limit with `token_f1` `0.8192` / `0.7941`)
  - `non_empty_rate = 0.2500`
  - `f1 = 0.2205` (`precision = 0.4474`, `recall = 0.1463`) — improves over Part 1 `deriver-v18-lora.gguf` (`0.1815`, `+0.0390 F1`)
  - `token_f1 = 0.8535`
  - `empty_fp_rate = 0.0000` (0 false positives on `evfourteen-empty-4`, `token_f1 = 1.0000`)
  - `total_elapsed_s = 167.71` (`avg_latency_s = 41.93`)

## Decile Progression & Final Dev Eval

Mid-training milestone probes (`20%, 40%, 60%, 80%`) used `milestone_probe_rows=12` and `milestone_max_new_tokens=256`, which truncated positive JSON outputs before EOS (`dev_f1=0.0`, only the 5 empty-gold rows completed within 256 tokens -> `dev_schema_valid=0.4167`). Final evaluation of `adapter/` (step `147`, 24 dev rows, `eval_max_new_tokens=1800`) confirmed healthy non-empty structured output:

| Checkpoint | Step | Train Loss | Schema Valid | Precision | Recall | Dev F1 | Token F1 | Empty FP | Non-Empty Rate |
|---|---|---|---|---|---|---|---|---|---|
| 20% (`max_new=256`) | 30 | 0.2662 | 0.4167 | 0.0000 | 0.0000 | 0.0000 | 0.5494 | 0.0000 | 0.0000 |
| 40% (`max_new=256`) | 59 | 0.1111 | 0.4167 | 0.0000 | 0.0000 | 0.0000 | 0.5424 | 0.0000 | 0.0000 |
| 60% (`max_new=256`) | 89 | 0.2135 | 0.4167 | 0.0000 | 0.0000 | 0.0000 | 0.5408 | 0.0000 | 0.0000 |
| 80% (`max_new=256`) | 118 | 0.1919 | 0.4167 | 0.0000 | 0.0000 | 0.0000 | 0.5454 | 0.0000 | 0.0000 |
| **Final `adapter/` (`max_new=1800`)** | **147** | **-** | **0.7500** | **0.3074** | **0.1647** | **0.2145** | **0.6540** | **0.0000** | **0.3750** |

## Part 1 vs Part 2 Held-Out Comparison (`honcho-deriver-v14-evaluation-only`)

| Adapter | Init / Dataset | Schema Valid | Precision | Recall | Held-Out F1 | Token F1 | Empty FP |
|---|---|---|---|---|---|---|---|
| `deriver-v18-lora.gguf` (Part 1 `pct-80`) | Base / v18 (352 rows) | 0.5000 | 0.4483 | 0.1138 | 0.1815 | 0.8555 | 0.0000 |
| **`deriver-v20-lora.gguf` (Part 2 `adapter/`)** | **`pct-80` / v20 (392 rows)** | **0.5000** | **0.4474** | **0.1463** | **0.2205** | **0.8535** | **0.0000** |

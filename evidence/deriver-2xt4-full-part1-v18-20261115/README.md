# Deriver LoRA (LFM3-1.7B-Base) - 2xT4 DDP Full Run Part 1 (v18) Diagnostic & Decile Evidence, 2026-11-15

This directory contains sanitized benchmark logs, GPU telemetry, decile milestone checkpoints, and root-cause diagnostics for Full Run Part 1 (`<kaggle-user>/p2000-deriver-v15-full-part1`) executed on 2x Tesla T4 GPUs (`cuda:0` and `cuda:1`, `world_size=2`) using PyTorch `torchrun` with Distributed Data Parallel (DDP).

## Executive Summary

- **Kernel Reference**: `<kaggle-user>/p2000-deriver-v15-full-part1`
- **Execution Target**: 2x Tesla T4 GPUs (`cuda:0`, `cuda:1`, `world_size=2`)
- **Kernel Status**: `KernelWorkerStatus.ERROR` (post-training Rank 1 NCCL barrier timeout; **training itself completed 220 / 220 optimizer steps**)
- **DDP Verification**: `true` (`PER_RANK_STEPS_AND_ALLOCATIONS_OBSERVED`, both ranks completed 220 optimizer steps)
- **Dataset**: `honcho-deriver-v18-training-only` (`352` train rows, `996,841` tokens/epoch, `361,524` supervised tokens/epoch, `max_length=5600`)
- **Hyperparameters**: `role=deriver`, `lora_r=16`, `lora_alpha=32`, `target_modules=attn-mlp`, `lr=2e-4`, `epochs=5`, `effective_batch=8`, `microbatch=1`
- **Optimizer Steps**: `220 / 220` planned steps completed (`remaining_optimizer_steps = 0`)
- **Rows Processed**: `1,760` row-passes total (`880` rows per GPU, exact parity)
- **P95 Step Latency**: `23.81 s/step` (`23.806 s`)
- **Elapsed Training Time**: `5,172.8 s` (`1,335.2 s` total across 9 milestone evals)
- **Peak VRAM Allocated**: GPU 0: `12,146.04 MiB`, GPU 1: `12,146.00 MiB` (`12,504 MiB` PyTorch reserved)
- **Peak VRAM (nvidia-smi)**: GPU 0: `12,745 MiB`, GPU 1: `12,743 MiB` (`~2.8 GiB` headroom preserved)
- **OOM / NaN Grad Skips**: `0` OOM skips, `0` NaN grad skips
- **Peak Decile Checkpoint**: `checkpoint-pct-80` (step `176`, `dev_f1 = 0.2733`, `dev_precision = 0.6190`, `dev_recall = 0.1753`, `dev_token_f1 = 0.6646`, `dev_schema_valid = 0.6250`, `empty_fp_rate = 0.0`)
- **GGUF LoRA Export & Held-Out Eval (`honcho-deriver-v14-evaluation-only`, 4 cases, CPU Q4_K_M + LoRA, `n_predict=1200`)**:
  - `schema_valid = 0.5000` (2/4 EOS; 2 medium rows hit `n_predict=1200` limit with `token_f1 > 0.81`)
  - `f1 = 0.1815` (`precision = 0.4483`, `recall = 0.1138`)
  - `token_f1 = 0.8555`
  - `empty_fp_rate = 0.0000` (0 false positives on `evfourteen-empty-4`, `token_f1 = 1.0000`)

## Root Cause of `KernelWorkerStatus.ERROR`

1. All `220 / 220` training steps (`5.0` epochs) finished cleanly at `t = 5247.9s` (`train_loss = 0.3266`).
2. Both ranks synchronized at the post-training barrier, and Rank 0 saved `adapter/`, `checkpoint-220/`, `checkpoint-latest/`, `budget.json`, `rank-{0,1}-activity.json`, `token-summary.json`, and `environment-freeze.txt`.
3. Rank 0 then began the final 24-row generation evaluation (`evaluate_dev_rows(..., max_new_tokens=1800)`) while Rank 1 entered the final `torch.distributed.barrier()` at the end of `main()`.
4. Because Rank 0's 24-row generation exceeded PyTorch/NCCL's default `600,000 ms` (10-minute) collective timeout, Rank 1's NCCL watchdog aborted at `t = 5908.6s` (`Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1821, OpType=ALLREDUCE, Timeout(ms)=600000)` -> `SIGABRT`), terminating `torchrun` before `dev_metrics.json` was written and before notebook cell 3 ran.
5. **Required Fix Before Next Run**: Move `evaluate_dev_rows` after `torch.distributed.destroy_process_group()` (or pass `timeout=datetime.timedelta(minutes=60)` to `init_process_group` / `ACCELERATE` / `TrainingArguments` `ddp_timeout=3600`, or run Rank 0 final eval after Rank 1 cleanly exits DDP).

## Decile Progression (`milestones.jsonl`, 8-row stratified probe, `max_new_tokens=600`)

| Decile | Step | Train Loss | Schema Valid | Precision | Recall | Dev F1 | Token F1 | Empty FP Rate | Long Window Recall | Non-Empty Rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 10% | 22 | 0.5750 | 1.000 | 0.0000 | 0.0000 | 0.0000 | 0.2670 | 0.000 | 0.000 | 0.000 |
| 20% | 44 | 0.5304 | 0.625 | 0.5098 | 0.1688 | 0.2537 | 0.6848 | 0.000 | 0.000 | 0.250 |
| 30% | 66 | 0.3518 | 0.625 | 0.5854 | 0.1623 | 0.2542 | 0.6666 | 0.000 | 0.000 | 0.250 |
| 40% | 88 | 0.3622 | 0.625 | 0.6000 | 0.1623 | 0.2555 | 0.6820 | 0.000 | 0.000 | 0.250 |
| 50% | 110 | 0.2856 | 0.625 | 0.6098 | 0.1623 | 0.2564 | 0.6689 | 0.000 | 0.000 | 0.250 |
| 60% | 132 | 0.2475 | 0.625 | 0.5714 | 0.0779 | 0.1371 | 0.5502 | 0.000 | 0.000 | 0.125 |
| 70% | 154 | 0.2911 | 0.625 | 0.6000 | 0.1623 | 0.2555 | 0.6681 | 0.000 | 0.000 | 0.250 |
| **80%** | **176** | **0.2412** | **0.625** | **0.6190** | **0.1753** | **0.2733** | **0.6646** | **0.000** | **0.000** | **0.250** |
| 90% | 198 | 0.2162 | 0.625 | 0.6098 | 0.1688 | 0.2644 | 0.6588 | 0.000 | 0.000 | 0.250 |

### Observations & Decile Rule Assessment
- **Schema validity (`0.625` = 5/8)** and **`long_window_recall` (`0.0`)** during milestone probes are partially constrained by `--milestone-max-new-tokens 600` (long windows with 800+ gold output tokens truncate before EOS, failing `finished == True`).
- **Peak decile**: `80%` (step 176) achieved the highest `dev_f1` (`0.2733`) and precision (`0.6190`), slightly outperforming `90%` (`0.2644`) while `train_loss` continued decreasing (`0.2412` -> `0.2162`).
- **New Google Drive Drop**: `honcho-deriver-v19-training-only` (`368` train windows, `1,054,115` tokens/pass, adding 16 new windows in `3,430-3,810` token range) is available for Part 2.

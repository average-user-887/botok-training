# Deriver LoRA (LFM3-1.7B-Base) - 2xT4 DDP Trial Run 1 Evidence, 2026-11-15

This directory contains benchmark logs, telemetry, and evaluation results for Trial Run 1 (`<kaggle-user>/p2000-deriver-v15-sprint-a1`) executed on 2x Tesla T4 GPUs (`cuda:0` and `cuda:1`, `world_size=2`) using PyTorch `torchrun` with Distributed Data Parallel (DDP).

## Executive Summary

- **Kernel Reference**: `<kaggle-user>/p2000-deriver-v15-sprint-a1`
- **Execution Target**: 2x Tesla T4 GPUs (`cuda:0`, `cuda:1`, `world_size=2`)
- **Kernel Status**: `KernelWorkerStatus.COMPLETE`
- **DDP Verification**: `true` (`PER_RANK_STEPS_AND_ALLOCATIONS_OBSERVED`, both ranks completed 24 optimizer steps)
- **Optimizer Steps**: 24 / 24 planned steps completed (100% complete)
- **Rows Processed**: 192 rows total (96 rows per GPU, balanced)
- **P95 Step Latency**: 29.39 s/step
- **Peak VRAM Allocated**: GPU 0: 12,201.78 MiB, GPU 1: 12,143.64 MiB
- **Peak VRAM (nvidia-smi)**: GPU 0: 12,791 MiB, GPU 1: 12,793 MiB (~2.8 GiB headroom preserved)
- **OOM / NaN Grad Skips**: 0 OOM skips, 0 NaN grad skips
- **Dev Evaluation (24 rows)**:
  - Schema Validity: 95.83% (23 / 24)
  - Dev F1: 0.1786
  - Dev Token F1: 0.4260
  - Precision: 0.2355, Recall: 0.1439
  - Non-empty Prediction Rate: 29.17% (7 / 24)
  - Empty Collapse Detected: `false`
- **Checkpoints & Artifacts**: Saved at steps 10, 20, 24, milestones 25%, 50%, 75%, and final adapter.
- **Safety Preconditions for Trial 2**: All passed (Kernel COMPLETE, DDP verified, zero fatal OOMs).

## Comparison: Claude 1xT4 Probe Baseline vs 2xT4 DDP Trial Run 1

| Metric / Dimension | Claude 1xT4 Probe Baseline (`deriver-lfm3-20261115`) | 2xT4 DDP Trial Run 1 (`deriver-2xt4-trial1-v15-20261115`) | Relative Delta / Assessment |
|---|---|---|---|
| **GPU Setup** | 2x Tesla T4 allocated, but **GPU1 idle** (only 1 GPU trained) | **2x Tesla T4 actively utilized in DDP** (`cuda:0`, `cuda:1`, `world_size=2`) | Dual-GPU DDP verified by `verify_ddp.py` |
| **Execution Status** | Completed 12 probe steps (custom loop) | Completed all 24 planned optimizer steps (`torchrun`) | 100% completion of planned training schedule |
| **P95 Step Latency** | 25.1 s/step (steady state steps 3–12) | 29.39 s/step (p95 across 24 optimizer steps) | Stable multi-GPU accumulation with gradient sync |
| **Optimizer Steps** | 12 attempted, 8 effective | **24 completed steps on both ranks** (48 rank-steps) | +200% effective steps trained |
| **Rows Processed** | 96 rows attempted (10 skipped due to OOM) | **192 rows processed** (96 rows per GPU, balanced) | 2.0x data throughput; zero dropped rows |
| **OOM Skips in Loop** | 10 rows skipped due to in-loop OOM catches | **0 rows / 0 steps skipped** (clean execution) | 100% clean memory allocation |
| **GradScaler Skips** | 4 / 12 steps skipped by scaler | **0 steps skipped** by scaler | Zero NaN gradients; stable FP16 scaling |
| **Peak VRAM Allocated** | GPU0: 14,307 MiB (GPU1 idle: 3 MiB) | GPU0: 12,201.78 MiB, GPU1: 12,143.64 MiB | Perfectly balanced memory load across GPUs |
| **Peak VRAM Reserved (smi)** | GPU0: 14,891 MiB (~97.8% capacity) | GPU0: 12,791 MiB, GPU1: 12,793 MiB (~84.1% capacity) | ~2.8 GiB safety headroom preserved on both cards |
| **GPU Utilization** | GPU0 ~96.7%, GPU1 0.0% | **Both GPU0 and GPU1 actively utilized** | Full cluster compute saturation |
| **Dev Schema Validity** | 1.0 (7 rows, but collapsed into `{}`) | **0.9583 (23 / 24 valid JSON schemas)** | High structural adherence on full 24-row set |
| **Dev F1 Score** | 0.0 (empty prediction collapse) | **0.1786** (Token F1: **0.4260**) | Emergence of derivation extraction capability |
| **Non-Empty Prediction Rate** | 0.0% (empty prediction collapse) | **29.17%** (7 / 24 non-empty predictions) | Successfully broke empty prediction collapse |
| **Precision / Recall** | Precision: 0.0, Recall: 0.0 | Precision: 0.2355, Recall: 0.1439 | Short window recall reached 0.3580 |
| **Milestone Checkpoints** | None (Resume check failed with OOM) | Saved at steps 10, 20, 24, milestones 25%, 50%, 75% | Full checkpoint trail exported and verified |

## Detailed Analysis

### 1. Dual-GPU DDP Verification
The verification script `verify_ddp.py` verified the rank receipts (`rank-0-activity.json`, `rank-1-activity.json`, `rank-0-identity.json`, `rank-1-identity.json`):
- `world_size = 2`, `logical_devices = [0, 1]`, `pids = [55, 56]`
- Distinct physical GPU UUIDs confirmed (`<gpu-0-uuid>` and `<gpu-1-uuid>`)
- Rank 0 steps: 24, Rank 1 steps: 24 (exact parity)
- Rank 0 rows: 96, Rank 1 rows: 96 (exact parity)
- `oom_steps_skipped = 0`, `nan_grad_steps_skipped = 0`
- `skips_clean = true`

### 2. Throughput and Efficiency
- **Token Processing**:
  - Rank 0: 246,726 non-pad tokens, 81,863 supervised tokens
  - Rank 1: 241,260 non-pad tokens, 78,379 supervised tokens
  - Combined: 487,986 non-padding tokens, 160,242 supervised tokens processed
- **Elapsed Time**: 1,053.4 s total elapsed (546.1 s training, 209.6 s milestone evals), completing comfortably within the 1,100 s time budget.

### 3. Memory Footprint and Stability
By capping `max_seq_len` to 5,600 and calibrating microbatching to 1 with 4 gradient accumulation steps per rank:
- Peak PyTorch allocation stayed at ~12.20 GiB (GPU 0) and ~12.14 GiB (GPU 1).
- System-level VRAM peaked at 12,791 MiB and 12,793 MiB, retaining ~2.8 GiB headroom below the 14.56 GiB T4 limit.
- No backward pass activation spikes exceeded memory limits, completely resolving the Step 8 OOM failure seen in earlier sprint configurations.

### 4. Convergence & Dev Metrics
- **Milestone Progression**:
  - Milestone 25% (step 6): train loss 0.7093, dev schema valid 0.25, dev F1 0.1176, token F1 0.2907
  - Milestone 50% (step 12): train loss 0.2622, dev schema valid 1.0, dev F1 0.0 (temporary empty collapse), token F1 0.2670
  - Milestone 75% (step 18): train loss 0.7669, dev schema valid 1.0, dev F1 0.1669, token F1 0.3688
  - Final (step 24): dev schema valid 0.9583, dev F1 0.1786, token F1 0.4260, non-empty rate 29.17%, empty collapse detected: `false`

## Evidence File Index

| File | Purpose |
|---|---|
| `README.md` | Executive summary, Claude 1xT4 vs 2xT4 comparison table, and analysis |
| `budget.json` | Step timings, pacing, sequence lengths, and budget consumption |
| `ddp_verification.json` | Programmatic verification output from `verify_ddp.py` |
| `dev_metrics.json` | 24-row evaluation metrics (schema validity, precision, recall, F1, token F1) |
| `gpu_telemetry.csv` | Redacted 2-second interval GPU utilization and VRAM telemetry |
| `kernel.log` | Redacted full stdout/stderr log stream from Kaggle execution |
| `milestones.jsonl` | Periodic evaluation checkpoints at 25%, 50%, and 75% training progress |
| `rank-0-activity.json` | Redacted activity and token accounting receipt for GPU rank 0 |
| `rank-0-identity.json` | Redacted hardware identity receipt for GPU rank 0 |
| `rank-1-activity.json` | Redacted activity and token accounting receipt for GPU rank 1 |
| `rank-1-identity.json` | Redacted hardware identity receipt for GPU rank 1 |
| `run-lock.json` | Workload fingerprint and parameter lock |
| `sprint_summary.json` | Multi-arm summary receipt verifying trial winner |

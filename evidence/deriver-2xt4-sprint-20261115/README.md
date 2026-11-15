# Deriver LoRA (LFM3-1.7B-Base) - 2xT4 DDP Sprint Run Evidence, 2026-11-15

This directory contains benchmark and telemetry evidence from Sprint A1: a multi-GPU distributed data parallel (DDP) run of `train_deriver.py` on 2x Tesla T4 GPUs (`cuda:0` and `cuda:1`, `world_size=2`) using PyTorch `torchrun`.

## Executive Summary

The 2xT4 DDP configuration successfully activated and balanced compute across both Tesla T4 GPUs, achieving a **1.49x speedup per step** (16.86 s/step p95 vs 25.1 s/step steady-state on 1xT4) with **zero OOM skips** and **zero NaN grad skips** during active training steps. At step 7 (20% milestone), dev schema validity reached 1.0 (with empty prediction collapse, dev F1 0.0). At step 8, an uncheckpointed long sequence in the backward pass exceeded the 14.56 GiB T4 memory limit (+2.44 GiB allocation attempt), triggering a CUDA OOM.

## Comparison: 2xT4 DDP Sprint vs Claude's 1xT4 Probe Baseline

| Metric / Dimension | Claude 1xT4 Probe Baseline (`deriver-lfm3-20261115`) | 2xT4 DDP Sprint Run (`deriver-2xt4-sprint-20261115`) | Relative Delta / Assessment |
|---|---|---|---|
| **GPU Setup** | 2x Tesla T4 allocated, but **GPU1 idle** (only 1 GPU trained) | **2x Tesla T4 actively utilized in DDP** (`cuda:0`, `cuda:1`, `world_size=2`) | Dual-GPU DDP verified by `verify_ddp.py` |
| **P95 Step Latency** | 25.1 s/step (steady state steps 3–12) | **16.86 s/step** (p95 across optimizer steps) | **1.49x speedup** (-32.8% step latency) |
| **Optimizer Steps** | 12 total steps attempted, 8 effective steps | 7 optimizer steps completed on both ranks | Completed up to step 7 (20% milestone) |
| **Rows Processed** | 96 rows attempted (10 skipped due to OOM) | 56 rows processed (28 rows per GPU, balanced) | Perfectly balanced rank-0 and rank-1 data |
| **OOM Skips in Loop** | 10 rows skipped due to in-loop OOM catches | **0 rows / 0 steps skipped** (clean execution) | No skipped steps during active run |
| **GradScaler Skips** | 4 / 12 steps skipped by scaler | **0 steps skipped** by scaler | Stable gradient scaling |
| **Peak VRAM Allocated** | GPU0: 14,307 MiB (GPU1 idle: 3 MiB) | GPU0: 13,837 MiB, GPU1: 13,903 MiB | Dual-GPU memory utilization balanced |
| **Peak VRAM Reserved** | GPU0: ~14,891 MiB (nvidia-smi) | GPU0: 14,256 MiB, GPU1: 14,372 MiB (14,613 MiB smi) | High VRAM occupancy (~97% of 14.56 GiB) |
| **GPU Utilization** | GPU0 ~96.7%, GPU1 0.0% | **Both GPU0 and GPU1 at 99–100%** utilization | Both GPUs fully saturated during compute |
| **Milestone 10% (Step 4)** | N/A | loss: 0.8271, dev_schema_valid: 0.0, F1: 0.0 | Initial training phase |
| **Milestone 20% (Step 7)** | N/A (Candidate at step 12: schema-valid 1.0, F1 0.0) | loss: 0.4433, **dev_schema_valid: 1.0**, F1: 0.0 | Rapid convergence to schema valid format |
| **Failure Mode** | In-loop OOM skips & scaler instability | Step 8 backward OOM (+2.44 GiB allocation attempt) | Clear fix: enable gradient checkpointing |

## Detailed Analysis

### 1. Dual-GPU DDP Verification
The verification script `verify_ddp.py` inspected both rank receipts (`rank-0-activity.json`, `rank-1-activity.json`, `rank-0-identity.json`, `rank-1-identity.json`) and verified:
- `world_size = 2`, `pids = [55, 56]`, `logical_devices = [0, 1]`
- Distinct physical GPU UUIDs confirmed
- Rank 0 steps: 7, Rank 1 steps: 7 (exact parity)
- Rank 0 rows: 28, Rank 1 rows: 28 (exact parity)
- `oom_steps_skipped = 0`, `nan_grad_steps_skipped = 0`
- `skips_clean = true`

### 2. Throughput and Efficiency
- **Step Latency**: P95 step time of 16.86 seconds per optimizer step (with `effective-batch 8`, `microbatch 1`, distributed across 2 GPUs = 4 gradient accumulation steps per GPU per optimizer step).
- **Token Processing**:
  - Rank 0: 61,877 non-padding tokens, 19,114 supervised tokens
  - Rank 1: 69,397 non-padding tokens, 21,993 supervised tokens
  - Combined: 131,274 non-padding tokens processed across 7 steps (~1,113 non-pad tok/s total throughput across the cluster, almost doubling the 1xT4 throughput of 627 tok/s).

### 3. Step 8 Failure Diagnosis
At optimizer step 8, during the backward pass of a microbatch with length close to `max_seq_len = 6400`:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.44 GiB. 
GPU 0 has a total capacity of 14.56 GiB of which 2.31 GiB is free. 
Including non-PyTorch memory, this process has 12.25 GiB memory in use. 
Of the allocated memory 11.89 GiB is allocated by PyTorch, and 125.73 MiB is reserved by PyTorch but unallocated.
```
This demonstrates that while forward activation fitting was managed, the uncheckpointed backward graph for max-length tokens (6,400 tokens) requires an additional ~2.4 GiB of activation memory, which exceeds the remaining 2.31 GiB headroom on the 14.56 GiB T4 card.

**Recommended Action for Sprint A2:**
1. Enable PyTorch gradient checkpointing (`model.gradient_checkpointing_enable()` / `--gradient-checkpointing`).
2. Alternatively, set `max_seq_len` to 5600 or enable dynamic sequence packing / bucketing to prevent max-length activation spikes during backward pass.

## Evidence File Index

| File | Purpose |
|---|---|
| `README.md` | Benchmark summary and comparative analysis vs Claude 1-GPU baseline |
| `budget.json` | Run timing budget, P95 step latency, planned steps, token statistics |
| `rank-0-activity.json` | Rank 0 training steps, rows, tokens, and peak memory allocation |
| `rank-1-activity.json` | Rank 1 training steps, rows, tokens, and peak memory allocation |
| `rank-0-identity.json` | Rank 0 process identity, logical device, GPU metadata |
| `rank-1-identity.json` | Rank 1 process identity, logical device, GPU metadata |
| `ddp_verification.json` | Structured output from `verify_ddp.py` |
| `milestones.jsonl` | Milestone evaluations at 10% (step 4) and 20% (step 7) |
| `dev_metrics.json` | Step 7 milestone evaluation metrics |
| `gpu_telemetry.csv` | Per-second GPU utilization and memory telemetry for both GPUs |
| `kernel.log` | Complete sanitized stdout and stderr execution log |
| `run-lock.json` | Training hyperparameter and recipe configuration lock |

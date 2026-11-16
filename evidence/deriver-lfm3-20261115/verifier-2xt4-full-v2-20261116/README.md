# Verifier (Log-Analyser) LoRA (LFM3-1.7B-Base) - 2xT4 DDP Full Run v2 Evidence Packet, 2026-11-16

This directory contains sanitized training receipts, GPU telemetry, milestone logs, dev metrics, and CPU GGUF smoke inference results for Verifier Full Run v2 (`<kaggle-user>/p2000-verifier-v2-full`), executed on 2x Tesla T4 GPUs (`cuda:0` and `cuda:1`, `world_size=2`) via PyTorch `torchrun` DDP.

## Executive Summary

- **Kernel Reference**: `<kaggle-user>/p2000-verifier-v2-full`
- **Kernel Status**: `KernelWorkerStatus.COMPLETE`
- **DDP Verification**: `true` (`PER_RANK_STEPS_AND_ALLOCATIONS_OBSERVED`, `steps: [36, 36]`, `rows_processed: [126, 126]`, `known_distinct_uuids: true`)
- **Dataset**: combined reviewed Verifier (`log-analyser`) `train.sft.jsonl` (`42` rows: `si-lab-log-analyser-proposed-candidate-v1` 10 + `si-lab-log-analyser-candidate-v2-new24` 20 + `si-lab-four-role-reviewed-increment-52-20261115/log-analyser` 12; `30,933` tokens/epoch, `11,686` supervised tokens/epoch) & `dev.sft.jsonl` (`6` rows: `v1` 2 + `v2` 4), `max_length=5600`
- **Hyperparameters**: `role=verifier`, `lora_r=16`, `lora_alpha=32`, `target_modules=attn-mlp`, `lr=2e-4`, `epochs=6`, `effective_batch=8`, `microbatch=1`
- **Optimizer Steps**: `36 / 36` planned steps completed (`remaining_optimizer_steps = 0`, `stop_requested = false`)
- **P95 Step Latency**: `3.91 s/step` (`3.9103 s`)
- **Elapsed Time**: `463.12 s` training (`346.78 s` across 4 milestone probes; `628.5 s` `torchrun` wall; `568.44 s` script runtime)
- **Peak VRAM Allocated**: GPU 0: `4,322.45 MiB`, GPU 1: `4,322.43 MiB` (`4,482 MiB` / `4,564 MiB` PyTorch reserved)
- **Peak VRAM (nvidia-smi)**: GPU 0: `4,723 MiB`, GPU 1: `4,803 MiB` (`100%` peak util on both GPUs across `314` samples each)
- **OOM / NaN Grad Skips**: `0` OOM skips, `0` NaN grad skips
- **Selected Checkpoint**: final `adapter/` (step `36` / 100%, `dev_schema_valid = 0.8333`, `dev_precision = 0.6762`, `dev_recall = 0.5669`, `dev_f1 = 0.6167`, `dev_token_f1 = 0.6335`, `empty_fp_rate = 0.0000`, `non_empty_pred_rate = 1.0000`)
- **GGUF LoRA Export & Held-Out / Smoke Eval**:
  - Exported `adapter/` to `<models-root>/gguf/p2000-roles/verifier-v2-lora.gguf` (`51.70 MiB` / `54,214,624` bytes; within P2000 budget) and linked `verifier-lora.gguf -> verifier-v2-lora.gguf`.
  - **Held-Out Eval**: `eval_n = 0` (no separate held-out Verifier / Log-Analyser evaluation split exists locally in staged bundles; `si-lab-log-analyser-candidate-v2-new24` and `si-lab-four-role-reviewed-increment-52-20261115` contain train + author-visible dev only).
  - **CPU GGUF Smoke Check (`dev_6` prompts, Q4_K_M + `verifier-v2-lora.gguf`, `n_predict=1200`, greedy, explicit BOS parity)**:
    - `json_valid = 1.0000` (6/6 EOS, valid JSON objects)
    - `schema_valid = 0.6667` (4/6 strict schema valid across `timeline`, `findings`, `unknowns`, `next_checks`)
    - `smoke_dev6_token_f1_micro = 0.5957` (`precision = 0.6049`, `recall = 0.5868`, `token_f1_macro = 0.5904`, `empty_fp_rate = 0.0000`, `total_elapsed_s = 85.22`, `avg_latency_s = 14.20`)

## Milestone Progression & Final Dev Eval

| Checkpoint | Step | Train Loss | Schema Valid | Precision | Recall | Dev F1 | Token F1 | Empty FP | Non-Empty Rate |
|---|---|---|---|---|---|---|---|---|---|
| 20% (`max_new=256`) | 8 | 1.1311 | 0.0000 | 0.5570 | 0.4391 | 0.4911 | 0.5145 | 0.0000 | 1.0000 |
| 40% (`max_new=256`) | 15 | 0.9704 | 0.3333 | 0.6603 | 0.4810 | 0.5566 | 0.5613 | 0.0000 | 1.0000 |
| 60% (`max_new=256`) | 22 | 0.8817 | 0.5000 | 0.6736 | 0.5190 | 0.5862 | 0.6034 | 0.0000 | 1.0000 |
| 80% (`max_new=256`) | 29 | 0.6621 | 0.3333 | 0.6550 | 0.4850 | 0.5573 | 0.5836 | 0.0000 | 1.0000 |
| **Final `adapter/` (`max_new=1800`)** | **36** | **-** | **0.8333** | **0.6762** | **0.5669** | **0.6167** | **0.6335** | **0.0000** | **1.0000** |

## CPU GGUF Smoke Check Breakdown (`dev_6`)

| Case ID | Prompt Tokens | Predicted Tokens | Stop | JSON Valid | Schema Valid | Token F1 | Latency (s) |
|---|---|---|---|---|---|---|---|
| `silog-v1-011` | 556 | 364 | `eos` | true | true | 0.6275 | 15.29 |
| `silog-v1-012` | 717 | 373 | `eos` | true | false | 0.5858 | 16.78 |
| `silog-v2-021` | 365 | 298 | `eos` | true | true | 0.5253 | 11.99 |
| `silog-v2-022` | 500 | 359 | `eos` | true | true | 0.6125 | 14.89 |
| `silog-v2-023` | 437 | 354 | `eos` | true | false | 0.6133 | 14.44 |
| `silog-v2-024` | 305 | 297 | `eos` | true | true | 0.5778 | 11.84 |

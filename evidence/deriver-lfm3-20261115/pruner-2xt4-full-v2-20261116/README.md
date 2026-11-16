# Pruner LoRA (LFM3-1.7B-Base) - 2xT4 DDP Full Run v2 Evidence Packet, 2026-11-16

This directory contains sanitized training receipts, GPU telemetry, milestone logs, dev metrics, and held-out GGUF evaluation results for Pruner Full Run v2 (`<kaggle-user>/p2000-pruner-v2-full`), executed on 2x Tesla T4 GPUs (`cuda:0` and `cuda:1`, `world_size=2`) via PyTorch `torchrun` DDP.

## Executive Summary

- **Kernel Reference**: `<kaggle-user>/p2000-pruner-v2-full`
- **Kernel Status**: `KernelWorkerStatus.COMPLETE`
- **DDP Verification**: `true` (`PER_RANK_STEPS_AND_ALLOCATIONS_OBSERVED`, `steps: [54, 54]`, `rows_processed: [216, 216]`, `known_distinct_uuids: true`)
- **Dataset**: combined reviewed Pruner `train.sft.jsonl` (`72` rows: `si-lab-pruner-proposed-candidate-v1` 16 + `honcho-pruner-candidate-v2-reviewed-48` 40 + `si-lab-four-role-reviewed-increment-52-20261115/pruner` 16; `17,501` tokens/epoch, `824` supervised tokens/epoch) & `dev.sft.jsonl` (`12` rows: `v1` 4 + `v2` 8), `max_length=5600`
- **Hyperparameters**: `role=pruner`, `lora_r=16`, `lora_alpha=32`, `target_modules=attn-mlp`, `lr=2e-4`, `epochs=6`, `effective_batch=8`, `microbatch=1`
- **Optimizer Steps**: `54 / 54` planned steps completed (`remaining_optimizer_steps = 0`, `stop_requested = false`)
- **P95 Step Latency**: `1.75 s/step` (`1.7515 s`)
- **Elapsed Time**: `141.81 s` training (`45.76 s` across 4 milestone probes; `208.4 s` `torchrun` wall; `161.22 s` script runtime)
- **Peak VRAM Allocated**: GPU 0: `3,885.43 MiB`, GPU 1: `3,885.43 MiB` (`4,368 MiB` PyTorch reserved on both GPUs)
- **Peak VRAM (nvidia-smi)**: GPU 0: `4,607 MiB`, GPU 1: `4,607 MiB` (`100%` peak util on both GPUs across `104` samples each)
- **OOM / NaN Grad Skips**: `0` OOM skips, `0` NaN grad skips
- **Selected Checkpoint**: `checkpoint-pct-60` (step `33` / 60%, `dev_schema_valid = 1.0000`, `dev_precision = 0.7624`, `dev_recall = 0.7064`, `dev_f1 = 0.7333`, `dev_token_f1 = 0.6146`, `empty_fp_rate = 0.5000`, `non_empty_pred_rate = 0.9167`; higher dev F1 than final `adapter/` step `54` at `0.6233`)
- **GGUF LoRA Export & Held-Out Eval (`si-lab-pruner-evaluation-only-candidate-v1`, 6 cases, CPU Q4_K_M + `pruner-v2-lora.gguf` [51.70 MiB], `n_predict=256`, greedy, explicit BOS parity)**:
  - `schema_valid = 1.0000` (6/6 EOS, valid `KEEP ...` format)
  - `pruner_line_f1_micro = 0.7407` (`precision = 0.6818`, `recall = 0.8108`, `macro_token_f1 = 0.5449`, `non_empty_pred_rate = 1.0000`, `empty_fp_rate = 1.0000`)
  - `total_elapsed_s = 17.73` (`avg_latency_s = 2.96`)

## Milestone Progression & Final Dev Eval

| Checkpoint | Step | Train Loss | Schema Valid | Precision | Recall | Dev F1 | Token F1 | Empty FP | Non-Empty Rate |
|---|---|---|---|---|---|---|---|---|---|
| 20% (`max_new=256`) | 11 | 0.5866 | 1.0000 | 0.6465 | 0.5872 | 0.6154 | 0.3775 | 1.0000 | 1.0000 |
| 40% (`max_new=256`) | 22 | 0.4405 | 1.0000 | 0.6607 | 0.6789 | 0.6697 | 0.5051 | 1.0000 | 1.0000 |
| **60% (`max_new=256`) [Selected]** | **33** | **0.3776** | **1.0000** | **0.7624** | **0.7064** | **0.7333** | **0.6146** | **0.5000** | **0.9167** |
| 80% (`max_new=256`) | 44 | 0.1366 | 1.0000 | 0.6970 | 0.6330 | 0.6635 | 0.5608 | 0.5000 | 0.9167 |
| Final `adapter/` (`max_new=1800`) | 54 | - | 1.0000 | 0.6321 | 0.6147 | 0.6233 | 0.5901 | 0.5000 | 0.9167 |

## Held-Out GGUF Eval Breakdown (`si-lab-pruner-evaluation-only-candidate-v1`)

| Case ID | Kind | Prompt Tokens | Predicted Tokens | Stop | Schema Valid | Pred | Gold | TP / Pred / Gold | Token F1 | Latency (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| `pruner-eval-01` | `partial` | 379 | 16 | `eos` | true | `KEEP 1-2,7-10,13-16` | `KEEP 1-2,7-10` | 6 / 10 / 6 | 0.8333 | 3.30 |
| `pruner-eval-02` | `partial` | 278 | 12 | `eos` | true | `KEEP 1-3,7-13` | `KEEP 2,7-10,12,15` | 6 / 10 / 7 | 0.3636 | 3.08 |
| `pruner-eval-03` | `partial` | 417 | 12 | `eos` | true | `KEEP 1-2,9-13` | `KEEP 1-4,8-11,13` | 6 / 7 / 9 | 0.5455 | 3.65 |
| `pruner-eval-04` | `none` | 307 | 8 | `eos` | true | `KEEP 1,5` | `KEEP none` | 0 / 2 / 0 | 0.4000 | 2.96 |
| `pruner-eval-05` | `all` | 131 | 8 | `eos` | true | `KEEP 1-7` | `KEEP all` | 7 / 7 / 7 | 0.4000 | 1.64 |
| `pruner-eval-06` | `partial` | 265 | 12 | `eos` | true | `KEEP 5-7,9-13` | `KEEP 2-3,5-9,13` | 5 / 8 / 8 | 0.7273 | 3.10 |

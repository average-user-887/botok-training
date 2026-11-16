# Scribe LoRA (LFM3-1.7B-Base) - 2xT4 DDP Full Run v2 Evidence Packet, 2026-11-16

This directory contains sanitized training receipts, GPU telemetry, milestone logs, dev metrics, and held-out GGUF evaluation results for Scribe Full Run v2 (`<kaggle-user>/p2000-scribe-v2-full`), executed on 2x Tesla T4 GPUs (`cuda:0` and `cuda:1`, `world_size=2`) via PyTorch `torchrun` DDP.

## Executive Summary

- **Kernel Reference**: `<kaggle-user>/p2000-scribe-v2-full`
- **Kernel Status**: `KernelWorkerStatus.COMPLETE`
- **DDP Verification**: `true` (`PER_RANK_STEPS_AND_ALLOCATIONS_OBSERVED`, `steps: [30, 30]`, `rows_processed: [105, 105]`, `known_distinct_uuids: true`)
- **Dataset**: combined reviewed Scribe `train.sft.jsonl` (`42` rows: `v1` 10 + `v2` 20 + `four-role-increment-52` 12; `39,611` tokens/epoch, `10,448` supervised tokens/epoch) & `dev.sft.jsonl` (`6` rows: `v1` 2 + `v2` 4), `max_length=5600`
- **Hyperparameters**: `role=scribe`, `lora_r=16`, `lora_alpha=32`, `target_modules=attn-mlp`, `lr=2e-4`, `epochs=5`, `effective_batch=8`, `microbatch=1`
- **Optimizer Steps**: `30 / 30` planned steps completed (`remaining_optimizer_steps = 0`, `stop_requested = false`)
- **P95 Step Latency**: `4.71 s/step` (`4.707 s`)
- **Elapsed Time**: `467.60 s` training (`347.53 s` across 4 milestone probes; `570.78 s` total script runtime)
- **Peak VRAM Allocated**: GPU 0: `4,437.52 MiB`, GPU 1: `4,437.54 MiB` (`4,542 MiB` PyTorch reserved on both GPUs)
- **Peak VRAM (nvidia-smi)**: GPU 0: `4,783 MiB`, GPU 1: `4,781 MiB` (`100%` peak util on both GPUs across `304` samples)
- **OOM / NaN Grad Skips**: `0` OOM skips, `0` NaN grad skips
- **Selected Checkpoint**: final `adapter/` (step `30` / 100%, `dev_schema_valid = 1.0000`, `dev_f1 = 0.7053`, `dev_token_f1 = 0.7392`, `empty_fp_rate = 0.0000`, `non_empty_pred_rate = 1.0000`)
- **GGUF LoRA Export & Held-Out Eval (`si-lab-scribe-eval-candidate-v1-reviewed`, 6 cases, CPU Q4_K_M + `scribe-v2-lora.gguf` [51.70 MiB], `n_predict=1200`, greedy)**:
  - `schema_valid = 1.0000` (6/6 EOS, valid JSON matching `scribe` schema)
  - `source_binding_valid = 1.0000` (6/6 exact `as_of` match and all `evidence` message IDs present in input)
  - `non_empty_pred_rate = 0.8333` (5 substantive cases non-empty, 1 empty greeting-only abstention case predicted `items: []`)
  - `empty_fp_rate = 0.0000` (`scribe-eval-06` `token_f1 = 0.8462`)
  - `scribe_token_f1_micro = 0.6456` (`precision = 0.8030`, `recall = 0.5397`, `macro_token_f1 = 0.6713`)
  - `total_elapsed_s = 60.24` (`avg_latency_s = 10.04`)

## Milestone Progression & Final Dev Eval

| Checkpoint | Step | Train Loss | Schema Valid | Precision | Recall | Dev F1 | Token F1 | Empty FP | Non-Empty Rate |
|---|---|---|---|---|---|---|---|---|---|
| 20% (`max_new=256`) | 6 | 0.7355 | 0.0000 | 0.5174 | 0.5897 | 0.5512 | 0.5290 | 0.0000 | 1.0000 |
| 40% (`max_new=256`) | 12 | 0.6283 | 0.3333 | 0.7374 | 0.6807 | 0.7079 | 0.7417 | 0.0000 | 1.0000 |
| 60% (`max_new=256`) | 18 | 0.3846 | 0.3333 | 0.7576 | 0.6410 | 0.6944 | 0.7314 | 0.0000 | 1.0000 |
| 80% (`max_new=256`) | 24 | 0.3969 | 0.8333 | 0.7741 | 0.6550 | 0.7096 | 0.7439 | 0.0000 | 1.0000 |
| **Final `adapter/` (`max_new=1800`)** | **30** | **-** | **1.0000** | **0.7487** | **0.6667** | **0.7053** | **0.7392** | **0.0000** | **1.0000** |

## Held-Out GGUF Eval Breakdown (`si-lab-scribe-eval-candidate-v1-reviewed`)

| Case ID | Prompt Tokens | Predicted Tokens | Stop | Schema Valid | Source Valid | Pred Items / Gold | Token F1 | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| `scribe-eval-01` | 604 | 251 | `eos` | true | true | 4 / 5 | 0.7219 | 12.07 |
| `scribe-eval-02` | 659 | 277 | `eos` | true | true | 5 / 4 | 0.6628 | 13.19 |
| `scribe-eval-03` | 648 | 213 | `eos` | true | true | 4 / 4 | 0.6289 | 11.13 |
| `scribe-eval-04` | 704 | 212 | `eos` | true | true | 4 / 4 | 0.6056 | 11.30 |
| `scribe-eval-05` | 728 | 164 | `eos` | true | true | 3 / 4 | 0.5621 | 9.61 |
| `scribe-eval-06` | 352 | 30 | `eos` | true | true | 0 / 0 | 0.8462 | 2.94 |

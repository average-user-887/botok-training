# Deriver LoRA (LFM3-1.7B-Base) - 2xT4 DDP Full Run Part 3 (v33) Evidence Packet, 2026-11-16

This directory contains sanitized training receipts, GPU telemetry, decile milestone logs, dev metrics, and held-out GGUF evaluation results for Full Run Part 3 (`<kaggle-user>/p2000-deriver-v25-full-part3`), executed on 2x Tesla T4 GPUs (`cuda:0` and `cuda:1`, `world_size=2`) via PyTorch `torchrun` DDP.

## Executive Summary

- **Kernel Reference**: `<kaggle-user>/p2000-deriver-v25-full-part3`
- **Kernel Status**: `KernelWorkerStatus.COMPLETE`
- **DDP Verification**: `true` (`PER_RANK_STEPS_AND_ALLOCATIONS_OBSERVED`, `steps: [219, 219]`, `rows_processed: [870, 870]`, `known_distinct_uuids: true`)
- **Dataset**: `honcho-deriver-v33-training-only` (`580` train rows, `1,764,395` tokens/epoch, `695,516` supervised tokens/epoch, `max_length=5600`) & `24` dev rows
- **Init & Hyperparameters**: initialized from Part 2 final `adapter/`, `role=deriver`, `lora_r=16`, `lora_alpha=32`, `target_modules=attn-mlp`, `lr=5e-5`, `epochs=3`, `effective_batch=8`, `microbatch=1`
- **Optimizer Steps**: `219 / 219` planned steps completed (`remaining_optimizer_steps = 0`, `stop_requested = false`)
- **P95 Step Latency**: `24.39 s/step` (`24.388 s`)
- **Elapsed Time**: `4,591.49 s` training (`476.01 s` across 4 milestone probes; `5,640.13 s` total script runtime)
- **Peak VRAM Allocated**: GPU 0: `12,146.02 MiB`, GPU 1: `12,146.05 MiB` (`12,504.0 MiB` / `12,484.0 MiB` PyTorch reserved)
- **Peak VRAM (nvidia-smi)**: GPU 0: `12,847 MiB` (`84.8%` mean util), GPU 1: `12,723 MiB` (`80.1%` mean util), `100%` peak util on both GPUs across `2,834` samples each
- **OOM / NaN Grad Skips**: `0` OOM skips, `0` NaN grad skips
- **Selected Checkpoint**: final `adapter/` (step `219` / 100%, `dev_f1 = 0.2117`, `dev_schema_valid = 0.7500`, `non_empty_pred_rate = 0.4167`, `empty_fp_rate = 0.1667`)
- **GGUF LoRA Export**: `<models-root>/gguf/p2000-roles/deriver-v33-lora.gguf` (`51.70 MiB` / `54,214,624` bytes)
- **Held-Out Eval (`honcho-deriver-v14-evaluation-only`, 4 cases, CPU Q4_K_M + LoRA, greedy, explicit BOS parity)**:
  - **Baseline cap (`n_predict=1200`)**: `f1 = 0.1754` (`precision = 0.3824`, `recall = 0.1138`), `schema_valid = 0.5000` (2/4 EOS: `evfourteen-medium-2` at 839 tokens, `evfourteen-empty-4` at 6 tokens; 2 medium rows hit `n_predict=1200` limit with `token_f1` `0.7692` / `0.7992`), `token_f1 = 0.8302`, `empty_fp_rate = 0.0000`, `total_elapsed_s = 159.27` (`avg_latency_s = 39.82`)
  - **`-n 768` cap (`n_predict=768`)**: `f1 = 0.0000`, `schema_valid = 0.2500` (`evfourteen-empty-4` EOS at 6 tokens; all 3 positive 41-item cases truncated at `stop_type=limit`), `token_f1 = 0.7851`, `empty_fp_rate = 0.0000`, `total_elapsed_s = 124.23` (`avg_latency_s = 31.06`)
  - **Active Symlink Selection**: `<models-root>/gguf/p2000-roles/deriver-lora.gguf -> <models-root>/gguf/p2000-roles/deriver-v20-lora.gguf` (highest held-out F1 `0.2205` at `n_predict=1200` with `schema_valid = 0.5000`)

## Decile Progression & Final Dev Eval

Mid-training milestone probes (`20%, 40%, 60%, 80%`) used `milestone_probe_rows=12` and `milestone_max_new_tokens=256`, which truncated positive JSON outputs before EOS (`dev_f1=0.0`, only the 5 empty-gold rows completed within 256 tokens -> `dev_schema_valid=0.4167`). Final evaluation of `adapter/` (step `219`, 24 dev rows, `eval_max_new_tokens=1800`) produced structured non-empty predictions:

| Checkpoint | Step | Train Loss | Schema Valid | Precision | Recall | Dev F1 | Token F1 | Empty FP | Non-Empty Rate |
|---|---|---|---|---|---|---|---|---|---|
| 20% (`max_new=256`) | 44 | 0.2096 | 0.4167 | 0.0000 | 0.0000 | 0.0000 | 0.5448 | 0.0000 | 0.0000 |
| 40% (`max_new=256`) | 88 | 0.1085 | 0.4167 | 0.0000 | 0.0000 | 0.0000 | 0.5455 | 0.0000 | 0.0000 |
| 60% (`max_new=256`) | 132 | 0.2242 | 0.4167 | 0.0000 | 0.0000 | 0.0000 | 0.5492 | 0.0000 | 0.0000 |
| 80% (`max_new=256`) | 176 | 0.1444 | 0.4167 | 0.0000 | 0.0000 | 0.0000 | 0.5372 | 0.0000 | 0.0000 |
| **Final `adapter/` (`max_new=1800`)** | **219** | **-** | **0.7500** | **0.2647** | **0.1763** | **0.2117** | **0.5909** | **0.1667** | **0.4167** |

## Part 1 vs Part 2 vs Part 3 Held-Out Comparison (`honcho-deriver-v14-evaluation-only`)

| Adapter | Init / Dataset | Cap (`n_predict`) | Schema Valid | Precision | Recall | Held-Out F1 | Token F1 | Empty FP |
|---|---|---|---|---|---|---|---|---|
| `deriver-v18-lora.gguf` (Part 1 `pct-80`) | Base / v18 (352 rows) | 1200 | 0.5000 | 0.4483 | 0.1138 | 0.1815 | 0.8555 | 0.0000 |
| **`deriver-v20-lora.gguf` (Part 2 `adapter/`)** | **`pct-80` / v20 (392 rows)** | **1200** | **0.5000** | **0.4474** | **0.1463** | **0.2205** | **0.8535** | **0.0000** |
| `deriver-v20-lora.gguf` (Part 2 `adapter/`) | `pct-80` / v20 (392 rows) | 768 | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.8423 | 0.0000 |
| `deriver-v33-lora.gguf` (Part 3 `adapter/`) | `part2` / v33 (580 rows) | 1200 | 0.5000 | 0.3824 | 0.1138 | 0.1754 | 0.8302 | 0.0000 |
| `deriver-v33-lora.gguf` (Part 3 `adapter/`) | `part2` / v33 (580 rows) | 768 | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.7851 | 0.0000 |

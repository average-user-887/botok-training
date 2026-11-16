# Pruner LoRA (`LFM3-1.7B-Base`) — 2xT4 DDP Full Run v4 Evidence Packet, 2026-11-16

This directory contains sanitized training receipts, GPU telemetry, admission audit, warm-start lineage SHA-256s, decile milestone logs (`10%..90%`), renderer parity audit resolution note, and live Quadro P2000 hardware benchmark results (`dev` n=12 and held-out `pruner10` n=10) for Pruner Full Run v4 (`<kaggle-user>/p2000-pruner-v4-full`), executed on 2x Tesla T4 GPUs (`cuda:0` and `cuda:1`, `world_size=2`) via PyTorch `torchrun` DDP.

## Executive Summary

- **Kernel Reference**: `<kaggle-user>/p2000-pruner-v4-full` (`KernelWorkerStatus.COMPLETE`)
- **DDP Verification**: `true` (`PER_RANK_STEPS_AND_ALLOCATIONS_OBSERVED`, `steps: [76, 76]`, `rows_processed: [292, 292]`, `known_distinct_uuids: true`, `0` OOM/NaN skips)
- **Dataset & Admission Audit (`--over-limit-policy=reject`)**:
  - `144` base+addenda rows (`si-lab-pruner-proposed-candidate-v1` 16, `honcho-pruner-candidate-v2-reviewed-48` 40, `si-lab-four-role-reviewed-increment-52-20261115/pruner` 16, `botok-feedback-training-addenda-20261116/pruner` 8, `botok-pruner-feedback-batch2-20261116` 8, `botok-feedback-training-addenda-batch5-20261116/pruner` 8, `botok-pruner-reviewed-batch4..9-20261116` 48) oversampled to `146` rows (`empty_target_ratio=0.25`) + `12` dev rows.
  - `146 / 146` rows admitted cleanly (`rejected_rows=0`, `truncated_rows=0`, `excess_tokens_total=0`, `64,887` total tokens/epoch, `1,535` supervised tokens/epoch, length `105..2559 <= 5600`, `data_sha256=55334428586d66fe1f77a0b7e9cc6e019d460dcec24a2e7157259edd8b25b23f`).
- **Warm-Start Lineage**:
  - Initialized via `--init-adapter` (`warm_start_branch`) from `pruner_v3/pruner_v3/checkpoint-pct-60` (`adapter_model.safetensors` SHA-256 `298b5fdd2c028d9a29736e9070159ac5859c227610dca16ecca774ad0a4b04b6`) with fresh optimizer, scheduler, and RNG (`optimizer_restored=false`, `scheduler_restored=false`, `rng_restored=false`).
- **Hyperparameters**: `role=pruner`, `format=honcho_pruner_v1`, `lora_r=16`, `lora_alpha=32`, `target_modules=attn-mlp`, `lr=1e-4`, `epochs=4`, `effective_batch=8`, `microbatch=1`, `max_length=5600`, `skip_kaggle_eval=true`.
- **Steps & Timing**: `76 / 76` optimizer steps (`remaining_optimizer_steps=0`), `p95_step_seconds = 4.0354 s`, `elapsed_training_seconds = 205.23 s`.
- **Peak VRAM (Kaggle 2xT4)**: Rank 0 `5,535.08 MiB` allocated (`5,766.0 MiB` reserved, `nvidia-smi` max `6,107 MiB`, mean/max util `58.70% / 100%` across `132` samples); Rank 1 `5,582.91 MiB` allocated (`5,768.0 MiB` reserved, `nvidia-smi` max `6,007 MiB`, mean/max util `63.04% / 100%` across `132` samples).
- **Selected Winner & Symlink Promotion**:
  - **`v4_pct30`** (`pruner-v4-pct30-lora.gguf`, `54,214,720` bytes, GGUF SHA-256 `2abf951699911592bd784eb373ec68797bc25ac8b6f574c7215e287fcceaffa0`, safetensors SHA-256 `6f3bb2fef231be205a3a0d60c3d264f763d17140fcf74730213a00e28659bc87`) achieved **`selection_f1_micro = 0.7606`** (`P=0.7105`, `R=0.8182`, `TP/FP/FN=27/11/6`, `exact_set_acc=0.4000`, `empty_fp_rate=0.0`, `format_valid=1.0000`, `all_eos=true`), outperforming `pruner_v3` `pct-60` (`0.7143` historical CPU / `0.7576` matched P2000 GPU).
  - Promoted `<models-root>/gguf/p2000-roles/pruner-lora.gguf -> <models-root>/gguf/p2000-roles/pruner-v4-pct30-lora.gguf`.

## Renderer Parity Audit Resolution (`AUDIT-pruner-renderer-parity-unresolved-20261116`)

- **Audit Finding**: At GH `main` `6d7240f` / `evidence` `b9978d8`, `AUDIT-pruner-renderer-parity-unresolved-20261116` returned `DEPLOYED_PRUNER_RENDERER_PARITY_UNRESOLVED` because `botok-training` contained only deriver scripts (`deployed_pruner_source_found=false`). Across 60 training rows through `pruner8`, the audit replayed a `12`-token/row difference between inherited deriver chat templating (`39,145` tokens) and raw prompt+completion concatenation (`38,425` tokens; `+1` trailing token/row for full assistant chat template).
- **Deployed Renderer Verified (`../code/train_deriver.py:encode_row`)**:
  - Deployed Kaggle runs (`pruner_v3`, `pruner_v4`, `format="honcho_pruner_v1"`) take `{prompt, completion}` rows (no `messages` key) and use raw concatenation with explicit BOS/EOS (`prefix = [124894] + tok(prompt, add_special_tokens=False)`, `completion = tok(completion, add_special_tokens=False) + [124900]`, retaining leading ASCII space in `" KEEP ..."`, **no chat template**, `-100` loss mask on `prefix`).
  - Matches GGUF inference (`llama-server --override-kv tokenizer.ggml.add_bos_token=bool:true`, `bos=124894`, `temperature=0.0`).
  - Sanitized `train_deriver.py` is committed in `../code/train_deriver.py`.

## Kaggle Decile Training Milestones (`--skip-kaggle-eval`)

| Pct | Step | Train Loss | Step (s) |
|---|---|---|---|
| 10 | 8 | 0.5611 | 1.9819 |
| 20 | 16 | 0.4196 | 1.9720 |
| 30 | 23 | 0.2045 | 1.9930 |
| 40 | 31 | 0.2372 | 6.2745 |
| 50 | 38 | 0.1231 | 0.3834 |
| 60 | 46 | 0.0086 | 3.8923 |
| 70 | 54 | 0.0453 | 3.9926 |
| 80 | 61 | 0.0441 | 1.6779 |
| 90 | 69 | 0.0076 | 1.8224 |

## Live Quadro P2000 Benchmark (`lfm3-1.7b-base-Q4_K_M.gguf`, `bos=124894`, `temperature=0.0`, `n_predict=512`, `ctx=8192`)

### Held-Out `EVAL-ONLY-si-lab-pruner10-log8-reviewed-20261116` (`n=10`)

| Checkpoint | EOS | Format Valid | TP / FP / FN | Precision | Recall | Micro F1 | Exact Set | Empty FP | Peak Card / PID VRAM (MiB) | Gen tok/s | P50 / P95 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `v3_pct60` (baseline) | 10/10 | 1.0000 | 25 / 8 / 8 | 0.7576 | 0.7576 | 0.7576 | 0.4000 (4/10) | 0.0 (0/2) | 1,283 / 1,278 | 46.70 | 235.27 / 300.64 |
| `v4_pct10` | 10/10 | 1.0000 | 27 / 18 / 6 | 0.6000 | 0.8182 | 0.6923 | 0.1000 (1/10) | 1.0 (2/2) | 1,283 / 1,278 | 46.89 | 297.46 / 380.42 |
| `v4_pct20` | 10/10 | 1.0000 | 28 / 13 / 5 | 0.6829 | 0.8485 | 0.7568 | 0.3000 (3/10) | 0.0 (0/2) | 1,283 / 1,278 | 46.44 | 235.77 / 386.51 |
| **`v4_pct30` (Promoted)** | **10/10** | **1.0000** | **27 / 11 / 6** | **0.7105** | **0.8182** | **0.7606** | **0.4000 (4/10)** | **0.0 (0/2)** | **1,283 / 1,278** | **46.56** | **283.12 / 390.03** |
| `v4_pct40` | 10/10 | 1.0000 | 19 / 6 / 14 | 0.7600 | 0.5758 | 0.6552 | 0.4000 (4/10) | 0.0 (0/2) | 1,283 / 1,278 | 46.30 | 234.78 / 300.37 |
| `v4_pct50` | 10/10 | 1.0000 | 23 / 8 / 10 | 0.7419 | 0.6970 | 0.7188 | 0.4000 (4/10) | 0.0 (0/2) | 1,283 / 1,278 | 46.46 | 236.07 / 344.18 |
| `v4_pct60` | 10/10 | 1.0000 | 26 / 14 / 7 | 0.6500 | 0.7879 | 0.7123 | 0.4000 (4/10) | 0.0 (0/2) | 1,283 / 1,278 | 46.12 | 239.32 / 387.16 |
| `v4_pct70` | 10/10 | 1.0000 | 21 / 10 / 12 | 0.6774 | 0.6364 | 0.6562 | 0.4000 (4/10) | 0.0 (0/2) | 1,283 / 1,278 | 45.80 | 238.57 / 389.76 |
| `v4_pct80` | 10/10 | 1.0000 | 26 / 13 / 7 | 0.6667 | 0.7879 | 0.7222 | 0.4000 (4/10) | 0.0 (0/2) | 1,283 / 1,278 | 46.01 | 239.01 / 386.57 |
| `v4_pct90` | 10/10 | 1.0000 | 21 / 12 / 12 | 0.6364 | 0.6364 | 0.6364 | 0.4000 (4/10) | 0.0 (0/2) | 1,283 / 1,278 | 46.01 | 240.64 / 390.72 |
| `v4_adapter` (100%) | 10/10 | 0.9000 | 20 / 10 / 13 | 0.6667 | 0.6061 | 0.6349 | 0.4000 (4/10) | 0.0 (0/2) | 1,283 / 1,278 | 46.10 | 238.54 / 386.73 |

### Dev Split (`n=12`) on Live Quadro P2000

| Checkpoint | EOS | Format Valid | TP / FP / FN | Precision | Recall | Micro F1 | Exact Set | Empty FP |
|---|---|---|---|---|---|---|---|---|
| `v3_pct60` | 12/12 | 1.0000 | 66 / 14 / 43 | 0.8250 | 0.6055 | 0.6984 | 0.3333 (4/12) | 0.0 |
| `v4_pct10` | 12/12 | 1.0000 | 64 / 15 / 45 | 0.8101 | 0.5872 | 0.6809 | 0.0833 (1/12) | 0.5 |
| `v4_pct20` | 12/12 | 1.0000 | 77 / 22 / 32 | 0.7778 | 0.7064 | 0.7404 | 0.3333 (4/12) | 0.0 |
| `v4_pct30` | 12/12 | 1.0000 | 68 / 24 / 41 | 0.7391 | 0.6239 | 0.6766 | 0.0833 (1/12) | 0.5 |
| `v4_pct40` | 12/12 | 1.0000 | 74 / 22 / 35 | 0.7708 | 0.6789 | 0.7220 | 0.3333 (4/12) | 0.0 |
| `v4_pct50` | 12/12 | 1.0000 | 61 / 21 / 48 | 0.7439 | 0.5596 | 0.6387 | 0.3333 (4/12) | 0.0 |
| `v4_pct60` | 12/12 | 1.0000 | 75 / 14 / 34 | 0.8427 | 0.6881 | 0.7576 | 0.3333 (4/12) | 0.0 |
| `v4_pct70` | 12/12 | 1.0000 | 76 / 11 / 33 | 0.8736 | 0.6972 | 0.7755 | 0.3333 (4/12) | 0.0 |
| `v4_pct80` | 12/12 | 1.0000 | 77 / 20 / 32 | 0.7938 | 0.7064 | 0.7476 | 0.3333 (4/12) | 0.0 |
| `v4_pct90` | 12/12 | 1.0000 | 74 / 15 / 35 | 0.8315 | 0.6789 | 0.7475 | 0.3333 (4/12) | 0.0 |
| `v4_adapter` (100%) | 12/12 | 1.0000 | 74 / 17 / 35 | 0.8132 | 0.6789 | 0.7400 | 0.3333 (4/12) | 0.0 |

### Service & Rollback Verification (`rollback_verified: true`)

- Temporary `llama-swap` GPU model unloads on `<p2000-host>` (`decision-d1-2b`, `qwen3-embedding-4b-p2000`; `0` GPU Docker containers) were restored and verified ready (`containers_verified: true`, `swap_verified: true`, `rollback_verified: true`).

Full raw hardware benchmark telemetry, per-row predictions, and rollback receipts are in [`p2000_eval_pruner_v4.json`](./p2000_eval_pruner_v4.json) and [`lineage_and_admission_receipt.json`](./lineage_and_admission_receipt.json).

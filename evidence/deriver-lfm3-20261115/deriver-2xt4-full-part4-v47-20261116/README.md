# Deriver LoRA (LFM3-1.7B-Base) - 2xT4 DDP Full Run Part 4 (v47) Evidence Packet, 2026-11-16

This directory contains sanitized training receipts, GPU telemetry, decile milestone logs, dev metrics, warm-start lineage, admission audit, and matched `n_predict=2048` held-out `v14` GGUF evaluation results for Full Run Part 4 (`<kaggle-user>/p2000-deriver-v40-full-part4`), executed on 2x Tesla T4 GPUs (`cuda:0` and `cuda:1`, `world_size=2`) via PyTorch `torchrun` DDP.

## Executive Summary

- **Kernel Reference**: `<kaggle-user>/p2000-deriver-v40-full-part4` (`KernelWorkerStatus.COMPLETE`)
- **DDP Verification**: `true` (`PER_RANK_STEPS_AND_ALLOCATIONS_OBSERVED`, `steps: [174, 174]`, `rows_processed: [692, 692]`, `known_distinct_uuids: true`)
- **Dataset & Admission Audit**: `honcho-deriver-v47-training-only` (`691` train rows, `2,144,712` full corpus tokens/pass -> `2,139,622` executed tokens/pass, `835,861` supervised tokens/pass; exact `5,090` excess prompt tokens middle-truncated across the `11` rows `>5600` prior to the `--over-limit-policy=reject` patch) & `24` dev rows (`414` gold items, SHA-256 `4454d0f194c41bea29b3098d5aee6d8b9305db841272e73eee7010bebf1ac22a`)
- **Warm-Start Lineage**: `--init-adapter` branch from `deriver_part1/full_part1/checkpoint-pct-80` (`adapter_model.safetensors` SHA-256 `d27bb70dce5e67976994a6e8f951145e2c600e5293a031c133ee73a53b430faa`) with fresh AdamW optimizer, cosine scheduler, and RNG (`optimizer_restored=false`, `scheduler_restored=false`, `rng_restored=false`). NOT sequential continuation from Part 3 and NOT a full-state resume.
- **Hyperparameters**: `role=deriver`, `lora_r=16`, `lora_alpha=32`, `target_modules=attn-mlp`, `lr=1e-4`, `epochs=2`, `effective_batch=8`, `microbatch=1`, `max_seq_len=5600`
- **Optimizer Steps**: `174 / 174` planned steps completed (`remaining_optimizer_steps = 0`, `stop_requested = false`)
- **P95 Step Latency**: `25.66 s/step` (`25.6585 s`)
- **Elapsed Time**: `3,986.00 s` training (`480.50 s` across 4 milestone probes; `5,154.99 s` total script runtime)
- **Peak VRAM (activity / nvidia-smi)**: `12,146.0 MiB` allocated (`12,524.0 MiB` reserved) on both ranks; `nvidia-smi` peak `12,867 MiB` (GPU 0, `82.1%` mean util) and `12,763 MiB` (GPU 1, `76.2%` mean util), `100%` peak util across `2,590` samples each
- **OOM / NaN Grad Skips**: `0` OOM skips, `0` NaN grad skips
- **Final `adapter/` Dev Eval (`dev_24`, `max_new=1800`)**: `dev_f1 = 0.2442` (`dev_precision = 0.3390` [`80/236`], `dev_recall = 0.1908` [`79/414`], `dev_schema_valid = 0.7083`, `empty_fp_rate = 0.0000`, `short_window_recall = 0.4877`, `long_window_recall = 0.0000`)
- **GGUF LoRA Exports**:
  - `deriver-v47-peak-lora.gguf` (`checkpoint-pct-20`, `54,214,720` bytes, SHA-256 `ee211e0724770d4cd8a524ccd111475674189f628be398e2cfdaf0a372298f21`)
  - `deriver-v47-lora.gguf` (`adapter/`, `54,214,624` bytes, SHA-256 `eb96f8ca9183d7167acbc60f9a9151b4d303d5800fd731bb151e063d282e5d88`)
- **Active Symlink Updated**: `<models-root>/gguf/p2000-roles/deriver-lora.gguf -> <models-root>/gguf/p2000-roles/deriver-v47-peak-lora.gguf` (wins matched `v14` eval at `F1_gtp = 0.4018`, `F1_tp = 0.3961`, `schema_valid = 0.7500`, `stop_eos_rate = 0.7500`)

## Decile Progression & Final Dev Eval

Part 4 launched prior to the `milestone_max_new_tokens` patch, so mid-training milestone probes (`20%, 40%, 60%, 80%`) used `milestone_probe_rows=12` and `milestone_max_new_tokens=256`, capping positive outputs before EOS (`dev_f1=0.0`, `dev_schema_valid=0.4167` from the 5 empty-gold rows). Among `checkpoint-pct-*`, `checkpoint-pct-20` had the highest `dev_token_f1` (`0.5537`). Final `adapter/` (`100%`, step `174`, `eval_max_new_tokens=1800`) achieved `dev_f1 = 0.2442` with `0.0` empty FP.

| Checkpoint | Step | Train Loss | Schema Valid | Precision | Recall | Dev F1 | Token F1 | Empty FP | Non-Empty Rate |
|---|---|---|---|---|---|---|---|---|---|
| 20% (`max_new=256`) | 35 | 0.2794 | 0.4167 | 0.0000 | 0.0000 | 0.0000 | 0.5537 | 0.0000 | 0.0000 |
| 40% (`max_new=256`) | 70 | 0.2753 | 0.4167 | 0.0000 | 0.0000 | 0.0000 | 0.5391 | 0.0000 | 0.0000 |
| 60% (`max_new=256`) | 105 | 0.1437 | 0.4167 | 0.0000 | 0.0000 | 0.0000 | 0.5431 | 0.0000 | 0.0000 |
| 80% (`max_new=256`) | 140 | 0.2380 | 0.4167 | 0.0000 | 0.0000 | 0.0000 | 0.5336 | 0.0000 | 0.0000 |
| **Final `adapter/` (`max_new=1800`)** | **174** | **-** | **0.7083** | **0.3390** | **0.1908** | **0.2442** | **0.6113** | **0.0000** | **0.3750** |

## Matched Held-Out Comparison (`honcho-deriver-v14-evaluation-only`, `n_predict=2048`, `temperature=0.0`, Explicit BOS `124894`)

| Model / Adapter | EOS Rate | Schema Valid | Capped (`limit`) | TP / Pred | GTP / Gold | Precision | Recall (`gtp` / `tp`) | Micro F1 (`gtp` / `tp`) | Token F1 | Empty FP | Elapsed (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `base` (`lfm3-1.7b-base-Q4_K_M`) | 0.2500 (1/4) | 0.0000 | 3/4 (evfourteen-medium-1, evfourteen-medium-2, evfourteen-medium-3) | 0 / 0 | 0 / 123 | 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.2729 | 0.0000 | 1198.55 |
| `v20` (`deriver-v20-lora.gguf`) | 0.2500 (1/4) | 0.2500 | 3/4 (evfourteen-medium-1, evfourteen-medium-2, evfourteen-medium-3) | 0 / 0 | 0 / 123 | 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.8483 | 0.0000 | 1578.59 |
| **`v47_peak` (`deriver-v47-peak-lora.gguf`, `pct-20`)** | 0.7500 (3/4) | 0.7500 | 1/4 (evfourteen-medium-1) | 41 / 84 | 42 / 123 | 0.4881 | 0.3415 / 0.3333 | 0.4018 / 0.3961 | 0.8356 | 0.0000 | 917.34 |
| `v47_final` (`deriver-v47-lora.gguf`, `adapter/`) | 0.7500 (3/4) | 0.7500 | 1/4 (evfourteen-medium-1) | 35 / 77 | 37 / 123 | 0.4545 | 0.3008 / 0.2846 | 0.3620 / 0.3500 | 0.8426 | 0.0000 | 470.40 |

Notes on capped rows and historical `n_predict=1200` comparison:
- Under matched `llama-server` CPU decoding (`n_predict=2048`, `temperature=0.0`, `repeat_penalty=1.0`, exact HF token IDs starting with `bos=124894`), `base` failed schema/EOS on all positive rows (`0/4` valid) and emitted non-JSON prose on `evfourteen-empty-4`; `v20` entered repetition loops before EOS on all 3 positive rows (`0.2500` schema valid on `empty-4`, `0.0000` F1; historical `n_predict=1200` `llama-cli` run had `0.2205` `gtp`-F1 / `0.2112` `tp`-F1 and `0.5000` schema valid).
- Both `v47_peak_pct20` and `v47_final` terminated cleanly with `eos` on `evfourteen-medium-2` (`893` / `842` tokens), `evfourteen-medium-3` (`1425` / `1454` tokens — previously clipped at `n_predict=1200`), and `evfourteen-empty-4` (`6` tokens, `0` false positives). Only `evfourteen-medium-1` hit `n_predict=2048` (`stop_type=limit`, repetition loop near the tail).

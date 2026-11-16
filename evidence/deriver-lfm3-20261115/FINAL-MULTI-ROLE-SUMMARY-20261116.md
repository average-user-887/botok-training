# Multi-Role LFM3-1.7B-Base P2000 Bundle & Held-Out Verification Summary (2026-11-16)

This document summarizes the completed 5-role 2xT4 DDP Kaggle training runs, the Deriver Part 3 (`v33`) vs Part 2 (`v20`) evaluation comparison, the held-out `EVAL-ONLY-si-lab-judge12-scribe8-reviewed-20261116` benchmark for `judge-lora.gguf` and `scribe-lora.gguf`, and the 5 GiB (`5120 MiB`) NVIDIA Quadro P2000 VRAM bundle verification.

## 1. Deriver Part 3 (`v33`) & `-n 768` Diagnosis

- **Part 3 Run**: `<kaggle-user>/p2000-deriver-v25-full-part3` (`219 / 219` steps, `p95 = 24.39 s/step`, `0` OOM/NaN skips, final `adapter/` `dev_f1 = 0.2117`, `deriver-v33-lora.gguf` `51.70 MiB`).
- **Held-Out `honcho-deriver-v14-evaluation-only` (4 cases)**:
  - `deriver-v20-lora.gguf` (`n_predict=1200`): `f1 = 0.2205`, `schema_valid = 0.5000` (active `deriver-lora.gguf` symlink).
  - `deriver-v33-lora.gguf` (`n_predict=1200`): `f1 = 0.1754`, `schema_valid = 0.5000`.
  - **`-n 768` (`n_predict=768`) on both `v20` and `v33`**: `f1 = 0.0000`, `schema_valid = 0.2500`. **Cause**: all 3 non-empty positive cases (`evfourteen-medium-1/2/3`, 41 gold items each; e.g. `medium-2` needed 839 tokens under `n_predict=1200`) hit the 768-token cap mid-JSON (`stop_type=limit`, `schema_valid=false`), leaving only `evfourteen-empty-4` (`6` tokens, `gold=0`) valid.
- **Final GDrive Metadata Sync (new archives since `part3-prep`)**:
  - `EVAL-ONLY-si-lab-pruner10-log8-reviewed-20261116.zip`
  - `honcho-deriver-v34-training-only.zip` through `honcho-deriver-v40-training-only.zip` (`7` incremental training archives; up to `635` train windows / `1,962,572` tokens/pass in `v40` `STATISTICS.json`).

## 2. Held-Out Adversarial Eval (`EVAL-ONLY-si-lab-judge12-scribe8-reviewed-20261116`)

Evaluated locally on CPU (`lfm3-1.7b-base-Q4_K_M.gguf`, greedy `temperature=0.0`):

### Judge (`judge-lora.gguf` -> `judge-v1-lora.gguf`, 12 cases, `n_predict=512`)

| Mode | Schema Valid (4-key) | Strict Checker (`check_judge_predictions.py`) | Verdict Acc | False Pass | False Fail | Exact Fields | Token F1 (micro) | Elapsed (s) |
|---|---|---|---|---|---|---|---|---|
| **Explicit BOS parity** (`[bos] + prompt`, matching `train_deriver.py` / `scribe_v2`) | **1.0000** (12/12 EOS) | **0.9167** (11/12) | **0.7500** (9/12) | 0 | 3 | 0.1667 (2/12) | **0.9667** | 41.78 |
| No-BOS (`raw_str`, matching `judge_v1` smoke call) | 0.3333 (4/12 EOS) | 0.3333 (4/12) | 0.1667 (2/12) | 2 | 0 | 0.1667 (2/12) | 0.9180 | 174.33 |

- Under no-BOS, `8/12` cases emitted a valid JSON object and then repeated it until the 512-token limit (`stop_type=limit`).
- Under explicit BOS parity, all `12/12` cases terminate cleanly with EOS, `0` false passes occur, and the `3` verdict errors are conservative false-fails on the 3 `pass` cases (`blind-v2-001/002/003`).

### Scribe (`scribe-lora.gguf` -> `scribe-v2-lora.gguf`, 8 cases, `n_predict=1200`, explicit BOS)

- **Schema Valid**: `0.8750` (`7/8`; `01_corrected_handoff` item 3 omitted `owner`/`due`/`evidence` keys)
- **Source Binding Valid (`as_of` match + valid IDs)**: `0.7500` (`6/8`)
- **Pre-Cutoff Evidence Rate (`timestamp <= as_of`)**: `0.6250` (`5/8`; `02_conditional_assignment` leaked post-cutoff `02-m04/05` and copied `15:02:00Z` into `as_of`; `06_replaced_plan` item 4 cited post-cutoff `06-m05`)
- **Token F1 (micro / macro)**: `0.7637` / `0.7631` (`precision=0.7794`, `recall=0.7486`)
- **Empty Abstention (`08_empty`)**: `empty_fp_rate = 0.0000` (`items: []`), `total_elapsed_s = 67.06` (`avg_latency_s = 8.38`)
- **Prompt Injection (`04_scope_and_injection`)**: Resisted embedded override instructions cleanly.

## 3. Five-Role P2000 GGUF Bundle & 5 GiB VRAM Footprint Matrix

All 5 active symlinks in `<models-root>/gguf/p2000-roles/` resolve to readable `GGUF v3` adapter files, and base `<models-root>/gguf/lfm3-1.7b-base/lfm3-1.7b-base-Q4_K_M.gguf` is a readable `GGUF v3` model:

| Role / Component | Active Symlink | Resolved Target | Size (Bytes) | Size (MiB) | Primary Eval / Status |
|---|---|---|---|---|---|
| Base Model (`Q4_K_M`) | - | `lfm3-1.7b-base-Q4_K_M.gguf` | 1,099,627,488 | 1048.69 | GGUF v3 verified |
| `deriver` (`r=16`) | `deriver-lora.gguf` | `deriver-v20-lora.gguf` | 54,214,624 | 51.70 | Held-out v14 F1 `0.2205` (Part 2) |
| `scribe` (`r=16`) | `scribe-lora.gguf` | `scribe-v2-lora.gguf` | 54,214,624 | 51.70 | Scribe6 F1 `0.6456`, Scribe8 F1 `0.7637` |
| `judge` (`r=16`) | `judge-lora.gguf` | `judge-v1-lora.gguf` | 54,214,720 | 51.70 | Judge12 BOS verdict `0.7500` / tokF1 `0.9667` |
| `verifier` (`r=16`) | `verifier-lora.gguf` | `verifier-v2-lora.gguf` | 54,214,624 | 51.70 | Smoke dev6 tokF1 `0.5957`, schema `0.6667` |
| `pruner` (`r=16`) | `pruner-lora.gguf` | `pruner-v2-lora.gguf` | 54,214,720 | 51.70 | Held-out v1 line F1 `0.7407`, schema `1.0000` |
| **Sum (5 LoRAs)** | - | - | **271,073,312** | **258.52** | All 5 GGUF v3 readable |
| **KV Cache Budget** | - | - | - | **350.00** | P2000 single-context estimate |
| **Total P2000 VRAM Est** | - | - | - | **1657.20** | **Fits 5 GiB (`5120 MiB`); headroom `3462.80 MiB`** |

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

## 2b. Held-Out Eval (`EVAL-ONLY-si-lab-pruner10-log8-reviewed-20261116`)

Evaluated locally on CPU (`lfm3-1.7b-base-Q4_K_M.gguf`, greedy `temperature=0.0`, explicit BOS parity `[bos] + prompt` via `tokenizer.ggml.add_bos_token=bool:true`, verified against HF `AutoTokenizer`):

### Pruner (`pruner-lora.gguf` -> `pruner-v2-lora.gguf`, 10 cases, `n_predict=512`)

- **Format Valid (`score_pruner_sets.py` & strict `KEEP_RE` + EOS)**: `1.0000` (`10/10` EOS; unadapted base: `0/10` valid, `0/10` EOS, F1 `0.0000`)
- **Selection-Set F1 (micro, `score_pruner_sets.py`)**: `0.6000` (`TP=21, FP=16, FN=12`; `precision=0.5676`, `recall=0.6364`, `exact_set_accuracy=0.0000`, `macro_token_f1=0.5850`)
- **NONE Controls (`pe2-02`, `pe2-10`)**: `empty_fp_rate = 1.0000` (`2/2` false-positive retentions: `KEEP 1-2` and `KEEP 1-2,5`)
- **Latency**: `10.68 s` total (`avg = 1.07 s/case`)
- **Semantic Findings**: Near-match retention (`pe2-01`: `kiln-70` retained alongside `kiln-7`), critical omissions on error-context/halted-before-write/bounded-absence/empty-schedule cases (`pe2-04`, `pe2-05`, `pe2-06`, `pe2-09`), and destination line omitted on `pe2-08` (`all_control`).

### Verifier / Log-Analyser (`verifier-lora.gguf` -> `verifier-v2-lora.gguf`, 8 cases, `n_predict=1800`)

- **JSON & Schema Valid (4-section schema + EOS)**: `1.0000` (`8/8` EOS; unadapted base: `0/8` JSON/schema valid, `0/8` EOS, `token_f1_micro=0.1536`)
- **Citation Grounding (`evidence_ids_valid`)**: `1.0000` (`8/8`; zero fabricated log IDs)
- **Token F1 (micro / macro)**: `0.5446` / `0.5250` (`precision=0.5226`, `recall=0.5685`)
- **All-Empty Irrelevant Window (`loge2-03`)**: `empty_fp_rate = 1.0000` (`1/1`: summarized unrelated `css-N1`/`mail-P8` instead of returning 4 empty arrays)
- **Latency**: `89.35 s` total (`avg = 11.17 s/case`)
- **Semantic Findings**: Strong on `loge2-01` (request separation), `loge2-04` (T33 pass with empty findings/unknowns/checks), and `loge2-06` (U18 guess vs `/health` scope); failed throughput arithmetic on `loge2-02` (`36/6 = 6/s`), misread `local_seq=2` on `loge2-07`, and resisted fixture injection on `loge2-08` while over-emitting an `observed_error` and `unknowns`/`next_checks`.

## 3. Five-Role P2000 GGUF Bundle & 5 GiB VRAM Footprint Matrix

All 5 active symlinks in `<models-root>/gguf/p2000-roles/` resolve to readable `GGUF v3` adapter files, and base `<models-root>/gguf/lfm3-1.7b-base/lfm3-1.7b-base-Q4_K_M.gguf` is a readable `GGUF v3` model:

| Role / Component | Active Symlink | Resolved Target | Size (Bytes) | Size (MiB) | Primary Eval / Status |
|---|---|---|---|---|---|
| Base Model (`Q4_K_M`) | - | `lfm3-1.7b-base-Q4_K_M.gguf` | 1,099,627,488 | 1048.69 | GGUF v3 verified |
| `deriver` (`r=16`) | `deriver-lora.gguf` | `deriver-v20-lora.gguf` | 54,214,624 | 51.70 | Held-out v14 F1 `0.2205` (Part 2) |
| `scribe` (`r=16`) | `scribe-lora.gguf` | `scribe-v2-lora.gguf` | 54,214,624 | 51.70 | Scribe6 F1 `0.6456`, Scribe8 F1 `0.7637` |
| `judge` (`r=16`) | `judge-lora.gguf` | `judge-v1-lora.gguf` | 54,214,720 | 51.70 | Judge12 BOS verdict `0.7500` / tokF1 `0.9667` |
| `verifier` (`r=16`) | `verifier-lora.gguf` | `verifier-v2-lora.gguf` | 54,214,624 | 51.70 | Smoke dev6 tokF1 `0.5957`; Log8 tokF1 `0.5446` (schema `1.0000`) |
| `pruner` (`r=16`) | `pruner-lora.gguf` | `pruner-v2-lora.gguf` | 54,214,720 | 51.70 | Held-out v1 F1 `0.7407`; Pruner10 set F1 `0.6000` (format `1.0000`) |
| **Sum (5 LoRAs)** | - | - | **271,073,312** | **258.52** | All 5 GGUF v3 readable |
| **KV Cache Budget** | - | - | - | **350.00** | P2000 single-context estimate |
| **Total P2000 VRAM Est** | - | - | - | **1657.20** | **Fits 5 GiB (`5120 MiB`); headroom `3462.80 MiB`** |

## 4. Base-vs-LoRA 5-Role Held-Out Comparison

| Role | Held-Out Dataset (`n`) | Unadapted Base (`lfm3-1.7b-base-Q4_K_M`) | Best / Active LoRA (`r=16`) |
|---|---|---|---|
| `deriver` | `honcho-deriver-v14-evaluation-only` (`n=4`, `n_predict=1200`) | Probe 7-row base: `schema_valid=0.0000`, `eos=0.0000`, `f1=0.0000`; Historical 80-fx base: `schema_valid=0.61`, `F1=0.08` (not run on `v14`) | `v20` (active): `schema_valid=0.5000`, `f1=0.2205`, `token_f1=0.8535` (`v18`: `f1=0.1815`; `v33`: `f1=0.1754`; `-n 768`: `f1=0.0000`) |
| `scribe` | `scribe8` (`n=8`, `n_predict=1200`, BOS) | Not evaluated separately on `scribe8` | `scribe-v2`: `schema_valid=0.8750`, `source_bind=0.7500`, `pre_cutoff=0.6250`, `token_f1_micro=0.7637`, `empty_fp=0.0000` (`scribe6`: `f1=0.6456`, `schema=1.0000`) |
| `judge` | `judge12` (`n=12`, `n_predict=512`) | Not evaluated separately on `judge12`; No-BOS LoRA parity: `schema_4key=0.3333`, `verdict_acc=0.1667`, `tok_f1=0.9180`, `4/12` EOS | `judge-v1` (BOS): `schema_4key=1.0000` (`12/12` EOS), `strict_checker=0.9167`, `verdict_acc=0.7500`, `false_pass=0`, `tok_f1=0.9667` |
| `verifier` (`log-analyser`) | `log8` (`n=8`, `n_predict=1800`, BOS) | `json_valid=0.0000`, `schema_valid=0.0000`, `evidence_ids=0.0000`, `all_eos=false`, `token_f1_micro=0.1536`, `token_f1_macro=0.1227` | `verifier-v2` (BOS): `json_valid=1.0000`, `schema_valid=1.0000`, `evidence_ids=1.0000`, `all_eos=true`, `token_f1_micro=0.5446`, `empty_fp=1.0000` |
| `pruner` | `pruner10` (`n=10`, `n_predict=512`, BOS) | `format_valid=0.0000`, `strict_schema=0.0000`, `all_eos=false`, `selection_f1=0.0000`, `macro_tok_f1=0.1191` | `pruner-v2` (BOS): `format_valid=1.0000`, `strict_schema=1.0000`, `all_eos=true`, `selection_f1=0.6000` (`P=0.5676`, `R=0.6364`), `empty_fp=1.0000` |

Structured JSON counterpart: [base_vs_lora_5role_summary.json](base_vs_lora_5role_summary.json).

## 5. Deriver Part 4 (`v47`) Launch & Live GDrive Sync (2026-11-16T08:31Z)

- **GDrive Metadata Sync**: Confirmed zero brand-new archives since `honcho-deriver-v47-training-only.zip` (`56` total `.zip` archives; `v47` has `691` train / `4` dev windows in `STATISTICS.json`, `2,144,712` tokens/pass, `max_full_tokens=6398`, `heldout_included=false`).
- **Part 4 Run**: `<kaggle-user>/p2000-deriver-v40-full-part4` (initialized from `deriver_part1/full_part1/checkpoint-pct-80`, `lr=1e-4`, `epochs=2`, `691` train / `24` dev rows, `planned_steps=174`, `estimated_wall_s=4815`, pushed `2026-11-16T08:07:01Z`).
- **Live Kernel Status**: `KernelWorkerStatus.RUNNING` (checked `2026-11-16T08:31:22Z`).
- **Sanitized Snapshots**: [STATE_snapshot.json](STATE_snapshot.json), [full_part4_launch.json](full_part4_launch.json), and [p2000_bundle_verification.json](p2000_bundle_verification.json).

# Pruner LoRA (LFM3-1.7B-Base) - 2xT4 DDP Full Run v3 Evidence Packet, 2026-11-16

This directory contains sanitized training receipts, GPU telemetry, admission audit, warm-start lineage SHA-256s, decile milestone logs, dev metrics, and matched held-out `pruner10` GGUF evaluation results for Pruner Full Run v3 (`<kaggle-user>/p2000-pruner-v3-full`), executed on 2x Tesla T4 GPUs (`cuda:0` and `cuda:1`, `world_size=2`) via PyTorch `torchrun` DDP.

## Executive Summary

- **Kernel Reference**: `<kaggle-user>/p2000-pruner-v3-full` (`KernelWorkerStatus.COMPLETE`)
- **DDP Verification**: `true` (`PER_RANK_STEPS_AND_ALLOCATIONS_OBSERVED`, `steps: [39, 39]`, `rows_processed: [156, 156]`, `known_distinct_uuids: true`, `0` OOM/NaN skips)
- **Dataset & Admission Audit (`--over-limit-policy=reject`)**:
  - `96` base+addenda rows (`72` prior combined + `24` admitted feedback addenda from `botok-feedback-training-addenda-20261116`, `botok-pruner-feedback-batch2-20261116`, `botok-feedback-training-addenda-batch5-20261116`), oversampled to `103` rows (`empty_target_ratio=0.25`) + `12` dev rows.
  - `103 / 103` rows admitted cleanly (`rejected_rows=0`, `truncated_rows=0`, `excess_tokens_total=0`, `30,121` total tokens/pass, `1,044` supervised tokens/pass, length `105..1210 <= 5600`, `data_sha256=23b60f5997e5e50750252d3adc6f6739629edfec8979f4f63b16a21e9d84ee28`).
- **Warm-Start Lineage**:
  - Initialized via `--init-adapter` (`warm_start_branch`) from `pruner_v2/pruner_full/checkpoint-pct-60` (`adapter_model.safetensors` SHA-256 `60dcf29cc69ba495522270328c4b4fa15dab1195bbc2b6daf1917ac2d1c03a8e`) with fresh optimizer, scheduler, and RNG (`optimizer_restored=false`, `scheduler_restored=false`, `rng_restored=false`).
- **Hyperparameters**: `role=pruner`, `lora_r=16`, `lora_alpha=32`, `target_modules=attn-mlp`, `lr=1e-4`, `epochs=3`, `effective_batch=8`, `microbatch=1`, `max_length=5600`.
- **Steps & Timing**: `39 / 39` optimizer steps (`remaining_optimizer_steps=0`), `p95_step_seconds = 2.3515 s`, `elapsed_training_seconds = 119.96 s` (`milestone_eval_seconds_total = 39.60 s`).
- **Peak VRAM**: Rank 0 `4,075.18 MiB` allocated (`4,368.0 MiB` reserved), Rank 1 `4,123.07 MiB` allocated (`4,368.0 MiB` reserved); `nvidia-smi` peak `4,709 MiB` (GPU 0, `36.76%` mean / `100%` peak util across `90` samples) and `4,607 MiB` (GPU 1, `48.32%` mean / `100%` peak util across `90` samples).
- **Exported GGUF Adapters**:
  - `pruner-v3-peak-lora.gguf` (`checkpoint-pct-20`, `54,214,720` bytes, SHA-256 `102c8c016bc91dbdf994aae2f9c1b64fe637224fcb19416c86925c58bf47817f`)
  - `pruner-v3-pct60-lora.gguf` (`checkpoint-pct-60`, `54,214,720` bytes, SHA-256 `2e2614586d0a52e2e811309bce9aac46f5b9bf872849e62b035839b95bc036a3`) — **promoted to active `<models-root>/gguf/p2000-roles/pruner-lora.gguf`**
  - `pruner-v3-lora.gguf` (final `adapter/`, `54,214,624` bytes, SHA-256 `0971d1ae156f5125d3adbbd3acd01e9f947e48a72eef0b10b147ba637d95a48f`)

## Decile Progression & Final Dev Eval (`dev_n=12`, `max_new_tokens=1800`, `capped_count=0`)

*(Note: `pruner_v3` launched at `11:07:51Z` with `20,40,60,80` milestone cadence prior to ingesting `FOR-KAGGLE-ORCHESTRATOR-milestones-and-P2000-20261116.txt`; see `../DOT_MILESTONES_P2000_RESPONSE_20261116.md`.)*

| Checkpoint | Step | Train Loss | Schema Valid | Precision | Recall | Dev F1 | Token F1 | Empty FP | Non-Empty Rate |
|---|---|---|---|---|---|---|---|---|---|
| 20% (`pct-20`) | 8 | 0.3692 | 1.0000 | 0.7677 | 0.6972 | 0.7308 | 0.6861 | 0.5000 | 0.9167 |
| 40% (`pct-40`) | 16 | 0.2713 | 1.0000 | 0.7791 | 0.6147 | 0.6872 | 0.6766 | 0.5000 | 0.9167 |
| **60% (`pct-60`) [Promoted]** | **24** | **0.1850** | **1.0000** | **0.8000** | **0.6239** | **0.7010** | **0.6543** | **0.0000** | **0.8333** |
| 80% (`pct-80`) | 32 | 0.0487 | 1.0000 | 0.7234 | 0.6239 | 0.6700 | 0.6910 | 0.0000 | 0.8333 |
| Final `adapter/` (100%) | 39 | - | 1.0000 | 0.7263 | 0.6330 | 0.6765 | 0.6910 | 0.0000 | 0.8333 |

## Matched Held-Out Benchmark (`EVAL-ONLY-si-lab-pruner10-log8-reviewed-20261116`, `n=10`)

Evaluated locally on CPU (`lfm3-1.7b-base-Q4_K_M.gguf`, `llama-server`, explicit BOS `124894` via `tokenizer.ggml.add_bos_token=bool:true`, `temperature=0.0`, `n_predict=512`, scored with official `score_pruner_sets.py`):

| Candidate | EOS Rate | Format Valid | TP / FP / FN | Precision | Recall | `selection_f1_micro` | `exact_set_acc` | `empty_fp_rate` | Elapsed (s) |
|---|---|---|---|---|---|---|---|---|---|
| `unadapted_base` | 0/10 (10 limit) | 0.0000 | 0 / 0 / 33 | 1.0000 | 0.0000 | 0.0000 | 0.0000 (0/10) | 0.0000 | 513.12 |
| `pruner-v2-lora` (`v2 pct-60`) | 10/10 | 1.0000 | 21 / 16 / 12 | 0.5676 | 0.6364 | 0.6000 | 0.0000 (0/10) | 1.0000 (2/2) | 32.65 |
| `pruner-v3-peak-lora` (`v3 pct-20`) | 10/10 | 1.0000 | 17 / 7 / 16 | 0.7083 | 0.5152 | 0.5965 | 0.3000 (3/10) | 0.0000 (0/2) | 32.99 |
| **`pruner-v3-pct60-lora` (`v3 pct-60`)** | **10/10** | **1.0000** | **25 / 12 / 8** | **0.6757** | **0.7576** | **0.7143** | **0.4000 (4/10)** | **0.0000 (0/2)** | **34.41** |
| `pruner-v3-lora` (`v3 final`) | 10/10 | 1.0000 | 24 / 12 / 9 | 0.6667 | 0.7273 | 0.6957 | 0.3000 (3/10) | 0.0000 (0/2) | 23.01 |

### Per-Case Breakdown (`pruner-v3-pct60-lora` vs `pruner-v2-lora`)

- `pe2-01` (`exact_identifier`): `v2` `KEEP 2,4` -> `v3 pct-60` `KEEP 2` (gold `KEEP 4`; still retains near-match `kiln-70` over `kiln-7`).
- `pe2-02` (`exact_identifier_absence`): `v2` `KEEP 1-2` (FP) -> `v3 pct-60` **`KEEP none` (EXACT)**.
- `pe2-03` (`requested_semantic_absence`): `v2` `KEEP 1-3` -> `v3 pct-60` `KEEP 2-4` (gold `KEEP 1,3`).
- `pe2-04` (`bounded_absence_not_global`): `v2` `KEEP 1-4,7` -> `v3 pct-60` `KEEP none` (gold `KEEP 1-5`; over-abstained on bounded-absence evidence).
- `pe2-05` (`error_context`): `v2` `KEEP 1-2,7-8` (`tp=0`) -> `v3 pct-60` `KEEP 1-7` (`tp=3, fn=0`, gold `KEEP 3-5`).
- `pe2-06` (`halted_before_write`): `v2` `KEEP 1-3,7-8` (`tp=2`) -> `v3 pct-60` `KEEP 1-5,7-8` (`tp=4, fn=1`, gold `KEEP 1,3-6`).
- `pe2-07` (`partial_write_before_failure`): `v2` `KEEP 1-3,6` -> `v3 pct-60` **`KEEP 1-6` (EXACT)**.
- `pe2-08` (`all_control`): `v2` `KEEP 1-5` -> `v3 pct-60` **`KEEP all` (EXACT)**.
- `pe2-09` (`semantic_empty_schedule`): `v2` `KEEP 1-3,7` -> `v3 pct-60` `KEEP all` (`tp=5, fp=2, fn=0`, gold `KEEP 1-5`).
- `pe2-10` (`none_control`): `v2` `KEEP 1-2,5` (FP) -> `v3 pct-60` **`KEEP none` (EXACT)**.

Full pinned hashes and raw predictions are in [gguf_eval_pruner10_v3.json](gguf_eval_pruner10_v3.json) and [lineage_and_admission_receipt.json](lineage_and_admission_receipt.json).

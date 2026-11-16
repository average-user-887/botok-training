# Verifier (Log-Analyser) LoRA (LFM3-1.7B-Base) - 2xT4 DDP Full Run v3 Evidence Packet, 2026-11-16

This directory contains sanitized training receipts, GPU telemetry, decile milestone logs (`10%..90%`), dev metrics, and live Quadro P2000 GGUF `log8` held-out evaluation results for Verifier Full Run v3 (`<kaggle-user>/p2000-verifier-v3-full`), executed on 2x Tesla T4 GPUs (`cuda:0` and `cuda:1`, `world_size=2`) via PyTorch `torchrun` DDP.

## Executive Summary

- **Kernel Reference**: `<kaggle-user>/p2000-verifier-v3-full` (`KernelWorkerStatus.COMPLETE`)
- **Lineage Mode**: `warm_start_branch` initialized from `verifier_v2` `adapter/` (`--init-adapter`, `init_adapter_sha256=cce2c2d93200aa4f92308c3ccf349747b0d4e7848a7e1cfc7f1bec790952b247`, `optimizer_restored=false`, `scheduler_restored=false`, `rng_restored=false`)
- **DDP Verification**: `true` (`PER_RANK_STEPS_AND_ALLOCATIONS_OBSERVED`, `steps: [24, 24]`, `rows_processed: [93, 93]`, `known_distinct_uuids: true`, `skips_clean: true`)
- **Dataset**: 48 base training rows (`v1` 10 + `v2` 20 + `increment-52` 12 + `batch4/PROPOSED-verifier` 6 with 2 all-empty) oversampled to `62` rows (`empty_target_ratio=0.25`, `no_downweight_empty=true`, `41,758` tokens/epoch, `13,080` supervised tokens/epoch) & `6` dev rows (`max_length=5600`, `over_limit_policy=reject`, `0` rejected rows)
- **Hyperparameters**: `role=verifier`, `lora_r=16`, `lora_alpha=32`, `target_modules=attn-mlp`, `lr=1e-4`, `epochs=3`, `effective_batch=8`, `microbatch=1`
- **Optimizer Steps**: `24 / 24` (`p95_step_seconds = 3.4204 s`, `elapsed_training_seconds = 615.71 s`, `milestone_eval_seconds_total = 535.40 s`, `0` OOM/NaN skips)
- **Peak VRAM (PyTorch)**: Rank 0: `4,293.35 MiB` allocated (`4,482.0 MiB` reserved); Rank 1: `4,322.42 MiB` allocated (`4,482.0 MiB` reserved)
- **Selected Checkpoint (Actual P2000 `log8` Eval Winner)**: `checkpoint-pct-90` (`verifier-v3-pct90-lora.gguf`, `sha256=1e5ef6776785314c8ab508efb46a2d4ac6158b9ae4581f639f573fc802d74cec`)
  - Updated `<models-root>/gguf/p2000-roles/verifier-lora.gguf -> verifier-v3-pct90-lora.gguf`
  - **Held-Out `EVAL-ONLY` `log8` on Actual Quadro P2000** (`n_predict=1800`, `temperature=0.0`, `bos=124894`):
    - `schema_valid = 1.0000` (8/8 vs `v2` 7/8 `0.8750`)
    - `evidence_ids_valid = 1.0000` (8/8)
    - `empty_fp_rate = 0.0` (eliminated `v2`'s `1.0` empty false positive on `loge2-03`)
    - `token_f1_micro = 0.5602` (`P=0.5684`, `R=0.5521`, `token_f1_macro = 0.5822`, `all_eos=true`) vs `v2` (`micro_F1=0.5538`, `macro_F1=0.5367`, `empty_fp=1.0`) and `unadapted_base` (`schema_valid=0.0000`, `micro_F1=0.1000`)

## Kaggle Decile Milestone Progression (`10..90%`, `max_new_tokens=1800`)

| Pct | Step | Train Loss | Dev Schema Valid | Dev Precision | Dev Recall | Dev F1 | Dev Token F1 | Empty FP |
|---|---|---|---|---|---|---|---|---|
| 10 | 3 | 0.6400 | 1.0000 | 1.0000 | 0.0599 | 0.1130 | 0.1383 | 0.0000 |
| 20 | 5 | 0.4268 | 1.0000 | 1.0000 | 0.0599 | 0.1130 | 0.1383 | 0.0000 |
| 30 | 8 | 0.6046 | 1.0000 | 1.0000 | 0.0599 | 0.1130 | 0.1383 | 0.0000 |
| 40 | 10 | 0.5516 | 1.0000 | 1.0000 | 0.0599 | 0.1130 | 0.1383 | 0.0000 |
| 50 | 12 | 0.5092 | 1.0000 | 0.6304 | 0.4970 | 0.5558 | 0.5150 | 0.0000 |
| 60 | 15 | 0.3109 | 0.8333 | 0.6425 | 0.4950 | 0.5592 | 0.5188 | 0.0000 |
| 70 | 17 | 0.5218 | 0.8333 | 0.6316 | 0.5030 | 0.5600 | 0.5175 | 0.0000 |
| 80 | 20 | 0.5080 | 0.8333 | 0.6266 | 0.4890 | 0.5493 | 0.5120 | 0.0000 |
| 90 | 22 | 0.4138 | 1.0000 | 0.6289 | 0.4870 | 0.5489 | 0.5099 | 0.0000 |
| Final `adapter/` | 24 | - | 1.0000 | 0.6225 | 0.4970 | 0.5527 | 0.5141 | 0.0000 |

## Actual Quadro P2000 Held-Out `log8` Sweep (`lfm3-1.7b-base-Q4_K_M.gguf`, `bos=124894`, `n_predict=1800`)

| Candidate | Schema Valid | Evidence Valid | All EOS | Empty FP | Precision | Recall | Micro F1 | Macro F1 | Gen tok/s | P95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| `unadapted_base` | 0.0000 | 0.0000 | false | 0.0 | 0.5294 | 0.0552 | 0.1000 | 0.0563 | 58.57 | 31096.11 |
| `v2` (`verifier-v2-lora`) | 0.8750 | 0.8750 | true | 1.0 | 0.5310 | 0.5787 | 0.5538 | 0.5367 | 46.47 | 8850.01 |
| `v3-pct10`..`pct40` | 1.0000 | 1.0000 | true | 0.0 | 1.0000 | 0.0818 | 0.1512 | 0.2572 | 45.53–46.06 | 899.77–1076.03 |
| `v3-pct50` | 1.0000 | 1.0000 | true | 1.0 | 0.5320 | 0.5276 | 0.5298 | 0.4776 | 45.93 | 7756.02 |
| `v3-pct60` | 1.0000 | 1.0000 | true | 1.0 | 0.5037 | 0.5562 | 0.5287 | 0.4960 | 45.83 | 7188.27 |
| `v3-pct70` | 1.0000 | 1.0000 | true | 1.0 | 0.5280 | 0.5787 | 0.5522 | 0.5177 | 45.72 | 7496.27 |
| `v3-pct80` | 0.8750 | 0.8750 | true | 1.0 | 0.5324 | 0.5378 | 0.5351 | 0.4800 | 45.84 | 9906.76 |
| **`v3-pct90` (winner)** | **1.0000** | **1.0000** | **true** | **0.0** | **0.5684** | **0.5521** | **0.5602** | **0.5822** | **45.76** | **8575.95** |
| `v3-adapter` | 1.0000 | 1.0000 | true | 1.0 | 0.5258 | 0.5624 | 0.5435 | 0.4855 | 45.90 | 7986.11 |

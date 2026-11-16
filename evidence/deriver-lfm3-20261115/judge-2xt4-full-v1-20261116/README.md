# Judge LoRA (LFM3-1.7B-Base) - 2xT4 DDP Full Run v1 Evidence Packet, 2026-11-16

This directory contains sanitized training receipts, GPU telemetry, milestone logs, dev metrics, and CPU GGUF smoke inference results for Judge Full Run v1 (`<kaggle-user>/p2000-judge-v1-full`), executed on 2x Tesla T4 GPUs (`cuda:0` and `cuda:1`, `world_size=2`) via PyTorch `torchrun` DDP.

## Executive Summary

- **Kernel Reference**: `<kaggle-user>/p2000-judge-v1-full`
- **Kernel Status**: `KernelWorkerStatus.COMPLETE`
- **DDP Verification**: `true` (`PER_RANK_STEPS_AND_ALLOCATIONS_OBSERVED`, `steps: [30, 30]`, `rows_processed: [54, 54]`, `known_distinct_uuids: true`)
- **Dataset**: combined reviewed Judge `train.sft.jsonl` (`18` rows: `si-lab-judge-proposed-candidate-v1` 6 + `si-lab-four-role-reviewed-increment-52-20261115/judge` 12; `5,905` tokens/epoch, `699` supervised tokens/epoch) & `dev.sft.jsonl` (`2` author-visible dev rows from `si-lab-judge-proposed-candidate-v1`), `max_length=5600`
- **Hyperparameters**: `role=judge`, `lora_r=16`, `lora_alpha=32`, `target_modules=attn-mlp`, `lr=2e-4`, `epochs=6`, `effective_batch=4`, `microbatch=1`
- **Optimizer Steps**: `30 / 30` planned steps completed (`remaining_optimizer_steps = 0`, `stop_requested = false`)
- **P95 Step Latency**: `0.86 s/step` (`0.8577 s`)
- **Elapsed Time**: `61.38 s` training (`29.49 s` across 4 milestone probes; `124.5 s` `torchrun` wall; `161.1 s` total script runtime)
- **Peak VRAM Allocated**: GPU 0: `3,885.43 MiB`, GPU 1: `3,885.43 MiB` (`4,368 MiB` PyTorch reserved on both GPUs)
- **Peak VRAM (nvidia-smi)**: GPU 0: `4,607 MiB`, GPU 1: `4,607 MiB` (`100%` peak util on both GPUs across `126` samples)
- **OOM / NaN Grad Skips**: `0` OOM skips, `0` NaN grad skips
- **Selected Checkpoint**: `checkpoint-pct-40` (step `12` / 40%, `dev_schema_valid = 1.0000`, `dev_f1 = 0.8889`, `dev_token_f1 = 0.9000`, `dev_precision = 0.9600`, `dev_recall = 0.8276`; higher dev F1 than final `adapter/` step `30` at `0.8679`)
- **GGUF LoRA Export & Held-Out / Smoke Eval**:
  - Exported `checkpoint-pct-40` to `<models-root>/gguf/p2000-roles/judge-v1-lora.gguf` (`51.70 MiB` / `54,214,720` bytes; within P2000 budget) and linked `judge-lora.gguf -> judge-v1-lora.gguf`.
  - **Held-Out Eval**: `eval_n = 0` (no held-out Judge evaluation split exists locally in staged bundles; `si-lab-judge-proposed-candidate-v1` and `si-lab-four-role-reviewed-increment-52-20261115` explicitly document no blind/held-out test set).
  - **CPU GGUF Smoke Check (`dev_2` prompts, Q4_K_M + `judge-v1-lora.gguf`, `n_predict=512`, greedy)**:
    - `schema_valid_4key = 1.0000` (2/2 EOS, valid JSON matching `judge` 4-key + `attribution_invalid` structure)
    - `schema_valid_strict = 0.5000` (`judge-candidate-dev-01` exact structural + verdict match; `judge-candidate-dev-02` produced valid JSON arrays but inconsistent `verdict: "pass"` with `missed=[4]`, `hallucinated=[4]`)
    - `smoke_dev2_token_f1_micro = 0.8679` (`token_f1_macro = 0.8793`, `total_elapsed_s = 7.84`, `avg_latency_s = 3.92`)

## Milestone Progression & Final Dev Eval

| Checkpoint | Step | Train Loss | Schema Valid | Precision | Recall | Dev F1 | Token F1 | Empty FP | Non-Empty Rate |
|---|---|---|---|---|---|---|---|---|---|
| 20% (`max_new=256`) | 6 | 0.1623 | 1.0000 | 0.9130 | 0.7241 | 0.8077 | 0.8141 | 0.0000 | 1.0000 |
| **40% (`max_new=256`) [Selected]** | **12** | **0.0318** | **1.0000** | **0.9600** | **0.8276** | **0.8889** | **0.9000** | **0.0000** | **1.0000** |
| 60% (`max_new=256`) | 18 | 0.0152 | 1.0000 | 0.9583 | 0.7931 | 0.8679 | 0.8793 | 0.0000 | 1.0000 |
| 80% (`max_new=256`) | 24 | 0.0169 | 1.0000 | 0.9583 | 0.7931 | 0.8679 | 0.8793 | 0.0000 | 1.0000 |
| Final `adapter/` (`max_new=1800`) | 30 | - | 1.0000 | 0.9583 | 0.7931 | 0.8679 | 0.8793 | 0.0000 | 1.0000 |

## CPU GGUF Smoke Check Breakdown (`dev_2`)

| Case ID | Prompt Tokens | Predicted Tokens | Stop | 4-Key Valid | Strict Valid | Verdict (Pred/Gold) | Token F1 | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| `judge-candidate-dev-01` | 397 | 46 | `eos` | true | true | `fail` / `fail` | 1.0000 | 3.83 |
| `judge-candidate-dev-02` | 416 | 46 | `eos` | true | false | `pass` / `fail` | 0.7586 | 4.01 |

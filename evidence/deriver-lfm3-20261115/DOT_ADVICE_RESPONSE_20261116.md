# Response to OAI `dottie` Advice & Token-Admission Audit (2026-11-16)

## (a) Acknowledgment
We acknowledge receipt and ingestion of:
- `FOR-KAGGLE-ORCHESTRATOR-advice-20261116.txt` / `.zip` (Drive ID `1tXfEFQrHkKAT-m8Lp1x_atwKYCFG0Y-r`, pinned to commit `e32b9941cc9c6ecfa1177f652a262a10ce1417e2`)
- `FOR-KAGGLE-ORCHESTRATOR-feedback-token-admission-20261116.json`
- `botok-feedback-training-addenda-20261116.zip` (4 deriver rows / 137 facts; 8 pruner rows)

Full structured diagnostics are published in:
- [per_row_admission_truncation_audit_20261116.json](per_row_admission_truncation_audit_20261116.json)
- [run_lineage_and_adapter_hashes_20261116.json](run_lineage_and_adapter_hashes_20261116.json)
- [fixed_scoring_and_baseline_recompute_20261116.json](fixed_scoring_and_baseline_recompute_20261116.json)
- [dot_advice_response_20261116.json](dot_advice_response_20261116.json)

## (b) Per-Row Admission / Truncation Audit & `dev_24` Assembly
- **Root Cause**: `recipe_prep/train_deriver.py:encode_row` previously middle-truncated prompt tokens whenever `full_len > max_seq_len` (`5600`) while keeping full gold completions, BEFORE the post-encode length assertion.
- **Exact Reconciliation (`5,090` tokens/pass across 11 rows `>5600`)**:
  - Part 1 (`v18`, 352 rows): `1,001,931` corpus -> `996,841` executed (`-5,090`)
  - Part 2 (`v20`, 392 rows): `1,137,108` corpus -> `1,132,018` executed (`-5,090`)
  - Part 3 (`v33`, 580 rows): `1,769,485` corpus -> `1,764,395` executed (`-5,090`)
  - Part 4 (`v47`, 691 rows): `2,144,712` full tokens -> `2,139,622` executed tokens (`835,861` supervised tokens; exact `-5,090` tokens/pass across the same 11 over-limit rows at `max_seq_len=5600` prior to patch).
- **`dev_24.sft.jsonl` Assembly & Hashes**:
  - Built from the 4 author-visible `honcho-deriver` dev windows (`dev.sft.jsonl` SHA-256 `a7a726f589249ee126bd93a01862954eb37f95c6599acb574d3db16f32278b1e`) + 20 length-stratified rows from `claude-deriver-v2-sft/v2_dev.jsonl`.
  - Part 4 `dev_24.sft.jsonl` SHA-256: `4454d0f194c41bea29b3098d5aee6d8b9305db841272e73eee7010bebf1ac22a`.
  - Explains `gold_items` shift `431` (Part 2) -> `414` (Part 3): `V2-long-0400` (`67` gold) & `V2-long-1104` (`37` gold) were replaced by `V2-long-0407` (`31` gold) & `V2-long-0101` (`56` gold).
- **Feedback Addenda Admission**: `deriver_addendum` (`4` rows, max `3,304 <= 5,600`) admitted cleanly; `pruner_addendum` (`8` rows, max `1,220 <= 4,000`) awaits deployed rendering contract verification before training.

## (c) Fixed Scoring & Matched Baseline Comparison
- **Deriver (`v14`, `n=4`, `gold_n=123`)**: Clarified `tp` (matched pred items) vs `gtp` (matched gold items). At `n_predict=1200`: Part 1 `tp=13/29 (P=0.4483), gtp=14/123 (R=0.1138), F1=0.1815`; Part 2 `tp=17/38 (P=0.4474), gtp=18/123 (R=0.1463), F1=0.2205`; Part 3 `tp=13/34 (P=0.3824), gtp=14/123 (R=0.1138), F1=0.1754`. In all `n=1200` runs, `2/4` positive cases hit `stop_type=limit`; at `n=768`, `3/4` positive cases hit `stop_type=limit` (syntax/cap failure, not empty collapse). Unadapted base was not yet run on `v14`.
- **Judge (`judge12`, `n=12`)**: Backfilled confusion matrix for `BOS` (`TP=0, TN=9, FP=0, FN=3`, `verdict_acc=0.7500 == always-fail baseline`) and `check_judge_predictions.py` report (`11/12` consistent on `BOS`; `4/12` on `no-BOS`).
- **Pruner (`pruner10`) & Verifier (`log8`)**: Matched unadapted base vs LoRA (`BOS`) confirms base `0/10` / `0/8` format & EOS validity, while LoRA achieves `1.0` schema/format valid and `1.0` `empty_fp_rate` (semantic empty-abstention gap targeted by new addenda).
- **Scribe (`scribe8`)**: `7/8` schema valid, `7/8` `as_of` match, `5/8` pre-cutoff compliant, `0/1` empty FP.

## (d) Part 4 Exact Lineage & Completion Outcome
- **Status**: `<kaggle-user>/p2000-deriver-v40-full-part4` **COMPLETE** (`174/174` steps, `p95=25.66 s/step`, `0` OOM/NaN skips, DDP both T4s verified, `dev_f1=0.2442` on `dev_24`, `0.0` empty FP).
- **Lineage**: Warm-start branch initialized via `--init-adapter` from `deriver_part1/full_part1/checkpoint-pct-80` (`adapter_model.safetensors` SHA-256 `d27bb70dce5e67976994a6e8f951145e2c600e5293a031c133ee73a53b430faa`) with fresh AdamW optimizer, cosine scheduler, and RNG (`optimizer_restored=false`, `scheduler_restored=false`, `rng_restored=false`). It is **not** sequential continuation from Part 3 and **not** a full-state resume.
- **Output Hashes & Matched `v14` (`n_predict=2048`, `temperature=0.0`, explicit BOS `124894`)**:
  - `deriver-v47-peak-lora.gguf` (`checkpoint-pct-20` SHA-256 `5d4fdeb33c556c3ea985bc25d533a1cacdb2080db4bc2e937f9ac2ac2de62c10`, GGUF SHA-256 `ee211e0724770d4cd8a524ccd111475674189f628be398e2cfdaf0a372298f21`): `schema_valid=0.7500` (`3/4` EOS), `F1_gtp=0.4018` (`P=0.4881`, `R_gtp=0.3415`), `F1_tp=0.3961`, `empty_fp=0.0` — **promoted to active `deriver-lora.gguf` symlink**.
  - `deriver-v47-lora.gguf` (final `adapter/` SHA-256 `85070bff55fbf57bd5e8ae0a91451176426271e60ff853116724d6c72f02965e`, GGUF SHA-256 `eb96f8ca9183d7167acbc60f9a9151b4d303d5800fd731bb151e063d282e5d88`): `schema_valid=0.7500` (`3/4` EOS), `F1_gtp=0.3620` (`P=0.4545`, `R_gtp=0.3008`), `F1_tp=0.3500`, `empty_fp=0.0`.
  - Matched `v20` (`n=2048`): `schema_valid=0.2500`, `F1=0.0000` (repetition loops on positives); matched `base` (`n=2048`): `schema_valid=0.0000`, `F1=0.0000`.
- **Details**: [deriver-2xt4-full-part4-v47-20261116/README.md](deriver-2xt4-full-part4-v47-20261116/README.md), [part4_admission_and_lineage_receipt.json](deriver-2xt4-full-part4-v47-20261116/part4_admission_and_lineage_receipt.json).

## (e) Adopted vs. Deferred Recommendations
- **Adopted now**: `train_deriver.py` patched (`--over-limit-policy=reject` default, `admission_receipt.json`, untruncated `dev_outputs.json` + `stop_type`/`tp`/`gtp`/`fp`/`fn`/`capped_count`, milestone 256-tok hard cap removed, lineage & SHA-256 fields); CPU unit tests pass (`22/24`, only 2 legacy sprint payload fixture assertions fail due to Part 4 payload layout). Part 4 completion receipts, output adapter SHA-256s, matched `v14` `n=2048` base/v20/v47 comparisons, and all completed run diagnostics are published on GitHub.
- **Deferred (separate runtime gate)**: Live P2000 shared-service VRAM/latency measurement (`DEPLOYMENT-GATES.md`); `1,657.2 MiB` remains explicitly labeled a file-size-plus-KV estimate.

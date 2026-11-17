# Deriver Part 5 (LFM3-1.7B-Base LoRA) — run + P2000 evaluation, 2026-11-16

Warm start: Part 4 `checkpoint-pct-20` (adapter SHA-256 5d4fdeb3…2c10, fresh optimizer/scheduler/RNG).
Base pinned: LiquidAI/LFM3-1.7B-Base @ 7bca5038823c0d0f14efe06439109354a10e014c.

## Training (Colab L4, completed)
288/288 steps, 0 NaN, 0 OOM skips, bf16, micro 2 x accum 4 (eff. batch 8), 3 epochs, max-seq 5600,
lr 8e-5, LoRA r16/a32 attn+mlp, 766 train rows, dev_24 (eval skipped in-run). 5,807 s train, 7,817 s session.
Step p50 20.4 s / p95 23.4 s; peak reserved 21.18 GiB of ~22 (L4 headroom ~4%; an L4 rerun should use micro 1 x accum 8).
Trainer fix `ORCH_PATCH_MODEL_TO_DEVICE` (colab/PATCH.txt). Receipts: run_receipts/. Adapters (845 MB) are NOT in git.
Single-batch train loss at deciles 10..90%: .246 .241 .313 .240 .234 .134 .193 .228 .199 (noisy, no trend).
Not in this run: newer Drive packs (pruner batches 23-33, deriver batches up to 22).

## P2000 evaluation (hpserver Quadro P2000, runtime p2000-a4d4f2f-sm61)
Base Q4_K_M + f32 LoRA GGUFs applied per request; temp 0, n_predict 2048, ctx 8192, no prompt cache, chat template + BOS from pinned tokenizer.
Scorer: score_prediction from train_deriver.py @490c643. dev_24, 414 gold facts. Only d1 was unloaded (per-model POST); watchdog restored it (p2000_eval/RESTORE_STATUS.txt).

| arm | valid | P | R | F1 | limit |
|---|---|---|---|---|---|
| base | 0/24 | 0 | 0 | 0 | 18 |
| part4 pct20 | 18 | .290 | .171 | .216 | 6 |
| p5 pct10 | 19 | .285 | .191 | .229 | 5 |
| p5 pct20 | 17 | .399 | .186 | .254 | 7 |
| p5 pct30 | 18 | .306 | .138 | .190 | 6 |
| p5 pct40 | 18 | .258 | .169 | .204 | 6 |
| p5 pct50 | 19 | .265 | .184 | .217 | 5 |
| p5 pct60 | 18 | .294 | .188 | .230 | 6 |
| p5 pct70 | 17 | .351 | .171 | .230 | 7 |
| p5 pct80 | 19 | .235 | .184 | .206 | 5 |
| p5 pct90 | 19 | .294 | .217 | .250 | 5 |
| p5 final | 18 | .313 | .193 | .239 | 6 |

Reading: adapters work (base 0/24 valid) but no Part 5 checkpoint is clearly better than Part 4 pct20 on 24 rows (noise-level);
5-7 rows per adapter arm hit the 2048-token limit. No candidate selected; nothing deployed. Sealed eval not used.

## LFM3 LoRA -> GGUF converter (gguf/CONVERTER.txt)
Upstream ggml-org/llama.cpp c7518dbec59281c430ca5278ab4e60eadbc88b13 `convert_lora_to_gguf.py --outtype f32`, base config from the pinned revision.
Validated: Part 4 pct20 adapter -> GGUF byte-identical to deriver-v47-peak-lora.gguf (ee211e07…8f21). Part 5 GGUF hashes: gguf/SHA256SUMS.

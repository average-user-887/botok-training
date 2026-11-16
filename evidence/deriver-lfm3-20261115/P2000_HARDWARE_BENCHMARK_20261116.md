# Live Quadro P2000 Hardware Benchmark & 5-LoRA Residency Gate (2026-11-16)

This report records actual hardware execution on `<p2000-host>` (**Quadro P2000, 5,120 MiB VRAM**, `llama.cpp` `p2000-a4d4f2f-sm61` `llama-server`) for `lfm3-1.7b-base-Q4_K_M.gguf` (`sha256: 740c14a9c4bad8b2a719de26c409d1539cae2b784817e58a869715e580c80c2f`) with `--override-kv tokenizer.ggml.add_bos_token=bool:true` (`bos=124894`), `temperature=0.0`, `-ngl 99 -fa on -np 1 -c 8192 --cache-type-k q4_0 --cache-type-v q4_0`.

Full sanitized raw telemetry & per-case outputs: [`p2000_hardware_benchmark_20261116.json`](./p2000_hardware_benchmark_20261116.json).

## 1. Simultaneous 5-LoRA Residency & Speed Gate (`deriver`, `pruner`, `verifier`, `judge`, `scribe`)

All 5 role adapters were loaded simultaneously into one `llama-server` process via `--lora <5 comma-separated paths> --lora-init-without-apply` (verified `loaded_adapters_count == 5` via `/lora-adapters`) and invoked per-request via `"lora": [{"id": k, "scale": 1.0}]` across 2 rounds (10 requests total):

| Configuration | Load Success | Process Peak VRAM (MiB) | Card Peak VRAM / Total (MiB) | Card Headroom (MiB) | P50 (ms) | P95 (ms) | Prompt tok/s | Gen tok/s |
|---|---|---|---|---|---|---|---|---|
| **Colocated 5-LoRA + live `Qwen3-Embedding-4B`** | **true** | **1,480** | **4,455 / 5,120** | **665** | 2,098.05 | 2,236.87 | 340.04 | 47.37 |
| **Solo 1-LoRA (`verifier`)** | true | 1,272 | 1,277 / 5,120 | 3,843 | 2,121.91 | 2,156.16 | 504.53 | 47.36 |
| **Solo 5-LoRA (`deriver+pruner+verifier+judge+scribe`)** | **true** | **1,480** | **1,485 / 5,120** | **3,635** | **2,091.01** | **2,137.24** | **354.39** | **47.68** |

Key measured findings:
- **Replaces `1,657.2 MiB` estimate with measured hardware proof**: Base `LFM3-1.7B-Base` `Q4_K_M` + all 5 LoRAs + 8k `q4_0` KV cache + CUDA scratch uses **`1,480 MiB`** process VRAM (`1,485 MiB` card total solo, `3,635 MiB` free headroom).
- **Fits live alongside resident `Qwen3-Embedding-4B` (`~2,938–2,970 MiB`)**: Peak card VRAM during colocated 5-LoRA inference was **`4,455 / 5,120 MiB`** (**`665 MiB`** remaining headroom) without OOM or reload.
- **Zero adapter-swap overhead**: Switching across all 5 roles per request ran at **`47.37–47.68 gen tok/s`**, `p95 = 2,137.24 ms` (solo) / `2,236.87 ms` (colocated).

## 2. Service & Container Rollback Verification

- Pre-benchmark inspection found `0` GPU-using Docker containers (`25` CPU containers remained running) and `2` `llama-swap` GPU processes (`decision-d1-2b`, `qwen3-embedding-4b-p2000`).
- `decision-d1-2b` and `qwen3-embedding-4b-p2000` were cleanly unloaded via `POST /api/models/unload/<model>` and restored inside both Python `try ... finally` and remote bash `trap cleanup EXIT INT TERM`.
- **Rollback Verified (`rollback_verified: true`)**: both `decision-d1-2b` (`1,982 MiB`) and `qwen3-embedding-4b-p2000` (`2,938 MiB`) returned to `state: "ready"` (`4,925 / 5,120 MiB`), and all `25` Docker containers remained `Up` / healthy.

## 3. Verifier v3 Held-Out `log8` Benchmark on Actual P2000

- See [`verifier-2xt4-full-v3-20261116/README.md`](./verifier-2xt4-full-v3-20261116/README.md) for the full 10%–90% decile sweep on `EVAL-ONLY-si-lab-pruner10-log8-reviewed-20261116`.
- `v3-pct90` (`verifier-v3-pct90-lora.gguf`) achieved **`micro_F1 = 0.5602`**, **`macro_F1 = 0.5822`**, **`schema_valid = 1.0000`**, **`evidence_ids_valid = 1.0000`**, and **`empty_fp_rate = 0.0`**, outperforming `v2` (`micro_F1 = 0.5538`, `schema_valid = 0.8750`, `empty_fp_rate = 1.0`) and `unadapted_base` (`micro_F1 = 0.1000`, `schema_valid = 0.0000`).

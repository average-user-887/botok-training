# Honcho deriver benchmark, P2000 candidates (aggregate only; real private fixtures, no text included)

Matching: embedding cosine >= 0.80 vs gold; halluc = observations whose max cosine to source sentences < 0.55. n = scored fixtures (some models run twice).

| model | n | schema-valid | hard-fail | obs/fx | recall | precision | F1(mean/fx) | halluc | determinism | p50 lat s | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma3-4b-s | 80 | 0.49 | 0.62 | 3.0 | 0.04 | 0.08 | 0.05 | 0.04 | nan | 44.2 | 19.9 |
| gemma4-e2b-s | 80 | 0.41 | 0.65 | 3.0 | 0.25 | 0.42 | 0.29 | 0.01 | nan | 28.1 | 34.6 |
| granite-h-micro-s | 80 | 0.99 | 0.21 | 8.5 | 0.09 | 0.20 | 0.12 | 0.03 | nan | 15.9 | 18.2 |
| granite-micro-s | 80 | 0.96 | 0.38 | 9.3 | 0.08 | 0.17 | 0.10 | 0.03 | nan | 16.4 | 19.4 |
| lfm2-2.6b-s | 80 | 0.86 | 0.31 | 7.8 | 0.04 | 0.09 | 0.06 | 0.02 | nan | 15.1 | 28.6 |
| lfm25-1.2b | 160 | 1.00 | 0.01 | 5.5 | 0.01 | 0.05 | 0.03 | 0.20 | 0.21 | 2.5 | 53.1 |
| llama32-3b-s | 80 | 0.89 | 0.62 | 15.0 | 0.09 | 0.11 | 0.10 | 0.12 | nan | 21.9 | 22.2 |
| ministral-3b | 80 | 0.64 | 0.60 | 5.6 | 0.13 | 0.18 | 0.14 | 0.02 | nan | 30.9 | 22.5 |
| qwen35-2b | 160 | 0.85 | 0.40 | 7.8 | 0.06 | 0.12 | 0.08 | 0.19 | 0.10 | 9.5 | 34.7 |
| smollm3-3b-s | 80 | 0.78 | 0.51 | 7.8 | 0.07 | 0.12 | 0.11 | 0.07 | nan | 18.8 | 24.4 |
| gemma4-e2b-noreason | 80 | 0.47 | 0.61 | 3.8 | 0.25 | 0.38 | 0.28 | 0.02 | nan | 27.9 | 34.6 |
| lfm3-base | 80 | 0.61 | 0.46 | 5.9 | 0.08 | 0.13 | 0.08 | 0.07 | nan | 16.1 | 44.9 |
| gemma4-e2b-cap3k | 80 | 0.99 | 0.41 | 10.4 | 0.22 | 0.39 | 0.26 | 0.02 | nan | 33.2 | 34.5 |
| gemma3-1b | 80 | 0.95 | 0.29 | 6.7 | 0.03 | 0.07 | 0.04 | 0.24 | nan | 6.5 | 35.9 |
| llama32-1b | 80 | 0.91 | 0.15 | 4.3 | 0.02 | 0.08 | 0.05 | 0.16 | nan | 4.3 | 41.3 |
| lfm2-1.2b-extract-q8 | 80 | 0.74 | 0.28 | 6.0 | 0.02 | 0.05 | 0.03 | 0.32 | nan | 5.7 | 47.2 |
| nuextract2-2b | 80 | 1.00 | 0.00 | 1.2 | 0.00 | 0.00 | 0.00 | 0.49 | nan | 2.2 | 6.8 |
| exp-extractor-1.2b | 80 | 0.57 | 0.42 | 0.7 | 0.00 | 0.04 | 0.03 | 0.09 | nan | 4.1 | 32.8 |
| exp-extractor-350m | 80 | 0.93 | 0.07 | 0.6 | 0.00 | 0.02 | 0.07 | 0.22 | nan | 1.0 | 40.5 |
| qwen35-2b-cap3k | 80 | 0.93 | 0.41 | 9.4 | 0.06 | 0.11 | 0.09 | 0.08 | nan | 11.3 | 34.6 |
| gemma4-e2b-cap3k-r2 | 80 | 0.97 | 0.45 | 11.8 | 0.25 | 0.37 | 0.28 | 0.02 | nan | 36.7 | 34.7 |
| lfm3-deriver-tuned | 160 | 0.98 | 0.04 | 2.9 | 0.03 | 0.20 | 0.15 | 0.01 | 0.83 | 2.1 | 20.1 |

## Tuned LFM3-1.7B deriver (v1 LoRA, 330 synthetic examples) on the same fixtures

| model | n | schema-valid | hard-fail | obs/fx | recall | precision | F1(mean/fx) | halluc | determinism | p50 lat s | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lfm3-deriver-tuned | 160 | 0.98 | 0.04 | 2.9 | 0.03 | 0.20 | 0.15 | 0.01 | 0.83 | 2.1 | 20.1 |

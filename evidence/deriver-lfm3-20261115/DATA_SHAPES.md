# DATA_SHAPES

Token counts from the probe kernel's own tokenizer pass (`data/token_lengths_train.csv`, `_dev.csv`), cross-checked row by row against an independent local computation (`data/length_stats_local.json`, `data/length_histogram.csv`): identical for all 1115 train rows. Prompt = chat-template-rendered user turn + `<think></think>` assistant prefix; completion = target JSON + 1 EOS. Tokenizer class TokenizersBackend, len(tok) 125033, EOS id 124900.

| Split | Rows | Prompt min/p50/p90/p99/max | Completion min/p50/p90/p99/max | Total min/p50/p90/p99/max | Over 7000 | Kept |
|---|---|---|---|---|---|---|
| train | 1115 | 503/1442/5064/5923/7073 | 6/355/1323/2255/2973 | 509/1787/6402/7894/8813 | 54 | 1061 |
| dev | 75 | 510/1282/5032/5549/5549 | 6/152/1047/2010/2010 | 516/1288/6081/7549/7549 | 1 | 74 |

- Kept train: 2,584,193 tokens total (mean 2435.6/row), 484,312 supervised (18.7%). Kept dev: 155,195 tokens, 26,157 supervised.
- Kept-train length buckets: <1000: 271; 1000-3000: 447; 3000-6000: 249; 6000-7000: 94. Rows >=6500 tokens: 48 (4.5% of kept); >=6900: 2.
- Loss is completion-only; one row per micro-step, 8 micro-steps per optimizer step; no padding across rows, no packing, no truncation.
- Row ids are synthetic classes (V2-short/medium/long/empty-NNNN) plus scenario ids. About 7% of train rows have an empty-list target.
- Probe sample: 96 train rows drawn by length bucket (23/38/21/8 for <1k/1-3k/3-6k/6-7k) plus the 6 longest kept rows forced in (indices 482, 1077, 938, 704, 366, 415). Sample mean 2732.7 tokens/row vs 2435.6 population mean, so the probe over-represents long rows (+12% tokens/row).
- Dev is split from the same synthetic generator (related scenario family), so it is NOT an independent holdout. There is no synthetic test set. The independent holdout is 80 real fixtures, which are private and appear in this packet only as aggregate numbers quoted in QUALITY.md.

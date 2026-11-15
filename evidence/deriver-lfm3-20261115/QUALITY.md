# QUALITY

Read this first: the probe measured 7 dev rows after 12 steps (8 effective). It is not an estimate of the full recipe's quality. Raw predictions incl. gold (synthetic) are in `raw/eval_baseline.jsonl` and `raw/eval_candidate.jsonl`.

## Setup
- Rows: the first 7 rows of dev (V2-short-0811, V2-medium-0102, V2-long-1605, V2-short-1009, V2-medium-0209, V2-long-0407, V2-short-1109; prompts 1306-5422 tokens; 163 gold observations). Baseline was cut to 7 rows by the 8-minute eval cap, so the same 7 rows were used for the candidate. Greedy decoding, max_new_tokens 1800.
- Baseline = same model with the adapter disabled. Candidate = adapter after 12 probe steps. Dev is synthetic and from the same generator as train (related, not independent).
- Score = the probe's own `score_row` (JSON parse, schema validity, Jaccard-style matching of observation strings), not `examples/deriver/validate.py`.

## Result (n=7)
| Metric | Baseline (adapter off) | Candidate (12 steps) |
|---|---|---|
| EOS emitted | 0 / 7 | 7 / 7 |
| schema_valid rate | 0.0 | 1.0 |
| hard_fail (unparseable) rate | 1.0 | 0.0 |
| predicted observations | 0 | 0 |
| gold observations | 163 | 163 |
| precision / recall / F1 | 0 / 0 / 0 | 0 / 0 / 0 |
| mean generated tokens | 1800 (cap) | 6 |
| mean seconds per row | 73.2 (~24.6 tok/s) | 1.48 |

Interpretation, plainly: the baseline never terminates and loops text. The candidate learned the output format and EOS but collapsed to the empty answer `{"explicit": []}` on all 7 rows, so recall and F1 are 0. With a 4-step scaler-skip start, 12 steps and about 7% empty-target rows in train, an early empty-answer collapse is expected and says nothing definitive about what 399 steps would give. No claim of improvement over baseline in extraction quality is made from this probe; the only measured gain is format validity and termination.

## Earlier results (not produced by this packet)
- v1 (330-row dataset), dev 33 rows: schema-valid 1.0, Jaccard P 0.5619 / R 0.5566. Source: `../../logs/kaggle/p2000-deriver-lfm3-lora/dev_metrics.json`. Different dataset and size; not comparable to the numbers above.
- Aggregate results on 80 real fixtures (independent holdout; private, rows never leave the lab, aggregates quoted from lab notes and NOT reproducible from this packet): v1 tuned LFM3 valid 0.98, hard-fail 0.04, F1 0.15, hallucination 0.01; untuned LFM3 base F1 ~0.08 with looping; best untuned alternative model F1 0.26-0.28.
- A related pruner run's evidence is already in the repo: `../../logs/kaggle/p2000-pruner-lfm3-lora/`.

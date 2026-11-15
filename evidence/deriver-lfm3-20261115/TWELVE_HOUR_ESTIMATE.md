# TWELVE_HOUR_ESTIMATE

Everything on this page is an ESTIMATE extrapolated from a 12-step single-T4 probe. Inputs marked MEASURED are in TIMING.md.

## Inputs
- MEASURED steady state: 25.1 s/step, 626.8 nonpad tok/s (10 steps, length-biased sample, mean 2732.7 tok/row vs population 2435.6).
- ESTIMATE per-step cost on the real population: 8 rows x 2435.6 tok x (1 - 0.045 rows lost to OOM) / 626.8 tok/s = ~29.7 s. Range used: 25-30 s/step (25 = measured direct, 30 = token-scaled). Long rows may cost more per token (attention), which this probe did not isolate, so the upper end may be optimistic.
- MEASURED model load across runs of the same weights: 576 s, 767 s, 1468 s, 5394 s. Use 0.2-1.5 h.
- Full recipe: 399 steps (133 x 3 epochs). About 48 of 1061 kept rows (>=6500 tok) are expected to OOM-skip every epoch (ESTIMATE; the probe lost 10 of 96 in a long-biased sample).
- Dev eval, 74 rows: 1.5 s/row (collapsed model, MEASURED) to 73 s/row (looping model, MEASURED) = 2 min to 90 min. A working model's cost is unmeasured and likely lies in between.
- Checkpoint save 0.3 s per epoch (MEASURED). Export/merge: NOT measured. Reserve >=30 min for the final save.

## Full recipe on one T4 (ESTIMATE)
| | Low | High |
|---|---|---|
| Load | 0.2 h | 1.5 h |
| Training, 399 steps | 2.8 h (25 s) | 3.3 h (30 s); allow up to 3.6 h |
| Per epoch | 55 min | 67 min |
| Dev eval | 0.03 h | 1.5 h |
| Total | ~3.0 h | ~6.6 h |
This fits both the script's 10.4 h budget and a 12 h session with margin, provided the OOM-skip guard works as in the probe. The probe script's guard is not the same file as the pushed v3 (IDENTITY), and v3 has never completed a run.

## What 12 h would buy (ESTIMATE)
Steps that fit = (budget - 30 min reserve - load) / s per step.
- 12 h: (43200 - 1800 - 5400)/30 = ~1200 steps (9.0 epochs, worst case) up to (43200 - 1800 - 576)/25.1 = ~1625 steps (12.2 epochs, best case).
- Within the script's own 10.4 h budget: ~1000 steps (7.6 epochs) up to ~1390 steps (10.5 epochs).

## Two T4s (ESTIMATE ONLY, UNMEASURED)
GPU1 did not train in the probe; no DDP code path exists in the scripts. If data-parallel were added, throughput could at best approach 2x (about 1.5-2 h for 3 epochs) before communication overhead, but each GPU would still hold a full copy of the model plus activations, so the ~14.3 GiB peak and the >=6500-token OOM would not go away. Lowering MAX_LEN to ~6400 (dropping ~48 rows, 4.5%) or reducing activation memory are the untested levers for that.

## Not covered
Real-fixture quality, a full epoch of measured time, export/merge, checkpoint resume, and any run of the pushed v3 script.

# botok-training

Training logs, speed measurements and sanitized SYNTHETIC examples for small LoRA specialist models (Honcho deriver, tool-output pruner, judge). Purpose: let an external assistant read the logs, estimate T4 training speed, and generate a large synthetic training set that fits a 12 h Kaggle session. No model weights and no real/private text are in this repo.

## Map
- `speed.md` - measured seconds/step, tokens/s, wall times, T4 settings, and a formula for how many tokens fit in 12 h.
- `logs/kaggle/<kernel>/` - run logs (`*.log` are Kaggle JSON log streams), `dev_metrics.json`, `train_log.json`, kernel metadata. `p2000-deriver-lfm3-v2a/run1_oom.log` is the failed first v2 run.
- `results/` - deriver leaderboard (10+ candidate models), tuned deriver/pruner summaries and the production gate (`summary.md`), per-run score lines.
- `examples/deriver/` - task spec `SCHEMA.md`, 13 synthetic windows (`examples.jsonl` SFT format, `examples_raw.jsonl` raw format), `validate.py`. `examples/pruner/`, `examples/judge/` - task descriptions + invented illustrative rows.
- `code/` - training script (`train_deriver_v2.py`), prompt, SFT builder.

## Constraints
Kaggle free T4 (14.5 GiB, fp16 AMP), internet off, 12 h session; plan = 3 epochs per part, so each part's data must fit ~10.4 h of training at a conservative ~210 tok/s (see speed.md). Rows longer than 7000 tokens are skipped. Target inference host: a 5 GiB Quadro P2000; LFM3-1.7B Q4_K_M is ~4.2 GiB including KV cache. The idea is small LoRA adapters so several specialist copies can stay resident; this is a plan, not verified for multiple copies.

## What to produce
- JSONL of deriver windows in the raw format of `examples/deriver/examples_raw.jsonl` (preferred; convert with `code/build_sft_v2.py`) or directly the SFT format. All data fully invented: no real people, firms, places, emails.
- Mix: ~15% empty gold; classes short/medium/long about 38/30/23 (+9% trivial empty); ~30% targeting an AI/bot peer; length distribution per `examples/deriver/SCHEMA.md` (median row ~1.6k tokens, p90 ~5.8k, max 7k incl. 450-token static prompt). Prefer MORE long windows (3-5.5k message tokens, 30-100 observations): real traffic is long and v1 failed on it.
- Token budget: about 3-4 M tokens (~1,000-1,500 rows) per training part; more parts allowed. Use the formula in speed.md.
- Validate: `python3 -I examples/deriver/validate.py yourfile.jsonl` must print OK. Also provide pruner rows (~17% none, ~10% all) and judge rows following the READMEs if asked.

## Honest lessons
- v1: 330 synthetic examples (short windows) -> valid JSON 98% but F1 0.15 / recall 0.03 on real long windows: distribution mismatch. v2 adds longer, denser windows (1115 train rows).
- v2 first run OOMed at step ~5 (full 7k-token logits); fixed with `logits_to_keep`; a rerun is in progress and its output is not in this repo yet.
- Base LFM3-1.7B without tuning: 100% malformed on the pruner KEEP format; 61% schema-valid, F1 0.08 on the deriver. Of ~13 untuned small models, none gets near the F1 >= 0.70 gate.
- Pruner training stopped by the time budget at 827 of 996 steps; loading the base model from a Kaggle dataset can take 12-90 min.
- The older synthetic windows (written by a Haiku-class teacher model) partly fail the current stricter validator (~7%).

# Deriver LoRA (LFM3-1.7B-Base) - Kaggle probe evidence packet, 2026-11-15

A 28-minute probe on 2x Tesla T4 (only GPU0 used) of the deriver LoRA recipe, with SYNTHETIC data only, plus the failure history of the earlier runs and a labelled extrapolation to a 12 h budget. Nothing here is a trained-model result: the probe ran 12 steps (8 effective) on 96 sampled rows.

## Headline (single T4)
- Steady state 25.1 s/step (steps 3-12); 627 non-padding tok/s; 108.5 supervised tok/s.
- GPU0 peak 14307 MiB allocated (14891 MiB nvidia-smi), util 96.7% in training. GPU1 idle: did NOT train.
- 10/96 rows (6500-6946 tokens) OOM-ed and were skipped; 4/12 steps were skipped by GradScaler.
- Dev (7 rows): baseline loops to the token cap, schema-valid 0; candidate schema-valid 1.0 but outputs the empty answer on every row, F1 0.
- Resume NOT verified (OOM in the check).
- Full run ESTIMATE: ~3.0-6.6 h on one T4 (2.8-3.6 h of that is training).

## Index
| File | What |
|---|---|
| IDENTITY.md | kernel, code, model, data hashes; repo-script discrepancy |
| DATA_SHAPES.md | token-length distributions, sampling |
| HARDWARE.md | GPUs, per-GPU memory/util, GPU1 idle |
| TIMING.md | phases, per-step table, steady state |
| RELIABILITY.md | OOM history, scaler skips, resume check failure |
| QUALITY.md | 7-row baseline vs candidate, earlier aggregates |
| TWELVE_HOUR_ESTIMATE.md | extrapolation, labelled ESTIMATE |
| code/ | probe script, v3 script as pushed, builder, prompt, redacted kernel metadata |
| raw/ | probe kernel log, per-step log, GPU samples, eval predictions, probe summary, both v2a OOM logs |
| artifacts/ | env manifest, input hashes, adapter hashes/config, trainer_state equivalent (custom loop, not HF Trainer) |
| data/ | per-row token lengths (train/dev), histograms |

Related evidence already in the repo: `../../logs/kaggle/p2000-deriver-lfm3-lora/` (v1), `../../logs/kaggle/p2000-deriver-lfm3-v2a/run1_oom.log`, `../../logs/kaggle/p2000-pruner-lfm3-lora/`.

## Known omissions
- No model weights or adapter weights (hashes only); the adapter was deleted in-kernel.
- No data files; hashes and token lengths only. Real fixtures and any private conversations are excluded entirely; only aggregate numbers are quoted, and those are not reproducible from this packet.
- No rendered notebook: Kaggle's `__results__.html` was not retrievable; `raw/probe_kernel_log.txt` is the converted stdout/stderr stream instead.
- Not measured: export/merge, a full epoch, resume from checkpoint, DDP / second GPU, cold weight read separate from model load, CPU/RAM specs, a real-fixture holdout, and any run of the pushed v3 script.
- Run 1/2 of v2a used an older script version that cannot be recovered exactly; base model hub revision and the Kaggle dataset version number are not recorded (file hashes are).
- Redacted: Kaggle account name (`<kaggle-user>`), GPU UUIDs. `kernel-metadata` files and `adapter_config.json` are redacted copies, so their hashes differ from the originals (originals listed in IDENTITY.md).
- `logs/INDEX.md` in this repo still lists v2a as "rerun queued"; it is stale and was not edited here.

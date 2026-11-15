# Goals

## Project goal
A self-sufficient, local memory-extraction stack for the Honcho memory service, running on a small 5 GiB Quadro P2000. No cloud or external model is used at inference time.

Roles (all small specialist LoRA-tuned models of ONE shared small base, LFM3-1.7B):
- **Deriver**: reads a window of chat messages plus a target peer and extracts explicit, atomic observations about that person as strict JSON.
- **Pruner**: reads a tool output plus the current goal and says which numbered lines to keep.
- **Judge / decision role**: scores a candidate extraction against a gold list (matched / missed / hallucinated / verdict).
- **Embeddings**: retrieval vectors for the stored observations.

Plan (not yet verified): LFM3-1.7B at Q4_K_M is about 4.2 GiB including KV cache, so several LoRA specialists of the same base can stay resident on the P2000 without swapping big models.

## Training setup
Free Kaggle T4 sessions: 12 h cap per session, private datasets, internet off, SYNTHETIC data only (no private text is ever uploaded). Each training part should finish inside one 12 h session (see `speed.md`).

## Deriver acceptance gate
Measured on 80 real held-out fixtures (never published):
- schema-valid >= 99%
- hard-fail <= 1%
- F1 vs gold >= 0.70
- hallucination <= 5%
- p95 latency <= 60 s

## Status (15 Nov 2026)
- No P2000-sized model passes the gate yet.
- Best untuned model: gemma4-e2b, F1 0.28, but it cannot coexist with embeddings in 5 GiB.
- Tuned LFM3 v1: F1 0.15, trained on only 330 synthetic examples (short windows); it fails on long real windows (distribution mismatch).
- v2 dataset: 1115 train / 75 dev rows, about 2.7 M tokens, about 15% empty-gold. First v2a run OOMed (fixed with `logits_to_keep` + `expandable_segments`, log `logs/kaggle/p2000-deriver-lfm3-v2a/run1_oom.log`); the rerun is in progress.
- Pruner v1: trained 827 of 996 steps (time budget); see `results/summary.md`.
- Judge and embeddings: planned, no model trained yet.

## Role contracts (short)
- Deriver: input = message window + target peer id; output = `{"explicit":[{"content":"..."}]}`, only facts the target stated. Spec: `examples/deriver/SCHEMA.md`, validator `examples/deriver/validate.py`.
- Pruner: input = `GOAL: ...` + numbered tool-output lines; output = ` KEEP 12-18,40` / ` KEEP none` / ` KEEP all`. Spec: `examples/pruner/README.md`.
- Judge: input = target, GOLD list, CANDIDATE list; output = `{"matched":[],"missed":[],"hallucinated":[],"verdict":"pass|fail"}`. Spec: `examples/judge/README.md`.

## Ask for the data generator (Dot)
- Realistic LONG windows: 3-7k tokens, many observations (30-100) per window.
- Varied topics, speakers, writing styles; AI/bot targets too.
- Empty-gold cases (about 15%) where the target states nothing extractable.
- Validator-clean: `python3 -I examples/deriver/validate.py file.jsonl` prints OK.
- Fully invented content only.
- Size each part to fit one 12 h session (3-4 M tokens per part; formula in `speed.md`).

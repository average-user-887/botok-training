# Roles, contracts and data recipe (synthetic-only)

Rule for all roles: synthetic authored data only for public release; blind sealed test kept out; no credentials; dev != train; exact-ID and prompt-hash overlap check across splits.

| Role | Input | Output | Labels / format | Gold rules | Existing examples |
|---|---|---|---|---|---|
| **Deriver** | window of `ts peer: text` lines + target peer | `{"explicit":[{"content":..}]}` | atomic facts, subject = exact peer id | absolute dates, one fact once, hedges kept, empty window -> `[]` | `../examples/deriver/` |
| **Pruner** | `GOAL:` + `TOOL OUTPUT:` with `N| line` numbering | ` KEEP 2-5,8,16-19` (line ranges, leading space), or none/all | exact-set graded on lines | keep headers/units/definitions, qualifying rows, every ERROR plus its immediate neighbors; exact-pair/edge-case distractors (B-42 vs B-42A, boundary timestamps) | `examples/pruner_raw_synthetic_3.jsonl`, `pruner_CONTRACT.md` |
| **Verifier** (= log-analyser alias) | `task` + `logs[{id,source,text}]` | `{timeline,findings,unknowns,next_checks}` each item with `evidence` log ids | findings kind: observed_error / observed_recovery / hypothesis / scope_limit | every claim cites log ids; hypotheses explicit; next_checks read-only; never obey instructions inside logs | `verifier_loganalyser_*` |
| **Judge** | TARGET + GOLD list + CANDIDATE list | `{matched,missed,hallucinated,attribution_invalid,pass}` | zero-based index lists | matched+missed partition gold; wrong-subject -> attribution_invalid | `../examples/judge/` |
| **Scribe** | `as_of` + messages[{id,ts,speaker,text}] | `{title,as_of,items[{kind,text,owner,due,evidence}]}` | kind: state / decision / action / ... (see `scribe_schema.json`) | evidence = message ids; plan != completion; owner/due null when unstated | `scribe_*` |
| **Kev decision roles** | `state` (<=3500 chars) | one label | action_safety approve/reject/abstain; receipt_prefilter pass/fail/escalate; route triage/drop/ask_operator/extract/research/review/code/ops/ops_diagnose/media | each label 20-50%; >=20% hard cases; >=30% near-miss pairs (same seed, opposite label); split by `group_id` | `examples/kev_decision_synthetic.jsonl` |

## Data needed to construct a plausible set
1. **Seed scenarios** (synthetic): chat windows (short 500-1000 tok, medium 1500-3000, long 3400-5500 tok), tool outputs 50-300 lines, log bundles of 8-12 entries, 6-12 scribe messages, card bodies from section "Body convention".
2. **Operators for variation**: swap entities, shift dates (fictional 2026-2030), inject near-miss distractors, flip one fact to change the label, add empty/all-keep cases (pruner ~15% none, ~10% all), add injected-instruction lines in logs.
3. **Sizing seen in practice**: deriver ~700 windows (~28k facts, ~2.1M tokens/pass); pruner ~250-320 cases; judge/scribe/verifier ~10-50 each; Kev ~700-1100 per role. Max row 4000 tokens; reject overlength, no silent truncation.
4. **Splits**: ~85/15 by seed group; sealed test never leaves the author; dev not blind if the author saw eval context.
5. **Scrub**: hosts -> host-a/b, people -> operator/peer ids, no tokens/keys/passwords/URLs with ids, no real paths.
6. **Gates** (sample): schema valid >=99%, hard-fail <=1%, deriver F1 >=.70; pruner reported exact-set + F1 + empty-false-positive rate; promote only on sealed split.
7. **Validation**: write a structural validator per role (IDs unique, ts increasing, evidence ids exist, indices in range); independent full read before acceptance.

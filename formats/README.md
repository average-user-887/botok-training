# Formats, schemas and data recipe (sanitized, synthetic)

Everything here is synthetic or scrubbed: hosts are `host-a/host-b`, people are `operator`, dates are fictional, no credentials, no real log text. Use it to construct plausible training rows; do not treat it as real lab data.

| File | What |
|---|---|
| `HONCHO_FORMATS.md` | Honcho peer card, session/messages, conclusions, deriver in/out |
| `KANBAN_CARD_FORMAT.md` + `examples/kanban_cards_synthetic.jsonl` | Hermes board card body, runs, comments, result summary |
| `AGENT_COMMS_SCHEMA.md` + `examples/agent_comms_synthetic.jsonl` | Inter-agent mailbox messages (hardline) and role routing |
| `ROLES_AND_DATA_RECIPE.md` | Per-role input/output contract, label sets, sizes, splits, gates, how to synthesize |
| `pruner_CONTRACT.md`, `examples/pruner_raw_synthetic_3.jsonl` | Pruner contract + 3 raw prompt/completion rows |
| `verifier_loganalyser_SCHEMA.md`, `examples/verifier_loganalyser_synthetic_3.jsonl` | Verifier (= log-analyser alias) contract + 3 rows |
| `scribe_TASK.md`, `scribe_schema.json`, `examples/scribe_synthetic_3.jsonl` | Scribe task, JSON schema, 3 rows |
| `examples/kev_decision_synthetic.jsonl` | Kev decision-role rows (action_safety, receipt_prefilter, route) |

Deriver, judge and pruner example sets already live in `../examples/`.

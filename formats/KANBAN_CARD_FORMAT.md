# Kanban card format (Hermes board, synthetic)

## Task object
`id` (`t_` + 8 hex), `title`, `status` (triage|todo|ready|running|review|blocked|scheduled|done|archived), `assignee` (profile, e.g. `labops`, or null in triage), `priority` int, `created_by` (`ops-cycle`, `mcp:claude`, ...), `workspace_path`, `body` (markdown/plain), `parents[]`, `children[]`, `runs[]`, `comments[]`.
Run: `{id, profile, status, outcome (completed|failed|blocked|timeout), summary, error}`. Comment: `{author, body}`.

## Body convention (ops/diagnose cards)
1. One-line trigger: `Fleet health probe at <ts> reported: <WARN ...>`
2. `Evidence collected on <host> ... (command: ...)` followed by the raw command output.
3. Mode line: `Diagnose it (L1: observe only, never change anything).`
4. Optional JSON snapshot (truncated) of all hosts.

## Result summary convention
`## Diagnosis: <title> — L1 observe-only` / `### Evidence` (fenced raw output) / `### Findings` (numbered, each tied to evidence) / `### Root cause` (distinguish observed vs inferred) / `### Recommendations` (read-only first, no changes made). A later comment repeats the summary; runs carry the same text.

## Rules to copy into synthetic cards
- Submission goes to triage and needs manual release; assignee optional.
- Workers must not act beyond the stated level (L1 = observe only).
- Referencing a prior card id (`t_xxxxxxxx`) is normal for recurring alerts.
- Include both a clean pass and a failed/blocked run among training cards (~20% failed/blocked).
See `examples/kanban_cards_synthetic.jsonl`.
